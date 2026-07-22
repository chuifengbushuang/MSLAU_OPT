import argparse
import os
import pathlib
import platform
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from loader import binary_class
from networks.mslau_net import MSLAU_net

if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = "save_models/kvasir_best_model.pth"
LEGACY_MODEL_PATH = "save_models/kvasir/kvasir_best_model.pth"
OPTIONAL_STATE_PREFIXES = ("edge_guidance.",)


def parse_int_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(int(item) for item in parts)


def parse_float_tuple(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        return None
    return tuple(float(item) for item in parts)


def parse_path_list(value):
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def find_compatible_checkpoint(model, device, exclude_path=None):
    expected_keys = {
        key for key in model.state_dict().keys()
        if not key.startswith(OPTIONAL_STATE_PREFIXES)
    }
    for candidate in (DEFAULT_MODEL_PATH, LEGACY_MODEL_PATH):
        candidate_path = resolve_path(candidate)
        if exclude_path and os.path.normcase(candidate_path) == os.path.normcase(exclude_path):
            continue
        if not os.path.exists(candidate_path):
            continue

        candidate_state = extract_state_dict(torch.load(candidate_path, map_location=device, weights_only=False))
        checkpoint_keys = set(candidate_state.keys())
        if expected_keys.issubset(checkpoint_keys):
            return candidate_path
    return None


def load_checkpoint(model, model_path, device):
    state_dict = extract_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        expected_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing_keys = sorted(expected_keys - checkpoint_keys)

        details = [f"Checkpoint is incompatible with the current MSLAU_net: {model_path}"]
        if missing_keys:
            preview = ", ".join(missing_keys[:10])
            suffix = " ..." if len(missing_keys) > 10 else ""
            details.append(f"Missing keys: {preview}{suffix}")
        compatible_path = find_compatible_checkpoint(model, device, exclude_path=model_path)
        if compatible_path is not None:
            details.append(f"Use this compatible checkpoint instead: {compatible_path}")
        details.append("Legacy CEM-related checkpoint keys are ignored automatically.")
        raise RuntimeError(f"{exc}\n\n" + "\n".join(details)) from exc


def load_ids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]
    return [sample_id if os.path.splitext(sample_id)[1] else f"{sample_id}.jpg" for sample_id in ids]


def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_tta_ops(mode):
    ops = [(lambda x: x, lambda x: x)]
    if mode == "flip":
        ops.extend(
            [
                (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
                (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
                (lambda x: torch.flip(x, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
            ]
        )
    return ops


def load_models(model_paths, device, args):
    models = []
    for model_path in model_paths:
        model = MSLAU_net(
            img_size=256,
            mla_channels=64,
            in_chans=3,
            num_classes=1,
            gfe_attn_type=args.gfe_attn_type,
            gfe_crossformer_group_sizes=args.gfe_crossformer_group_sizes,
            gfe_crossformer_intervals=args.gfe_crossformer_intervals,
            gfe_crossformer_adaptive_interval=args.gfe_crossformer_adaptive_interval,
            edge_guidance_enabled=args.edge_guidance_enabled,
            linear_attn_type=args.linear_attn_type,
        )
        load_checkpoint(model, model_path, device)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models


def predict_probs(models, imgs, tta_mode):
    probs_sum = None
    count = 0
    for forward_aug, inverse_aug in get_tta_ops(tta_mode):
        aug_imgs = forward_aug(imgs)
        for model in models:
            logits = model(aug_imgs)
            probs = torch.sigmoid(inverse_aug(logits))
            probs_sum = probs if probs_sum is None else probs_sum + probs
            count += 1
    return probs_sum / count


def compute_batch_metrics(preds, masks, smooth=1.0):
    preds = preds.float().reshape(preds.shape[0], -1)
    masks = masks.float().reshape(masks.shape[0], -1)

    intersection = (preds * masks).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    mask_sum = masks.sum(dim=1)
    union = pred_sum + mask_sum - intersection

    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    accuracy = (preds == masks).float().mean(dim=1)
    precision = torch.where(
        pred_sum > 0,
        intersection / pred_sum.clamp_min(1e-7),
        torch.zeros_like(intersection),
    )
    recall = torch.where(
        mask_sum > 0,
        intersection / mask_sum.clamp_min(1e-7),
        torch.zeros_like(intersection),
    )
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall).clamp_min(1e-7),
        torch.zeros_like(precision),
    )

    return {
        "iou": iou.cpu().tolist(),
        "dice": dice.cpu().tolist(),
        "accuracy": accuracy.cpu().tolist(),
        "precision": precision.cpu().tolist(),
        "recall": recall.cpu().tolist(),
        "f1": f1.cpu().tolist(),
    }


def save_debug_images(debug_dir, dataset_path, image_ids, preds, masks):
    os.makedirs(debug_dir, exist_ok=True)
    pred_np = preds.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    mask_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255

    for idx, image_id in enumerate(image_ids):
        stem, _ = os.path.splitext(image_id)
        image = cv2.imread(os.path.join(dataset_path, "images", image_id))
        if image is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{stem}.jpg"), image)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_pred.png"), pred_np[idx])
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_gt.png"), mask_np[idx])


def remove_small_components(preds, min_area):
    preds_np = preds.squeeze(1).cpu().numpy().astype(np.uint8)
    filtered_masks = []

    for mask in preds_np:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered_mask = np.zeros_like(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == label] = 1
        filtered_masks.append(filtered_mask)

    filtered_masks = np.stack(filtered_masks, axis=0)[:, None, :, :]
    return torch.from_numpy(filtered_masks).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Kvasir-SEG", type=str, help="dataset directory")
    parser.add_argument("--split_txt", default="Kvasir-SEG/val.txt", type=str, help="evaluation ids txt")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        type=str,
        help="bestmodel checkpoint",
    )
    parser.add_argument(
        "--models",
        default=None,
        type=str,
        help="comma-separated checkpoint paths for ensemble",
    )
    parser.add_argument("--batch", default=8, type=int, help="batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="dataloader workers")
    parser.add_argument("--threshold", default=0.5, type=float, help="binarization threshold")
    parser.add_argument(
        "--thresholds",
        default=None,
        type=parse_float_tuple,
        help="comma-separated thresholds to scan, e.g. 0.4,0.45,0.5,0.55",
    )
    parser.add_argument("--tta", default="none", choices=["none", "flip"], help="test-time augmentation mode")
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=["none", "remove_small_components"],
        help="optional mask post-processing",
    )
    parser.add_argument(
        "--min_component_area",
        default=512,
        type=int,
        help="minimum connected-component area kept when postprocess is enabled",
    )
    parser.add_argument("--debug", default=False, type=str2bool, help="save predicted masks")
    parser.add_argument("--debug_dir", default="debug_kvasir", type=str, help="debug output directory")
    parser.add_argument(
        "--gfe_attn_type",
        type=str,
        default="msla",
        choices=["msla", "crossformer", "crossformer_lsda"],
        help="attention type used in GFE blocks",
    )
    parser.add_argument(
        "--linear_attn_type",
        type=str,
        default="legacy",
        choices=["legacy", "relu", "elu"],
        help="linear attention kernel used when --gfe_attn_type=msla",
    )
    parser.add_argument(
        "--gfe_crossformer_group_sizes",
        type=parse_int_tuple,
        default=(7, 7),
        help="comma-separated group sizes for CrossFormer attention, e.g. 7,7",
    )
    parser.add_argument(
        "--gfe_crossformer_intervals",
        type=parse_int_tuple,
        default=(8, 4),
        help="comma-separated intervals for CrossFormer attention, e.g. 8,4",
    )
    parser.add_argument(
        "--gfe_crossformer_adaptive_interval",
        action="store_true",
        help="enable adaptive interval for CrossFormer attention",
    )
    parser.add_argument(
        "--edge_guidance_enabled",
        default=False,
        type=str2bool,
        help="enable the low-risk MEGANet-style edge guidance block after MLAHead",
    )
    args = parser.parse_args()

    if len(args.gfe_crossformer_group_sizes) != 2:
        raise ValueError("--gfe_crossformer_group_sizes must contain exactly 2 integers.")
    if len(args.gfe_crossformer_intervals) != 2:
        raise ValueError("--gfe_crossformer_intervals must contain exactly 2 integers.")

    dataset_path = resolve_path(args.dataset)
    split_txt = resolve_path(args.split_txt)
    model_paths = [resolve_path(path) for path in parse_path_list(args.models)] if args.models else [resolve_path(args.model)]
    debug_dir = resolve_path(args.debug_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not os.path.exists(split_txt):
        raise FileNotFoundError(f"Split txt not found: {split_txt}")
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    eval_files = load_ids(split_txt)
    print(f"Using checkpoints ({len(model_paths)}):")
    for model_path in model_paths:
        print(model_path)
    print(f"Testing samples: {len(eval_files)}")

    eval_dataset = binary_class(dataset_path, eval_files, get_transform())
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(model_paths, device, args)
    threshold_candidates = list(args.thresholds) if args.thresholds else [args.threshold]
    threshold_candidates = [float(threshold) for threshold in threshold_candidates]
    threshold_results = {}
    time_cost = []
    all_probs = []
    all_masks = []
    all_image_ids = []

    since = time.time()
    with torch.no_grad():
        for imgs, masks, image_ids in eval_loader:
            imgs = imgs.float().to(device, non_blocking=True)
            masks = masks.float().to(device, non_blocking=True)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            probs = predict_probs(models, imgs, args.tta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            per_image_time = (end - start) / max(1, imgs.shape[0])
            time_cost.extend([per_image_time] * imgs.shape[0])
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())
            all_image_ids.extend(image_ids)

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    for threshold in threshold_candidates:
        preds = (all_probs >= threshold).float()
        if args.postprocess == "remove_small_components":
            preds = remove_small_components(preds, args.min_component_area)
        threshold_results[threshold] = compute_batch_metrics(preds, all_masks)

    best_threshold = max(
        threshold_results,
        key=lambda threshold: np.mean(threshold_results[threshold]["iou"]),
    )

    if args.debug:
        debug_preds = (all_probs >= best_threshold).float()
        save_debug_images(debug_dir, dataset_path, all_image_ids, debug_preds, all_masks)

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    if time_cost:
        print("FPS: {:.2f}".format(1.0 / (sum(time_cost) / len(time_cost))))
    print(f"TTA mode: {args.tta}")
    print(f"Postprocess: {args.postprocess}")
    if args.postprocess == "remove_small_components":
        print(f"Min component area: {args.min_component_area}")
    print(f"Ensemble size: {len(models)}")
    for threshold in threshold_candidates:
        metrics = threshold_results[threshold]
        print(
            "threshold {:.2f} | mean IoU: {:.4f} {:.4f} | mean dice: {:.4f} {:.4f} | mean accuracy: {:.4f} {:.4f}".format(
                threshold,
                np.mean(metrics["iou"]),
                np.std(metrics["iou"]),
                np.mean(metrics["dice"]),
                np.std(metrics["dice"]),
                np.mean(metrics["accuracy"]),
                np.std(metrics["accuracy"]),
            )
        )
    best_metrics = threshold_results[best_threshold]
    print("best threshold:", round(best_threshold, 4))
    print("best mean precision:", round(np.mean(best_metrics["precision"]), 4), round(np.std(best_metrics["precision"]), 4))
    print("best mean recall:", round(np.mean(best_metrics["recall"]), 4), round(np.std(best_metrics["recall"]), 4))
    print("best mean f1:", round(np.mean(best_metrics["f1"]), 4), round(np.std(best_metrics["f1"]), 4))
