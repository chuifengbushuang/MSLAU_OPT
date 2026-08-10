import argparse
import os
import pathlib
import platform
import time

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


def load_ids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]
    return [sample_id if os.path.splitext(sample_id)[1] else f"{sample_id}.jpg" for sample_id in ids]


def get_transform(img_size=256):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Kvasir-SEG", type=str, help="dataset directory")
    parser.add_argument("--split_txt", default="Kvasir-SEG/val.txt", type=str, help="evaluation ids txt")
    parser.add_argument(
        "--model",
        default="save_models/kvasir/kvasir_best_model.pth",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument("--batch", default=8, type=int, help="batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="dataloader workers")
    parser.add_argument("--img_size", default=256, type=int, help="square input size")
    parser.add_argument("--threshold", default=0.5, type=float, help="binarization threshold")
    parser.add_argument("--debug", default=False, type=str2bool, help="save predicted masks")
    parser.add_argument("--debug_dir", default="debug_kvasir", type=str, help="debug output directory")
    parser.add_argument("--disable_edge_guidance", action="store_true", help="evaluate the P0-only model")
    parser.add_argument(
        "--fusion_mode",
        default="fixed",
        choices=["fixed", "lff_scale"],
        help="feature fusion mode used by the checkpoint",
    )
    parser.add_argument("--decoder_dropout", default=0.0, type=float,
                        help="Dropout2d probability used during training")
    parser.add_argument(
        "--decoder_mode",
        default="legacy",
        choices=["legacy", "progressive_wavelet", "cascade_reverse"],
        help="decoder architecture used by the checkpoint",
    )
    parser.add_argument("--progressive_channels", default=96, type=int,
                        help="feature channels in the P3 progressive decoder")
    parser.add_argument("--disable_p3_wavelet_edge", action="store_true",
                        help="disable the P3 wavelet edge multiplication gate")
    parser.add_argument("--disable_p3_reverse_attention", action="store_true",
                        help="disable the P3 reverse-attention multiplication gate")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    split_txt = resolve_path(args.split_txt)
    model_path = resolve_path(args.model)
    debug_dir = resolve_path(args.debug_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not os.path.exists(split_txt):
        raise FileNotFoundError(f"Split txt not found: {split_txt}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    eval_files = load_ids(split_txt)
    print(f"Using checkpoint: {model_path}")
    print(f"Testing samples: {len(eval_files)}")

    eval_dataset = binary_class(
        dataset_path, eval_files, get_transform(args.img_size))
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSLAU_net(
        img_size=args.img_size,
        mla_channels=64,
        in_chans=3,
        num_classes=1,
        edge_guidance_enabled=not args.disable_edge_guidance,
        fusion_mode=args.fusion_mode,
        decoder_dropout=args.decoder_dropout,
        decoder_mode=args.decoder_mode,
        progressive_channels=args.progressive_channels,
        p3_use_wavelet_edges=not args.disable_p3_wavelet_edge,
        p3_use_reverse_attention=not args.disable_p3_reverse_attention,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    metrics = {
        "iou": [],
        "dice": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    time_cost = []

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
            logits = model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).float()
            batch_metrics = compute_batch_metrics(preds, masks)
            for name, values in batch_metrics.items():
                metrics[name].extend(values)

            per_image_time = (end - start) / max(1, imgs.shape[0])
            time_cost.extend([per_image_time] * imgs.shape[0])

            if args.debug:
                save_debug_images(debug_dir, dataset_path, image_ids, preds, masks)

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    if time_cost:
        print("FPS: {:.2f}".format(1.0 / (sum(time_cost) / len(time_cost))))
    print("mean IoU:", round(np.mean(metrics["iou"]), 4), round(np.std(metrics["iou"]), 4))
    print("mean dice:", round(np.mean(metrics["dice"]), 4), round(np.std(metrics["dice"]), 4))
    print("mean accuracy:", round(np.mean(metrics["accuracy"]), 4), round(np.std(metrics["accuracy"]), 4))
    print("mean precision:", round(np.mean(metrics["precision"]), 4), round(np.std(metrics["precision"]), 4))
    print("mean recall:", round(np.mean(metrics["recall"]), 4), round(np.std(metrics["recall"]), 4))
    print("mean F1-score:", round(np.mean(metrics["f1"]), 4), round(np.std(metrics["f1"]), 4))
