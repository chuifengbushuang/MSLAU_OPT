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
import pandas as pd
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
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
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
        image_path = os.path.join(dataset_path, "images", image_id)
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{stem}.png"), image)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_pred.png"), pred_np[idx])
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_gt.png"), mask_np[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CVC_ClinicDB", type=str, help="dataset directory")
    parser.add_argument(
        "--csvfile",
        default="src/CVC_ClinicDB/test_train_data.csv",
        type=str,
        help="two columns [image_id, category(train/test)]",
    )
    parser.add_argument(
        "--model",
        default="save_models/best_model_0.052903_epoch_177_0.902547.pth",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument("--batch", default=8, type=int, help="batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="dataloader workers")
    parser.add_argument("--threshold", default=0.5, type=float, help="binarization threshold")
    parser.add_argument("--debug", default=False, type=str2bool, help="save predicted masks")
    parser.add_argument("--debug_dir", default="debug", type=str, help="debug output directory")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    csv_path = resolve_path(args.csvfile)
    model_path = resolve_path(args.model)
    debug_dir = resolve_path(args.debug_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    df = pd.read_csv(csv_path)
    df = df[df.category == "test"].reset_index(drop=True)
    test_files = list(df.image_id)

    print(f"Using checkpoint: {model_path}")
    print(f"Testing samples: {len(test_files)}")

    test_dataset = binary_class(dataset_path, test_files, get_transform())
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSLAU_net(img_size=256, mla_channels=64, in_chans=3, num_classes=1)
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
        for imgs, masks, image_ids in test_loader:
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
