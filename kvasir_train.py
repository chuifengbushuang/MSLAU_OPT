import argparse
import copy
import json
import logging
import os
import pathlib
import platform
import random
import sys
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from loader import binary_class
from loss import BCEDiceLoss_binary, DiceLoss_binary, IoU_binary
from networks.mslau_net import MSLAU_net

if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "save_models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "kvasir_train-3.13.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def parse_int_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(int(item) for item in parts)


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


def make_safe_name(value):
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "run"


def parse_position_tuple(value):
    if value is None:
        return None
    positions = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if not positions:
        raise argparse.ArgumentTypeError("Expected a comma-separated edge guidance position list.")
    allowed = {"none", "decoder", "pre_concat", "stage1", "stage2", "stage3", "stage4"}
    unique_positions = []
    for position in positions:
        if position not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported edge guidance position: {position}")
        if position == "none":
            if len(positions) > 1:
                raise argparse.ArgumentTypeError("'none' cannot be combined with other positions.")
            return tuple()
        if position not in unique_positions:
            unique_positions.append(position)
    return tuple(unique_positions)


def format_positions(positions):
    return ",".join(positions) if positions else "none"


def resolve_edge_guidance_positions(args):
    if args.edge_guidance_positions is not None:
        return parse_position_tuple(args.edge_guidance_positions)
    if args.edge_guidance_position is not None:
        return parse_position_tuple(args.edge_guidance_position)
    return ("decoder",) if args.edge_guidance_enabled else tuple()


def prepare_run_dir(args, edge_guidance_positions):
    if args.run_name:
        run_name = make_safe_name(args.run_name)
        run_dir = os.path.join(SAVE_DIR, "runs", run_name)
        if os.path.exists(run_dir):
            raise ValueError(f"Run directory already exists: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = make_safe_name(f"kvasir_eg_{format_positions(edge_guidance_positions)}_{timestamp}")
        run_dir = os.path.join(SAVE_DIR, "runs", run_name)

    os.makedirs(run_dir, exist_ok=False)
    return run_name, run_dir


def add_run_file_logger(run_dir):
    log_path = os.path.join(run_dir, "train.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return log_path


def write_run_config(args, run_dir, edge_guidance_positions, paths, seed):
    config = {
        "args": vars(args),
        "edge_guidance_position": format_positions(edge_guidance_positions),
        "edge_guidance_positions": list(edge_guidance_positions),
        "seed": seed,
        "paths": paths,
        "run_dir": run_dir,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False, default=str)
    return config_path


def load_ids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]
    return [sample_id if os.path.splitext(sample_id)[1] else f"{sample_id}.jpg" for sample_id in ids]


def get_train_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(shift_limit=0, p=0.25),
            A.CoarseDropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_encoder_pretrained(model):
    candidate_paths = [
        os.path.join(BASE_DIR, "pretrained", "best.pth"),
        os.path.join(os.path.dirname(BASE_DIR), "pretrained", "best.pth"),
    ]
    pretrained_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if pretrained_path is None:
        logging.warning("Pretrained checkpoint not found. Checked: %s", candidate_paths)
        return

    logging.info("Loading pretrained encoder from %s", pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
    model_dict = model.encoder.state_dict()
    full_dict = copy.deepcopy(pretrained_dict["model"])

    for key in list(full_dict.keys()):
        if key in model_dict and full_dict[key].shape != model_dict[key].shape:
            logging.info(
                "Skip pretrained param %s due to shape mismatch: %s vs %s",
                key,
                full_dict[key].shape,
                model_dict[key].shape,
            )
            del full_dict[key]

    msg = model.encoder.load_state_dict(full_dict, strict=False)
    logging.info("Pretrained load result: %s", msg)


def train_model(model, criterion, optimizer, scheduler, dataloaders, metric, num_epochs, save_dir):
    since = time.time()
    best_loss = float("inf")
    best_loss_epoch = 0
    best_loss_model_wts = copy.deepcopy(model.state_dict())
    best_iou = float("-inf")
    best_iou_epoch = 0
    best_iou_model_wts = copy.deepcopy(model.state_dict())

    loss_list = {"train": [], "valid": []}
    accuracy_list = {"train": [], "valid": []}

    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train(True)
            else:
                model.eval()

            running_loss = []
            running_corrects = []

            for inputs, labels, _ in dataloaders[phase]:
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                labels = labels.float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    score = metric(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss.append(loss.item())
                running_corrects.append(score.item())

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)

            logging.info("{} Loss: {:.4f} IoU: {:.4f}".format(phase, epoch_loss, epoch_acc))

            loss_list[phase].append(epoch_loss)
            accuracy_list[phase].append(epoch_acc)

            if phase == "valid" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_loss_epoch = epoch
                best_loss_model_wts = copy.deepcopy(model.state_dict())

            if phase == "valid" and epoch_acc >= best_iou:
                best_iou = epoch_acc
                best_iou_epoch = epoch
                best_iou_model_wts = copy.deepcopy(model.state_dict())

            if phase == "train":
                logging.info("Current learning rate : %f", optimizer.param_groups[0]["lr"])
                scheduler.step()

        print()

    best_loss_iou = accuracy_list["valid"][best_loss_epoch]
    best_iou_loss = loss_list["valid"][best_iou_epoch]
    last_model_wts = copy.deepcopy(model.state_dict())
    loss_path = os.path.join(
        save_dir,
        f"best_loss_{best_loss:.6f}_epoch_{best_loss_epoch}_{best_loss_iou:.6f}.pth",
    )
    iou_path = os.path.join(
        save_dir,
        f"best_iou_{best_iou:.6f}_epoch_{best_iou_epoch}_{best_iou_loss:.6f}.pth",
    )
    stable_path = os.path.join(save_dir, "kvasir_best_model.pth")
    last_path = os.path.join(save_dir, "last.pth")
    torch.save(best_loss_model_wts, loss_path)
    torch.save(best_iou_model_wts, iou_path)
    torch.save(best_iou_model_wts, stable_path)
    torch.save(last_model_wts, last_path)

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val loss: %.6f at epoch %d", best_loss, best_loss_epoch)
    logging.info("Best val IoU: %.6f at epoch %d", best_iou, best_iou_epoch)
    logging.info("Saved best-loss checkpoint: %s", loss_path)
    logging.info("Saved best-IoU checkpoint: %s", iou_path)
    logging.info("Saved fixed checkpoint: %s", stable_path)
    logging.info("Saved last checkpoint: %s", last_path)

    model.load_state_dict(best_iou_model_wts)
    return model, loss_list, accuracy_list


if __name__ == "__main__":
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Kvasir-SEG", help="dataset directory")
    parser.add_argument("--train_txt", type=str, default="Kvasir-SEG/train.txt", help="training ids txt")
    parser.add_argument("--val_txt", type=str, default="Kvasir-SEG/val.txt", help="validation ids txt")
    parser.add_argument("--loss", default="bce_dice", choices=["ce", "dice", "bce_dice"], help="loss type")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE weight used by bce_dice loss")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice weight used by bce_dice loss")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--run_name", type=str, default=None, help="unique experiment name under save_models/runs")
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
    parser.add_argument(
        "--edge_guidance_position",
        default=None,
        choices=["none", "decoder", "pre_concat", "stage1", "stage2", "stage3", "stage4"],
        help="where to apply edge guidance; omitted keeps --edge_guidance_enabled compatibility",
    )
    parser.add_argument(
        "--edge_guidance_positions",
        default=None,
        help="comma-separated edge guidance positions, e.g. stage4,decoder; overrides --edge_guidance_position",
    )
    args = parser.parse_args()

    if len(args.gfe_crossformer_group_sizes) != 2:
        raise ValueError("--gfe_crossformer_group_sizes must contain exactly 2 integers.")
    if len(args.gfe_crossformer_intervals) != 2:
        raise ValueError("--gfe_crossformer_intervals must contain exactly 2 integers.")
    if args.bce_weight < 0 or args.dice_weight < 0 or args.bce_weight + args.dice_weight <= 0:
        raise ValueError("--bce_weight and --dice_weight must be non-negative and cannot both be zero.")

    edge_guidance_positions = resolve_edge_guidance_positions(args)
    edge_guidance_position_text = format_positions(edge_guidance_positions)
    run_name, run_dir = prepare_run_dir(args, edge_guidance_positions)
    run_log_path = add_run_file_logger(run_dir)
    logging.info("Run name: %s", run_name)
    logging.info("Run directory: %s", run_dir)
    logging.info("Run log: %s", run_log_path)

    dataset_path = resolve_path(args.dataset)
    train_txt = resolve_path(args.train_txt)
    val_txt = resolve_path(args.val_txt)
    config_path = write_run_config(
        args,
        run_dir,
        edge_guidance_positions,
        {
            "dataset": dataset_path,
            "train_txt": train_txt,
            "val_txt": val_txt,
            "shared_log": os.path.join(LOG_DIR, "kvasir_train-3.13.log"),
            "run_log": run_log_path,
        },
        seed,
    )
    logging.info("Saved run config: %s", config_path)

    train_files = load_ids(train_txt)
    val_files = load_ids(val_txt)
    print(train_files)
    print(len(train_files))
    print(val_files)
    print(len(val_files))

    train_dataset = binary_class(dataset_path, train_files, get_train_transform())
    val_dataset = binary_class(dataset_path, val_files, get_valid_transform())

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=24,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=max(1, args.batch // 2),
        shuffle=False,
        drop_last=False,
        num_workers=24,
        pin_memory=torch.cuda.is_available(),
    )
    dataloaders = {"train": train_loader, "valid": val_loader}

    model = MSLAU_net(
        img_size=256,
        mla_channels=64,
        in_chans=3,
        num_classes=1,
        gfe_attn_type=args.gfe_attn_type,
        gfe_crossformer_group_sizes=args.gfe_crossformer_group_sizes,
        gfe_crossformer_intervals=args.gfe_crossformer_intervals,
        gfe_crossformer_adaptive_interval=args.gfe_crossformer_adaptive_interval,
        edge_guidance_enabled=bool(edge_guidance_positions),
        edge_guidance_positions=edge_guidance_positions,
        linear_attn_type=args.linear_attn_type,
    )
    logging.info(
        "Model config: type=%s, linear_attn_type=%s, group_sizes=%s, intervals=%s, adaptive_interval=%s, edge_guidance_enabled=%s, edge_guidance_positions=%s",
        args.gfe_attn_type,
        args.linear_attn_type,
        args.gfe_crossformer_group_sizes,
        args.gfe_crossformer_intervals,
        args.gfe_crossformer_adaptive_interval,
        bool(edge_guidance_positions),
        edge_guidance_position_text,
    )
    logging.info(
        "Loss config: loss=%s, bce_weight=%.3f, dice_weight=%.3f",
        args.loss,
        args.bce_weight,
        args.dice_weight,
    )
    load_encoder_pretrained(model)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss == "ce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss_binary()
    elif args.loss == "bce_dice":
        criterion = BCEDiceLoss_binary(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    metric = IoU_binary()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=0, last_epoch=-1)
    train_model(model, criterion, optimizer, scheduler, dataloaders, metric, num_epochs=args.epoch, save_dir=run_dir)
