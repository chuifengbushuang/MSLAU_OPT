import argparse
import copy
import logging
import os
import pathlib
import platform
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
from loss import DiceLoss_binary, IoU_binary
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


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


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


def train_model(model, criterion, optimizer, scheduler, dataloaders, metric, num_epochs):
    since = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

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
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "train":
                logging.info("Current learning rate : %f", optimizer.param_groups[0]["lr"])
                scheduler.step()

        print()

    best_iou = accuracy_list["valid"][best_epoch]
    best_name = f"best_model_{best_loss:.6f}_epoch_{best_epoch}_{best_iou:.6f}.pth"
    dynamic_path = os.path.join(SAVE_DIR, best_name)
    stable_path = os.path.join(SAVE_DIR, "kvasir_best_model.pth")
    torch.save(best_model_wts, dynamic_path)
    torch.save(best_model_wts, stable_path)

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val loss: {:4f}".format(best_loss))
    logging.info("Saved checkpoint: %s", dynamic_path)
    logging.info("Saved fixed checkpoint: %s", stable_path)

    model.load_state_dict(best_model_wts)
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
    parser.add_argument("--loss", default="dice", help="loss type")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    train_txt = resolve_path(args.train_txt)
    val_txt = resolve_path(args.val_txt)

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
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=max(1, args.batch // 2),
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    dataloaders = {"train": train_loader, "valid": val_loader}

    model = MSLAU_net(img_size=256, mla_channels=64, in_chans=3, num_classes=1)
    load_encoder_pretrained(model)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss == "ce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss_binary()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    metric = IoU_binary()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=0, last_epoch=-1)
    train_model(model, criterion, optimizer, scheduler, dataloaders, metric, num_epochs=args.epoch)
