import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import copy
import logging
import pathlib
import platform
import random
import sys
import time

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from loader import binary_class
from loss import DiceLoss_binary, IoU_binary
from networks.mslau_net import MSLAU_net

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'mslau_net3.12-1.log'), level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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
        os.path.join(BASE_DIR, 'pretrained', 'best.pth'),
        os.path.join(os.path.dirname(BASE_DIR), 'pretrained', 'best.pth'),
    ]
    pretrained_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if pretrained_path is None:
        logging.warning('Pretrained checkpoint not found. Checked: %s', candidate_paths)
        return

    logging.info('Loading pretrained encoder from %s', pretrained_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
    model_dict = model.encoder.state_dict()
    full_dict = copy.deepcopy(pretrained_dict['model'])

    for key in list(full_dict.keys()):
        if key in model_dict and full_dict[key].shape != model_dict[key].shape:
            logging.info(
                'Skip pretrained param %s due to shape mismatch: %s vs %s',
                key, full_dict[key].shape, model_dict[key].shape
            )
            del full_dict[key]

    msg = model.encoder.load_state_dict(full_dict, strict=False)
    logging.info('Pretrained load result: %s', msg)


def train_model(model, criterion, optimizer, scheduler, dataloaders, accuracy_metric, num_epochs=5):
    since = time.time()

    loss_list = {'train': [], 'valid': []}
    accuracy_list = {'train': [], 'valid': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            running_loss = []
            running_corrects = []

            for inputs, labels, image_id in dataloaders[phase]:
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                labels = labels.float()
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    score = accuracy_metric(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss.append(loss.item())
                running_corrects.append(score.item())

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)

            logging.info('{} Loss: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            loss_list[phase].append(epoch_loss)
            accuracy_list[phase].append(epoch_acc)

            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                logging.info('Current learning rate : %f', optimizer.param_groups[0]['lr'])
                scheduler.step()

        print()

    best_iou = accuracy_list['valid'][best_epoch]
    save_name = f'best_model_{best_loss:.6f}_epoch_{best_epoch}_{best_iou:.6f}.pth'
    torch.save(best_model_wts, os.path.join(BASE_DIR, 'save_models', save_name))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, loss_list, accuracy_list


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CVC_ClinicDB', help='the path of images')
    parser.add_argument('--csvfile', type=str, default='src/CVC_ClinicDB/test_train_data.csv',
                        help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoches')
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE_DIR, 'save_models'), exist_ok=True)

    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(BASE_DIR, args.dataset)
    csv_path = args.csvfile if os.path.isabs(args.csvfile) else os.path.join(BASE_DIR, args.csvfile)

    df = pd.read_csv(csv_path)
    df = df[df.category == 'train']
    df.reset_index(drop=True, inplace=True)

    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold

    fold = 0
    val_files = list(df[df.fold == fold].image_id)
    print(val_files)
    print(len(val_files))
    train_files = list(df[df.fold != fold].image_id)
    print(train_files)
    print(len(train_files))

    train_dataset = binary_class(dataset_path, train_files, get_train_transform())
    val_dataset = binary_class(dataset_path, val_files, get_valid_transform())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=8
    )
    val_batch_size = max(1, args.batch // 4)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False, num_workers=8
    )
    dataloaders = {'train': train_loader, 'valid': val_loader}

    model_ft = MSLAU_net(img_size=256, mla_channels=64, in_chans=3, num_classes=1)
    load_encoder_pretrained(model_ft)
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    if args.loss == 'ce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice':
        criterion = DiceLoss_binary()
    else:
        raise ValueError(f'Unsupported loss type: {args.loss}')

    accuracy_metric = IoU_binary()
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, args.epoch, eta_min=0, last_epoch=-1)
    model_ft, Loss_list, Accuracy_list = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, accuracy_metric, num_epochs=args.epoch
    )
