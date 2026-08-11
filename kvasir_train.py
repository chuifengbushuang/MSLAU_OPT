import argparse
import copy
import logging
import os
import pathlib
import platform
import random
import sys
import time

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from loader import binary_class
from loss import (
    BCEDiceLoss_binary,
    BCEDiceLovaszLoss_binary,
    DiceLoss_binary,
    IoU_binary,
    StructureLoss_binary,
)
from networks.mslau_net import MSLAU_net

if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def configure_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "kvasir_train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def load_ids(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        ids = [line.strip() for line in file if line.strip()]
    return [sample_id if os.path.splitext(sample_id)[1] else f"{sample_id}.jpg" for sample_id in ids]


def get_train_transform(img_size=256):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(shift_limit=0, p=0.25),
            A.CoarseDropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transform(img_size=256):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_encoder_pretrained(model, requested_path=None):
    candidate_paths = [resolve_path(requested_path)] if requested_path else [
        os.path.join(BASE_DIR, "pretrained", "best.pth")
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


FUSION_ROUTE_NAMES = (
    "stage1_shallow",
    "stage2",
    "stage3",
    "stage4_deep",
)


def log_fusion_scale_stats(model, epoch):
    fusion_logits = getattr(model.conv_mla, "fusion_logits", None)
    if fusion_logits is None:
        return

    with torch.no_grad():
        gamma = 4.0 * torch.softmax(fusion_logits.detach(), dim=0)
        for route_index, route_name in enumerate(FUSION_ROUTE_NAMES):
            route_gamma = gamma[route_index]
            logging.info(
                "Fusion gamma epoch=%d route=%s mean=%.6f std=%.6f "
                "min=%.6f max=%.6f deviation_l1=%.6f",
                epoch, route_name,
                route_gamma.mean().item(), route_gamma.std().item(),
                route_gamma.min().item(), route_gamma.max().item(),
                (route_gamma - 1.0).abs().mean().item(),
            )
        logging.info(
            "Fusion gamma epoch=%d sum_max_error=%.8e",
            epoch, (gamma.sum(dim=0) - 4.0).abs().max().item(),
        )


def make_boundary_target(labels):
    dilated = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-labels, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def snapshot_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def atomic_save_state_dict(state_dict, path):
    temporary_path = path + ".tmp"
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, path)


class DINOv2FeatureDistiller(nn.Module):
    """Training-only multi-level feature distillation from frozen DINOv2."""

    def __init__(self, teacher_name="vit_small_patch14_dinov2",
                 student_channels=(64, 128, 256, 512), input_size=364,
                 checkpoint=None):
        super().__init__()
        import timm

        self.teacher_name = teacher_name
        self.input_size = input_size
        self.layer_indices = (2, 5, 8, 11)
        self.teacher = timm.create_model(
            teacher_name,
            pretrained=checkpoint is None,
            dynamic_img_size=True,
        )
        if checkpoint is not None:
            teacher_state = torch.load(
                checkpoint, map_location="cpu", weights_only=False)
            teacher_state.pop("mask_token", None)
            self.teacher.load_state_dict(teacher_state, strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        teacher_channels = self.teacher.embed_dim
        self.adapters = nn.ModuleList([
            nn.Conv2d(channels, teacher_channels, kernel_size=1)
            for channels in student_channels
        ])

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def adapter_parameters(self):
        return self.adapters.parameters()

    def forward(self, normalized_images, student_features):
        teacher_inputs = F.interpolate(
            normalized_images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            teacher_features = self.teacher.get_intermediate_layers(
                teacher_inputs,
                n=self.layer_indices,
                reshape=True,
                norm=True,
            )

        losses = []
        for adapter, student_feature, teacher_feature in zip(
                self.adapters, student_features, teacher_features):
            student_feature = adapter(student_feature)
            student_feature = F.interpolate(
                student_feature,
                size=teacher_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            student_feature = F.normalize(student_feature, dim=1)
            teacher_feature = F.normalize(teacher_feature.detach(), dim=1)
            losses.append(1.0 - (student_feature * teacher_feature).sum(dim=1).mean())
        return torch.stack(losses).mean()


def train_model(model, criterion, optimizer, scheduler, dataloaders, metric,
                num_epochs, save_dir, aux_criterion=None, aux_d1_weight=0.0,
                aux_d2_weight=0.2, aux_d3_weight=0.1, boundary_weight=0.1,
                distiller=None, distill_weight=0.0):
    since = time.time()
    best_loss = float("inf")
    best_loss_epoch = 0
    best_loss_model_wts = snapshot_state_dict(model)
    best_iou = float("-inf")
    best_iou_epoch = 0
    best_iou_model_wts = snapshot_state_dict(model)
    stable_path = os.path.join(save_dir, "kvasir_best_model.pth")

    loss_list = {"train": [], "valid": []}
    accuracy_list = {"train": [], "valid": []}

    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train(True)
                if distiller is not None:
                    distiller.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_main_loss = 0.0
            running_iou = 0.0
            running_distill_loss = 0.0
            running_samples = 0

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
                    model_outputs = model(
                        inputs,
                        return_aux=model.decoder_mode != "legacy",
                        return_features=distiller is not None and phase == "train",
                    )
                    if isinstance(model_outputs, dict):
                        outputs = model_outputs["logits"]
                        main_loss = criterion(outputs, labels)
                        supervision_criterion = aux_criterion or criterion
                        aux_d1_loss = (
                            supervision_criterion(model_outputs["aux_d1"], labels)
                            if "aux_d1" in model_outputs else outputs.new_zeros(())
                        )
                        aux_d2_loss = (
                            supervision_criterion(model_outputs["aux_d2"], labels)
                            if "aux_d2" in model_outputs else outputs.new_zeros(())
                        )
                        aux_d3_loss = (
                            supervision_criterion(model_outputs["aux_d3"], labels)
                            if "aux_d3" in model_outputs else outputs.new_zeros(())
                        )
                        boundary_loss = (
                            supervision_criterion(
                                model_outputs["boundary_logits"],
                                make_boundary_target(labels),
                            )
                            if boundary_weight > 0 and "boundary_logits" in model_outputs
                            else outputs.new_zeros(())
                        )
                        loss = (
                            main_loss
                            + aux_d1_weight * aux_d1_loss
                            + aux_d2_weight * aux_d2_loss
                            + aux_d3_weight * aux_d3_loss
                            + boundary_weight * boundary_loss
                        )
                    else:
                        outputs = model_outputs
                        main_loss = criterion(outputs, labels)
                        loss = main_loss
                    distill_loss = outputs.new_zeros(())
                    if distiller is not None and phase == "train":
                        distill_loss = distiller(
                            inputs, model_outputs["encoder_features"])
                        loss = loss + distill_weight * distill_loss
                    score = metric(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_samples = inputs.size(0)
                running_loss += loss.item() * batch_samples
                running_main_loss += main_loss.item() * batch_samples
                running_iou += score.item() * batch_samples
                running_distill_loss += distill_loss.item() * batch_samples
                running_samples += batch_samples

            epoch_loss = running_loss / running_samples
            epoch_main_loss = running_main_loss / running_samples
            epoch_acc = running_iou / running_samples
            epoch_distill_loss = running_distill_loss / running_samples

            logging.info(
                "{} Loss: {:.4f} MainLoss: {:.4f} DistillLoss: {:.4f} IoU: {:.4f}".format(
                    phase, epoch_loss, epoch_main_loss,
                    epoch_distill_loss, epoch_acc))

            loss_list[phase].append(epoch_main_loss)
            accuracy_list[phase].append(epoch_acc)

            if phase == "valid" and epoch_main_loss <= best_loss:
                best_loss = epoch_main_loss
                best_loss_epoch = epoch
                best_loss_model_wts = snapshot_state_dict(model)

            if phase == "valid" and epoch_acc >= best_iou:
                best_iou = epoch_acc
                best_iou_epoch = epoch
                best_iou_model_wts = snapshot_state_dict(model)
                atomic_save_state_dict(best_iou_model_wts, stable_path)
                logging.info(
                    "Saved running best-IoU checkpoint: %s", stable_path)

            if phase == "train":
                learning_rates = ", ".join(
                    "{}={:.8e}".format(
                        group.get("name", "group_{}".format(index)),
                        group["lr"],
                    )
                    for index, group in enumerate(optimizer.param_groups)
                )
                logging.info("Current learning rates: %s", learning_rates)
                scheduler.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            log_fusion_scale_stats(model, epoch)

        print()

    best_loss_iou = accuracy_list["valid"][best_loss_epoch]
    best_iou_loss = loss_list["valid"][best_iou_epoch]
    best_loss_path = os.path.join(
        save_dir,
        f"best_loss_{best_loss:.6f}_epoch_{best_loss_epoch}_{best_loss_iou:.6f}.pth",
    )
    best_iou_path = os.path.join(
        save_dir,
        f"best_iou_{best_iou:.6f}_epoch_{best_iou_epoch}_{best_iou_loss:.6f}.pth",
    )
    torch.save(best_loss_model_wts, best_loss_path)
    torch.save(best_iou_model_wts, best_iou_path)
    atomic_save_state_dict(best_iou_model_wts, stable_path)

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val loss: %.6f at epoch %d", best_loss, best_loss_epoch)
    logging.info("Best val IoU: %.6f at epoch %d", best_iou, best_iou_epoch)
    logging.info("Saved best-loss checkpoint: %s", best_loss_path)
    logging.info("Saved best-IoU checkpoint: %s", best_iou_path)
    logging.info("Saved fixed best-IoU checkpoint: %s", stable_path)

    model.load_state_dict(best_iou_model_wts)
    return model, loss_list, accuracy_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Kvasir-SEG", help="dataset directory")
    parser.add_argument("--train_txt", type=str, default="Kvasir-SEG/train.txt", help="training ids txt")
    parser.add_argument("--val_txt", type=str, default="Kvasir-SEG/val.txt", help="validation ids txt")
    parser.add_argument("--pretrained", type=str, default=None, help="encoder pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="save_models/kvasir", help="checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs/kvasir", help="log directory")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--img_size", type=int, default=256, help="square input size")
    parser.add_argument(
        "--loss", default="bce_dice",
        choices=["ce", "dice", "bce_dice", "bce_dice_lovasz", "structure"],
        help="final-mask loss type",
    )
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE weight in bce_dice loss")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice weight in bce_dice loss")
    parser.add_argument("--lovasz_weight", type=float, default=0.3,
                        help="Lovasz weight in bce_dice_lovasz loss")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--lff_lr",
        type=float,
        default=5e-3,
        help="initial learning rate for LFF fusion parameters",
    )
    parser.add_argument("--decoder_dropout", type=float, default=0.0, help="Dropout2d after MLA feature concatenation")
    parser.add_argument(
        "--decoder_mode",
        default="legacy",
        choices=[
            "legacy", "progressive_wavelet", "cascade_reverse",
            "p5_hiera_reverse", "p5_sam2unet"],
        help="legacy, P3/P4, legacy P5 hybrid, or official-style SAM2-UNet",
    )
    parser.add_argument(
        "--encoder_name",
        default="mslau",
        choices=["mslau", "sam2_hiera_large"],
        help="feature encoder; P5 uses a frozen SAM2 Hiera-L backbone",
    )
    parser.add_argument(
        "--hiera_checkpoint",
        type=str,
        default=None,
        help="local pretrained SAM2 Hiera checkpoint used to initialize P5",
    )
    parser.add_argument("--progressive_channels", type=int, default=96,
                        help="feature channels in the P3 progressive decoder")
    parser.add_argument("--disable_p3_wavelet_edge", action="store_true",
                        help="disable the P3 wavelet edge multiplication gate")
    parser.add_argument("--disable_p3_reverse_attention", action="store_true",
                        help="disable the P3 reverse-attention multiplication gate")
    parser.add_argument("--aux_d2_weight", type=float, default=0.2,
                        help="P3 D2 auxiliary segmentation loss weight")
    parser.add_argument("--aux_d1_weight", type=float, default=0.1,
                        help="P4 D1 auxiliary segmentation loss weight")
    parser.add_argument("--aux_d3_weight", type=float, default=0.1,
                        help="P3 coarse D3 segmentation loss weight")
    parser.add_argument("--boundary_weight", type=float, default=0.1,
                        help="P3 boundary supervision loss weight")
    parser.add_argument("--encoder_lr", type=float, default=5e-5,
                        help="P3 encoder learning rate")
    parser.add_argument("--decoder_lr", type=float, default=2e-4,
                        help="P3 decoder learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="linear warmup epochs before cosine decay")
    parser.add_argument(
        "--distill_teacher", default="none",
        choices=["none", "vit_small_patch14_dinov2"],
        help="training-only frozen feature teacher",
    )
    parser.add_argument("--distill_weight", type=float, default=0.0,
                        help="DINOv2 feature-distillation loss weight")
    parser.add_argument("--dino_input_size", type=int, default=364,
                        help="DINOv2 teacher input size; must be divisible by 14")
    parser.add_argument(
        "--dino_checkpoint", type=str, default=None,
        help="local DINOv2 teacher checkpoint (avoids runtime download)",
    )
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--disable_edge_guidance",
        action="store_true",
        help="disable P2 decoder edge guidance and run the P0-only model",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="fixed",
        choices=["fixed", "lff_scale"],
        help="feature fusion mode in Conv_MLA",
    )
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    dataset_path = resolve_path(args.dataset)
    train_txt = resolve_path(args.train_txt)
    val_txt = resolve_path(args.val_txt)
    output_dir = resolve_path(args.output_dir)
    log_dir = resolve_path(args.log_dir)
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(log_dir)
    logging.info("Random seed: %d", seed)

    train_files = load_ids(train_txt)
    val_files = load_ids(val_txt)
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    train_dataset = binary_class(
        dataset_path, train_files, get_train_transform(args.img_size))
    val_dataset = binary_class(
        dataset_path, val_files, get_valid_transform(args.img_size))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=max(1, args.batch // 2),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dataloaders = {"train": train_loader, "valid": val_loader}

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
        encoder_name=args.encoder_name,
        hiera_checkpoint=(
            resolve_path(args.hiera_checkpoint) if args.hiera_checkpoint else None),
    )
    logging.info("Encoder: %s", model.encoder_name)
    logging.info("Decoder mode: %s", model.decoder_mode)
    logging.info("Edge guidance enabled: %s", model.edge_guidance_enabled)
    logging.info("Fusion mode: %s", model.fusion_mode)
    logging.info("Decoder Dropout2d: %.3f", model.decoder_dropout.p)
    logging.info("Input size: %d", args.img_size)
    if model.decoder_mode == "progressive_wavelet":
        logging.info(
            "P3 configuration: channels=%d aux_d2=%.3f aux_d3=%.3f "
            "boundary=%.3f wavelet_edge=%s reverse_attention=%s",
            args.progressive_channels, args.aux_d2_weight,
            args.aux_d3_weight, args.boundary_weight,
            not args.disable_p3_wavelet_edge,
            not args.disable_p3_reverse_attention,
        )
    elif model.decoder_mode == "cascade_reverse":
        logging.info(
            "P4 configuration: channels=%d aux_d1=%.3f aux_d2=%.3f "
            "aux_d3=%.3f boundary=%.3f",
            args.progressive_channels, args.aux_d1_weight,
            args.aux_d2_weight, args.aux_d3_weight, args.boundary_weight,
        )
    elif model.decoder_mode == "p5_hiera_reverse":
        if not args.hiera_checkpoint:
            raise ValueError("P5 requires --hiera_checkpoint with pretrained Hiera-L weights")
        if not os.path.exists(resolve_path(args.hiera_checkpoint)):
            raise FileNotFoundError(
                "Hiera checkpoint not found: {}".format(
                    resolve_path(args.hiera_checkpoint)))
        logging.info(
            "P5 configuration: frozen_hiera=%s channels=%d aux_d2=%.3f "
            "aux_d3=%.3f boundary=%.3f wavelet_edge=False reverse_attention=True",
            model.encoder.freeze_backbone, args.progressive_channels,
            args.aux_d2_weight, args.aux_d3_weight, args.boundary_weight,
        )
    elif model.decoder_mode == "p5_sam2unet":
        if not args.hiera_checkpoint:
            raise ValueError("P5 SAM2-UNet requires --hiera_checkpoint")
        if not os.path.exists(resolve_path(args.hiera_checkpoint)):
            raise FileNotFoundError(
                "Hiera checkpoint not found: {}".format(
                    resolve_path(args.hiera_checkpoint)))
        logging.info(
            "P5 SAM2-UNet configuration: adapter=block_prompt bottleneck=32 "
            "backbone_frozen=%s rfb_channels=64 aux_d2=%.3f aux_d3=%.3f",
            model.encoder.freeze_backbone,
            args.aux_d2_weight, args.aux_d3_weight,
        )
    if args.encoder_name == "mslau":
        load_encoder_pretrained(model, args.pretrained)
    elif args.pretrained:
        raise ValueError("--pretrained is only valid for the original MSLAU encoder")
    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss == "ce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss_binary()
    elif args.loss == "bce_dice":
        criterion = BCEDiceLoss_binary(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )
    elif args.loss == "bce_dice_lovasz":
        criterion = BCEDiceLovaszLoss_binary(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            lovasz_weight=args.lovasz_weight,
        )
    elif args.loss == "structure":
        criterion = StructureLoss_binary()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    aux_criterion = (
        criterion if args.loss == "structure"
        else BCEDiceLoss_binary(bce_weight=0.5, dice_weight=0.5))
    metric = IoU_binary()
    distiller = None
    if args.distill_teacher != "none":
        if model.decoder_mode != "cascade_reverse":
            raise ValueError("DINOv2 distillation is currently supported for P4 only")
        if args.distill_weight <= 0:
            raise ValueError("distill_weight must be positive when a teacher is enabled")
        if args.dino_input_size % 14 != 0:
            raise ValueError("dino_input_size must be divisible by DINOv2 patch size 14")
        logging.info(
            "Loading frozen distillation teacher: %s input=%d weight=%.4f checkpoint=%s",
            args.distill_teacher, args.dino_input_size, args.distill_weight,
            args.dino_checkpoint,
        )
        dino_checkpoint = (
            resolve_path(args.dino_checkpoint) if args.dino_checkpoint else None)
        if dino_checkpoint is not None and not os.path.exists(dino_checkpoint):
            raise FileNotFoundError(
                "DINOv2 checkpoint not found: {}".format(dino_checkpoint))
        distiller = DINOv2FeatureDistiller(
            teacher_name=args.distill_teacher,
            input_size=args.dino_input_size,
            checkpoint=dino_checkpoint,
        )
        if torch.cuda.is_available():
            distiller = distiller.cuda()
    fusion_logits = getattr(model.conv_mla, "fusion_logits", None)
    if model.decoder_mode != "legacy":
        encoder_params = [
            param for param in model.encoder.parameters() if param.requires_grad
        ]
        decoder_params = [
            param for name, param in model.named_parameters()
            if not name.startswith("encoder.")
        ]
        parameter_groups = [
            {"params": encoder_params, "lr": args.encoder_lr,
             "name": "hiera_adapters" if args.encoder_name != "mslau" else "encoder"},
            {"params": decoder_params, "lr": args.decoder_lr,
             "name": "{}_decoder".format(model.decoder_mode)},
        ]
        logging.info(
            "Parameters: total=%d trainable=%d encoder_trainable=%d",
            sum(param.numel() for param in model.parameters()),
            sum(param.numel() for param in model.parameters() if param.requires_grad),
            sum(param.numel() for param in encoder_params),
        )
        if distiller is not None:
            parameter_groups.append({
                "params": distiller.adapter_parameters(),
                "lr": args.decoder_lr,
                "name": "distill_adapters",
            })
        optimizer = optim.AdamW(
            parameter_groups,
            weight_decay=5e-4,
        )
    elif fusion_logits is None:
        optimizer = optim.AdamW(
            [{"params": model.parameters(), "lr": args.lr, "name": "base"}],
            weight_decay=5e-4,
        )
    else:
        lff_param_ids = {id(fusion_logits)}
        base_params = [
            param for param in model.parameters()
            if id(param) not in lff_param_ids
        ]
        optimizer = optim.AdamW(
            [
                {"params": base_params, "lr": args.lr, "name": "base"},
                {"params": [fusion_logits], "lr": args.lff_lr, "name": "lff"},
            ],
            weight_decay=5e-4,
        )
    logging.info(
        "Initial learning rates: %s",
        ", ".join(
            "{}={:.8e}".format(group["name"], group["lr"])
            for group in optimizer.param_groups
        ),
    )
    if args.warmup_epochs > 0:
        if args.warmup_epochs >= args.epoch:
            raise ValueError("warmup_epochs must be smaller than epoch")
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.2, total_iters=args.warmup_epochs)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoch - args.warmup_epochs, eta_min=0)
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch, eta_min=0, last_epoch=-1)
    train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        metric,
        num_epochs=args.epoch,
        save_dir=output_dir,
        aux_criterion=aux_criterion,
        aux_d1_weight=(
            args.aux_d1_weight if model.decoder_mode == "cascade_reverse" else 0.0),
        aux_d2_weight=args.aux_d2_weight,
        aux_d3_weight=args.aux_d3_weight,
        boundary_weight=args.boundary_weight,
        distiller=distiller,
        distill_weight=args.distill_weight,
    )
