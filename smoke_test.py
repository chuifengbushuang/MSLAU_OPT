import argparse
import json

import torch

from kvasir_train import get_train_transform, load_encoder_pretrained, load_ids
from loader import binary_class
from loss import DiceLoss_binary
from networks.mslau_net import MSLAU_net


def main():
    parser = argparse.ArgumentParser(description="Run one MSLAU-Net training smoke-test batch.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_ids = load_ids(args.split)
    dataset = binary_class(args.dataset, image_ids[:1], get_train_transform())
    image, mask, image_id = dataset[0]
    image = image.unsqueeze(0).float().to(device)
    mask = mask.unsqueeze(0).unsqueeze(0).float().to(device)

    model = MSLAU_net(img_size=256, mla_channels=64, in_chans=3, num_classes=1)
    load_encoder_pretrained(model, args.pretrained)
    model = model.to(device)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = model(image)
    loss = DiceLoss_binary()(output, mask)
    loss.backward()

    result = {
        "status": "ok",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "sample": image_id,
        "input_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "output_shape": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if device.type == "cuda"
            else None
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
