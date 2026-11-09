"""Strict deterministic runtime for reproducible training/evaluation."""

import logging

import torch


def configure_runtime():
    """Enable strict deterministic algorithms (raises on unsupported ops).

    Call once, before building the model / data pipeline. The environment
    variables CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED and
    NO_ALBUMENTATIONS_UPDATE must be set before the Python process starts
    (see the launcher script); setting them here would be too late.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    logging.info(
        "Strict runtime enabled: use_deterministic_algorithms(True, "
        "warn_only=False); cuDNN deterministic; matmul TF32 off")
