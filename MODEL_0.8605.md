# Kvasir 0.8605 Edge-Guided MSLAU-Net

This branch preserves the locally saved Kvasir checkpoint:

```text
save_models/best_iou_0.860554_epoch_86_0.115706.pth
```

The checkpoint contains decoder-level `edge_guidance.*` weights and MSLA legacy
linear-attention weights. It does not contain ReLU/ELU linear-attention,
CrossFormer, stage edge-guidance, pre-concat edge-guidance, or LCM weights.

Key differences from the original first-commit baseline:

- MSLA token restoration is fixed with `transpose(1, 2).reshape(...)` before
  spatial depthwise convolutions.
- Decoder-level `EdgeGuidedAttention` is available through
  `--edge_guidance_enabled true`.
- `--linear_attn_type legacy|relu|elu` is selectable; the 0.8605 checkpoint uses
  `legacy`.
- `BCEDiceLoss_binary` is available and was used for the best run.
- Training saves both best-loss and best-IoU checkpoints.
- Kvasir testing supports threshold scans and flip TTA.

Recommended evaluation command:

```bash
python kvasir_test.py \
  --model save_models/best_iou_0.860554_epoch_86_0.115706.pth \
  --edge_guidance_enabled true \
  --linear_attn_type legacy \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55 \
  --tta flip
```

Recommended training command for the same model family:

```bash
python kvasir_train.py \
  --edge_guidance_enabled true \
  --linear_attn_type legacy \
  --loss bce_dice \
  --bce_weight 0.5 \
  --dice_weight 0.5 \
  --batch 32 \
  --epoch 200
```
