# V2CONT_HANDOVER — `kvasir-p5-mafa-v2-cont`

> 给接手 Agent 的说明文档。目标：在这条分支上理解、复现、继续开发 **V2-cont（MAFA-v2 适配器）** 这条线。
> 分支：`kvasir-p5-mafa-v2-cont`　提交：`4f0697e`　远程：`git@github.com:chuifengbushuang/MSLAU_OPT.git`
> PR：https://github.com/chuifengbushuang/MSLAU_OPT/pull/new/kvasir-p5-mafa-v2-cont

---

## 1. TL;DR

- 这条分支是 **V2-cont 的开发快照**：包含 `HieraBlockAdapterMAFAv2`（`adapter_mode=block_mafa_v2`）、对应 `decoder_mode=p5_sam2unet_mafa_v2`、训练/评测脚本、以及可选的 NAF 模块与 Kvasir split 配置。
- “V2-cont” = **两段式训练** 的 MAFA-v2：
  1. Stage 1：MAFA-v2 base，从零，lr `1e-3`，20 ep；
  2. Stage 2：从 Stage 1 最佳权重继续，lr `3e-4`，20 ep。
- 官方 880/120 划分下的干净结果（Kvasir-SEG val 120）：
  - Stage 1 best val IoU = **0.832684**（ep19）
  - Stage 2 best val IoU = **0.838153**（ep12）
  - TTA（hflip+vflip）@thr 0.50 = **0.8415**；@thr 0.55 = **0.8422**
- 注意：旧的 **0.8824** 是**旧随机划分**（旧 train 含 102 张官方 val 图）下的结果，**不能与 0.8382 直接比较**。

---

## 2. 分支里有什么 / 没有什么

### 已提交（commit `4f0697e`，16 个文件）
- `networks/mslau_net.py`：`HieraBlockAdapterMAFAv2`、`HieraBlockAdapterMAFAv2GN/AR`、`HieraBlockAdapterMAFAv3`、`HieraFeatureEncoder` 的 `block_mafa_v2` 映射、`SAM2UNetDecoder`、`MSLAU_net`。
- `kvasir_train.py` / `kvasir_test.py`：训练/评测入口（含 NAF 相关参数，见第 9 节）。
- `networks/naf_discrete.py`、`networks/naf_select_reg.py`：NAF 模块（**可选**，V2-cont 不需要）。
- `paired_inputs.py`、`strict_runtime.py`：`kvasir_train.py` 的依赖。
- `requirements.txt`。
- `configs/splits/kvasir_official_880_120/`：官方 880/120（train/val/test，test=val 镜像）。
- `configs/splits/kvasir_800_100_100_seed1234_20260722_1608/`：旧的 800/100/100 划分。

### 未提交（按约定排除，工作区仍在）
- `AGENTS.md`（已修改但未暂存）、`SERVER_USAGE.md`、`environment.lock.yml`、CVC split、`make_cvc_splits.py`。
- 3 个垃圾文件（**请勿提交**）：
  - `= torch.randn(2, 1, 16, 16, requires_grad=True)`
  - `s.py', 'kvasir_train.py'):`
  - `t`
- 训练权重/日志不在仓库内，位于服务器 `/data/models/zsy/mslau-net/runs/...`。

---

## 3. 环境与依赖

- Python 环境：`/home/zsy/miniconda3/envs/mslau-net/bin/python`（conda env `mslau-net`，Python 3.10）。
- 关键依赖：`torch`、`timm`、`albumentations`、`safetensors`、`torchmetrics`（以 `requirements.txt` 为准）。
- 预训练主干（Hiera-L）：
  `/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors`
- 数据：
  `/data/dataset/zsy/mslau-net/Kvasir-SEG/{images,masks}`（1000 张）。
- 显存：Hiera-L + MAFA-v2，`img=352, batch=12` 约需 15–16 GB；共享 GPU 做**推理**时可用 `batch=4`。

快速安装（若需要）：
```bash
cd /home/zsy/projects/mslau-net
pip install -r requirements.txt
```

---

## 4. 代码地图

| 文件 | 作用 |
|---|---|
| `networks/mslau_net.py` | 模型定义。关键类：`HieraBlockAdapterMAFAv2`、`HieraFeatureEncoder`、`SAM2UNetDecoder`、`MSLAU_net` |
| `kvasir_train.py` | 训练入口；`--decoder_mode`、`--adapter_mode`（由 decoder_mode 决定）、loss/aux/NAF 等参数 |
| `kvasir_test.py` | 评测入口；必须显式传入与训练一致的模型配置 |
| `loader.py` | `binary_class` 数据集加载 |
| `loss.py` | `StructureLoss_binary` 等 |
| `paired_inputs.py` / `strict_runtime.py` | 训练依赖（paired 采样 / 严格运行时） |
| `configs/splits/kvasir_official_880_120/` | 官方 880/120 划分 |
| `configs/splits/kvasir_800_100_100_seed1234_20260722_1608/` | 旧 800/100/100 划分 |

**适配器映射**（`HieraFeatureEncoder`）：
- `adapter_mode="block_mafa_v2"` → `HieraBlockAdapterMAFAv2(block, stage_adaptive=True, hidden=64)`，插在**每个 Hiera block 之前**。
- 对应 `decoder_mode="p5_sam2unet_mafa_v2"`（训练/评测都要一致）。

---

## 5. 模块定义（MAFA-v2 / MSA-HBA）

设 Hiera 第 `i` 个 block 的输入为 `X ∈ R^{B×H×W×C}`；适配器在通道前置布局下处理，输出残差增量 `Δ`，并送回原 block：

```text
y = Block_i( X + Δ )
```

内部流程（d=32, h=64）：
1. 下投影：`F = GELU(Conv1x1^{C→d}(X))`
2. 多尺度分支：`F_k = GELU(Conv_{k×k}^{d→d}(F))`，`k ∈ K_s`
   - `K_s = {1,3,5,7}` if `C ∈ {144,288}`
   - `K_s = {1,3}` if `C ∈ {576,1152}`
   - `n = |K_s|`（图中以 n=4 为例）
3. 双池化描述子：`g_k = [GAP(F_k); GMP(F_k)] ∈ R^{2d}`
4. 拼接 + 层归一化：`z = [g_1;…;g_n] ∈ R^{2nd}`，`z̃ = LN(z)`
5. 选择网络：`s = W2 · GELU(W1 · z̃)`，`W1∈R^{h×2nd}`, `W2∈R^{nd×h}`
6. 分支维 softmax：`S = reshape(s) ∈ R^{n×d}`，`α_{k,c} = softmax_k(S_{k,c})`
7. 加权融合：`F̂ = Σ_k α_k ⊙ F_k`
8. 上投影（**零初始化**）：`Δ = Conv1x1^{d→C}(F̂)`，权重/偏置初始化为 0
9. 回注：`y = Block_i(X + Δ)`

训练时主干冻结，仅适配器与 decoder 可训练；零初始化使模块在训练初始为恒等映射。

---

## 6. 数据划分

- **官方（推荐，干净口径）**：`configs/splits/kvasir_official_880_120/`
  - train 880 / val 120 / test 120（test 为 val 的镜像，官方只给 train/val）。
  - 来源：`DebeshJha/Kvasir-SEG/Data-split`。
- **旧 880/120（仓库里已跟踪的 `configs/splits/kvasir/`）**：随机 seed1234、val=test；**V2-cont 最初就是在这个旧划分上训练的**（旧 0.8824 的来源）。
- **旧 800/100/100**：`configs/splits/kvasir_800_100_100_seed1234_20260722_1608/`。

> 做正式对比请用**官方 880/120**；不要混用旧划分。

---

## 7. 复现 V2-cont（两段式）

### 7.1 Stage 1：MAFA-v2 base（from scratch，lr 1e-3）

```bash
cd /home/zsy/projects/mslau-net
PY=/home/zsy/miniconda3/envs/mslau-net/bin/python
SPLIT=configs/splits/kvasir_official_880_120
DATA=/data/dataset/zsy/mslau-net/Kvasir-SEG
HIERA=/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors
RUN=/data/models/zsy/mslau-net/runs/v2cont_repro

$PY -u kvasir_train.py \
  --dataset $DATA \
  --train_txt $SPLIT/train.txt \
  --val_txt $SPLIT/val.txt \
  --output_dir $RUN/stage1/checkpoints \
  --log_dir $RUN/stage1/logs \
  --decoder_mode p5_sam2unet_mafa_v2 \
  --encoder_name sam2_hiera_large \
  --hiera_checkpoint $HIERA \
  --loss structure \
  --img_size 352 \
  --batch 12 \
  --num_workers 8 \
  --epoch 20 \
  --seed 1234 \
  --encoder_lr 1e-3 \
  --decoder_lr 1e-3 \
  --aux_d2_weight 1.0 \
  --aux_d3_weight 1.0
```

### 7.2 Stage 2：V2-cont（init Stage 1，lr 3e-4）

```bash
S1=$RUN/stage1/checkpoints/kvasir_best_model.pth   # 不存在则用 best_iou_*.pth

$PY -u kvasir_train.py \
  --dataset $DATA \
  --train_txt $SPLIT/train.txt \
  --val_txt $SPLIT/val.txt \
  --output_dir $RUN/stage2/checkpoints \
  --log_dir $RUN/stage2/logs \
  --decoder_mode p5_sam2unet_mafa_v2 \
  --encoder_name sam2_hiera_large \
  --hiera_checkpoint $HIERA \
  --loss structure \
  --img_size 352 \
  --batch 12 \
  --num_workers 8 \
  --epoch 20 \
  --seed 1234 \
  --encoder_lr 3e-4 \
  --decoder_lr 3e-4 \
  --aux_d2_weight 1.0 \
  --aux_d3_weight 1.0 \
  --init_checkpoint $S1
```

说明：
- 两次都**没有**显式传 `--boundary_weight`，即使用默认值 `0.1`。
- `--aux_d2_weight 1.0 --aux_d3_weight 1.0`；`--loss structure`。
- Stage 1/2 各 20 ep；cosine 调度（脚本默认）。
- 产物：`kvasir_best_model.pth`（running best-IoU）、`best_iou_*.pth`、`best_loss_*.pth`。

### 7.3 已经跑过的那次（可直接用）
```
/data/models/zsy/mslau-net/runs/kvasir_official880_120_v2cont_2stage_e20e20_sam2unet_hiera_l_blockmafav2_rfb64_structure_352_b12_seed1234_gpu3_nw8_20261104_1209/
  stage1/checkpoints/best_iou_0.832684_epoch_19_0.427421.pth
  stage2/checkpoints/best_iou_0.838153_epoch_12_0.418480.pth   # V2-cont 最佳
```

---

## 8. 评测与 TTA

### 8.1 官方 val（base，thr=0.5）
```bash
$PY kvasir_test.py \
  --dataset $DATA \
  --test_txt $SPLIT/val.txt \
  --checkpoint .../stage2/checkpoints/best_iou_0.838153_epoch_12_0.418480.pth \
  --decoder_mode p5_sam2unet_mafa_v2 \
  --encoder_name sam2_hiera_large \
  --img_size 352
```
（`kvasir_test.py` 的参数名以脚本为准；务必与训练配置一致。）

### 8.2 TTA 结果（官方 val 120）

| variant | thr0.45 | thr0.50 | thr0.55 | thr0.60 |
|---|---:|---:|---:|---:|
| base（单次） | 83.73 | 83.81 | 83.87 | 83.89 |
| **vflip** | **84.16** | 84.14 | 84.11 | 84.07 |
| **hflip+vflip (hv)** | 84.01 | 84.15 | **84.22** | 84.13 |
| 多尺度 ms | 83.59 | 83.62 | 83.61 | 83.48 |
| 多尺度+v | 83.58 | 83.80 | 83.83 | 83.79 |
| 多尺度+hv | 83.87 | 83.95 | 84.05 | 83.95 |

- **推荐对外口径**：hv TTA @thr 0.50 = **0.8415**（固定阈值、无 val 调参）。
- 最高：hv @0.55 = **0.8422**（阈值在 val 上选，需注明）。
- **多尺度 TTA 在这套模型上会掉分**，不要用。
- base@0.5 = 0.8381，与训练报告的 `0.838153` 对齐，可作为评测正确性的 sanity check。

### 8.3 TTA 脚本（可直接用）
把下面保存为 `tta_eval.py`，修改 `CKPT` 即可：

```python
import sys, torch, numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
sys.path.insert(0, "/home/zsy/projects/mslau-net")
from loader import binary_class
from networks.mslau_net import MSLAU_net

DATA = "/data/dataset/zsy/mslau-net/Kvasir-SEG"
VAL = "/home/zsy/projects/mslau-net/configs/splits/kvasir_official_880_120/val.txt"
IMG = 352
MODE = "p5_sam2unet_mafa_v2"
CKPT = ("/data/models/zsy/mslau-net/runs/kvasir_official880_120_v2cont_2stage_e20e20_"
        "sam2unet_hiera_l_blockmafav2_rfb64_structure_352_b12_seed1234_gpu3_nw8_20261104_1209/"
        "stage2/checkpoints/best_iou_0.838153_epoch_12_0.418480.pth")

def build():
    m = MSLAU_net(img_size=IMG, mla_channels=64, in_chans=3, num_classes=1,
                  edge_guidance_enabled=False, fusion_mode="fixed",
                  decoder_mode=MODE, encoder_name="sam2_hiera_large")
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
    return m.cuda().eval()

model = build()
ids = [l.strip() for l in open(VAL) if l.strip()]
tf = A.Compose([A.Resize(IMG, IMG),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
dl = DataLoader(binary_class(DATA, ids, tf), batch_size=4, shuffle=False,
                num_workers=8, pin_memory=True)

FLIP_V = [[2]]; FLIP_HV = [[3], [2], [2, 3]]

def _fwd(im, flips):
    p = torch.sigmoid(model(im))
    for dims in flips:
        p = p + torch.flip(torch.sigmoid(model(torch.flip(im, dims))), dims)
    return p / (1 + len(flips))

def predict(imgs, sizes=(IMG,), flip=None):
    acc = None
    with torch.no_grad():
        for sz in sizes:
            im = imgs if sz == IMG else F.interpolate(imgs, size=(sz, sz),
                                                      mode="bilinear", align_corners=False)
            flips = [] if flip is None else (FLIP_V if flip == "v" else FLIP_HV)
            p = _fwd(im, flips)
            acc = p if acc is None else acc + p
    return acc / len(sizes)

THR = [0.45, 0.50, 0.55, 0.60]
variants = [("base", (IMG,), None), ("vflip", (IMG,), "v"), ("hv", (IMG,), "hv"),
            ("ms", (320, 352, 384), None), ("ms+v", (320, 352, 384), "v"),
            ("ms+hv", (320, 352, 384), "hv")]
res = {n: {t: [] for t in THR} for n, _, _ in variants}

for imgs, masks, _ in dl:
    imgs = imgs.cuda(); masks = masks.cuda().float()
    if masks.dim() == 3: masks = masks.unsqueeze(1)
    for name, sizes, flip in variants:
        pr = predict(imgs, sizes=sizes, flip=flip)
        for t in THR:
            pred = (pr > t).float()
            inter = (pred * masks).sum((1, 2, 3))
            union = ((pred + masks) > 0).float().sum((1, 2, 3))
            res[name][t] += ((inter + 1e-6) / (union + 1e-6)).tolist()

print("%-8s" % "variant", "".join("%9s" % ("thr%.2f" % t) for t in THR))
for name, _, _ in variants:
    print("%-8s" % name, "".join("%9.4f" % float(np.mean(res[name][t])) for t in THR))
```

运行：
```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  /home/zsy/miniconda3/envs/mslau-net/bin/python tta_eval.py
```

---

## 9. NAF（本分支附带，V2-cont 不需要）

本分支同时包含 NAF 研究代码，便于后续继续。它们与 V2-cont **相互独立**：
- `networks/naf_select_reg.py`：`PyramidNAF2D` 等；
- `networks/naf_discrete.py`：`CascadeDiscreteNAF2D`；
- `kvasir_train.py` 支持 `--naf_reg_weight`、`--naf_epoch_offset`；
- 对应 decoder modes：`p5_sam2unet_mafa_v2_naf`、`p5_sam2unet_mafa_v2_nafcascade`。
- **复现 V2-cont 时不要加 NAF 参数**，使用 `p5_sam2unet_mafa_v2` 即可。

---

## 10. 已知问题 / 注意事项

1. **旧 0.8824 不可比**：那是旧随机划分（旧 train 含 102 张官方 val 图）。官方干净口径是 **0.8382（base）/ 0.8415（hv TTA @0.5）**。
2. **阈值/模型选择偏差**：checkpoint 按同一 val 选最佳 epoch；单 seed。严格对外建议补 3 seeds 或固定最后 epoch。
3. 官方划分没有独立 test；`test.txt` 是 `val.txt` 的镜像。
4. 工作区有 3 个垃圾文件（见第 2 节），**不要 `git add .`**；也不要提交 `AGENTS.md` / `SERVER_USAGE.md` / `environment.lock.yml`（按约定排除）。
5. 权重/日志在 `/data/models`，不在仓库；克隆后无法直接拿到 checkpoint。
6. `kvasir_test.py` 必须传与训练一致的 `--decoder_mode` / `--encoder_name` / `--img_size`。

---

## 11. 快速检查清单（接手时）

- [ ] 能 `git log -1` 看到 `4f0697e`，分支为 `kvasir-p5-mafa-v2-cont`。
- [ ] `git grep "HieraBlockAdapterMAFAv2"` 能命中（说明 V2-cont 代码在）。
- [ ] 能找到 `configs/splits/kvasir_official_880_120/{train,val}.txt`（880/120）。
- [ ] 能用 7.2 的命令复现 Stage 2，best val IoU ≈ 0.838。
- [ ] 能用第 8.3 的脚本跑出 base 0.8381 / hv 0.8415（@0.5）。

---

## 12. 后续可做

- 3 seeds（官方划分）报 mean±std；
- 固定最后 epoch（不按 val 选模型）再报一次；
- 把 TTA（hv @0.5）固化进 `kvasir_test.py` 或独立脚本；
- 若继续做 NAF，请另开分支（避免与 V2-cont 快照混淆）。
