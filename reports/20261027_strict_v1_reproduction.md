# 在自己的P5.1分支加入严格v1改动的实施说明

本说明供原实验者在自己的分支修改原P5.1基线，试验产生0.882054那版训练协议。不需要切换到研究分支，也不需要下载源码压缩包或采用我们的实验启动器。

先更正“只改配置”的说法：网络架构确实未改，但实际增加了输入随机性控制代码，改变了样本排列和增强序列；其中v1还存在增强开关联动。下面给出全部核心逻辑和接入位置，保留这一历史行为，便于对照。

## 目标结果与适用范围

原代码依据是`kvasir-p5-sam2unet-official-adapter`分支提交`81b19e2c801efcf74826b8127f5ea712b0247ec5`。
目标运行使用seed2026，在RTX3090单卡上完成20轮，最佳验证IoU为**0.8820540308952332**，出现在epoch18，即第19轮；Dice=0.9333、Precision=0.9321、Recall=0.9427。

这次0.882054目前只有一次完整运行。原作者0.878616来自seed1234，不能把两者差值直接归因于严格模式。严格v1三个seed的IoU为：42→0.874577、1234→0.853762、2026→0.882054，均值0.870131；高单次峰值不等于平均性能已提升。

## 改动清单

| 部分 | 原始代码 | 本次需要实现的变化 |
|---|---|---|
|网络、RFB、Decoder、损失|原P5.1|保持原样，不加NAF或其他模块|
|图像增强工厂|Resize、翻转、ShiftScaleRotate、CoarseDropout、Normalize、ToTensorV2|算子顺序及参数保持原样；读取样本前固定Compose内部seed|
|样本顺序|DataLoader shuffle=True|按seed和epoch，使用独立Generator产生randperm|
|worker随机性|原DataLoader初始化|独立worker Generator及worker_init_fn，每个阶段更新epoch|
|CuDNN|原本已deterministic=True、benchmark=False|保留，不算新增性能改进|
|严格算法|未调用全局严格算法开关|新增torch.use_deterministic_algorithms(True, warn_only=False)|
|进程环境|原脚本在进程内设置部分变量|Python启动前设置CUBLAS_WORKSPACE_CONFIG和PYTHONHASHSEED|
|审计日志|原日志|可选记录输入和状态哈希；不是训练算法或新增损失|

历史实现修改`kvasir_train.py`和`kvasir_test.py`，增加`paired_inputs.py`和`strict_runtime.py`。
为方便在自己的分支手动实现，下文把其中影响输入及严格计算的核心逻辑合并到一个`reproducibility_v1.py`。
不包含原实现的只读哈希审计和输出包装；不需要移植run.py、config.json或我们的目录整理脚本。

## 第一步 新增核心辅助文件

在自己分支的`kvasir_train.py`同级新建`reproducibility_v1.py`，放入下列完整代码。
函数、类名可以按项目风格调整，但若想保持相同输入序列，必须保留`PROTOCOL`字符串、SHA256派生算法、phase名称、样本ID、epoch编号及各随机流的键。
不能用`seed + epoch + index`或Python内置hash替代。

```python
"""Sample/epoch-keyed input randomness; independent of model and worker RNGs."""
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, SequentialSampler

from loader import binary_class

PROTOCOL = 'paired_aug_v1'


def stable_seed(seed, stream, *parts):
    payload = json.dumps([PROTOCOL, int(seed), stream, *parts],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


class PairedDataset(binary_class):
    """Keep the original reader/mask conversion; only seed its Compose call."""
    def __init__(self, path, data, transform, seed, phase):
        super().__init__(path, data, transform)
        self.seed, self.phase, self.epoch = int(seed), phase, 0
        if not hasattr(transform, 'set_random_seed'):
            raise TypeError('This protocol requires Compose.set_random_seed (Albumentations 1.4.24)')

    def __getitem__(self, index):
        sample_id = self.folders[index]
        self.transforms.set_random_seed(stable_seed(
            self.seed, 'augmentation', self.phase, self.epoch, sample_id))
        return super().__getitem__(index)


class EpochRandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        generator = torch.Generator().manual_seed(stable_seed(
            self.dataset.seed, 'sample_order', self.dataset.epoch))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    # These seeds cover other code using global RNGs. Compose is seeded per
    # sample above, NOT from worker_id or a worker-local augmentation stream.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, num_workers, pin_memory=False):
    train = dataset.phase == 'train'
    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=EpochRandomSampler(dataset) if train else SequentialSampler(dataset),
                        drop_last=train, num_workers=num_workers, pin_memory=pin_memory,
                        worker_init_fn=seed_worker, generator=torch.Generator(),
                        persistent_workers=False)
    set_loader_epoch(loader, 0)
    return loader


def set_loader_epoch(loader, epoch):
    if loader.persistent_workers:
        raise ValueError('Set epoch before iteration; paired_aug_v1 requires persistent_workers=False')
    dataset = loader.dataset
    dataset.epoch = int(epoch)
    loader.generator.manual_seed(stable_seed(dataset.seed, 'worker_base', dataset.phase, int(epoch)))


def configure_runtime():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

```

此文件继承原`loader.binary_class`，图像读取、mask>0转换和图像/mask共同增强保持原样。
这些代码管理输入与计算随机性，不新增特征处理、可训练参数或损失项；v1对增强联合分布的影响见后文。
样本ID沿用原`load_ids`返回的文件名，包括扩展名，不替换为样本索引。

## 第二步 在训练脚本启用严格计算

在`kvasir_train.py`导入：

```python
from reproducibility_v1 import (
    PairedDataset, make_loader, set_loader_epoch, configure_runtime,
)
```

在原脚本的seed设置之后、创建数据集及模型之前调用一次：

```python
configure_runtime()
```

原有Python、NumPy、torch及CUDA的seed设置保持原顺序，不新增消耗随机数的模型或初始化操作。
`configure_runtime()`检查CuBLAS环境变量并启用严格算法，同时保留原CuDNN设置。
不要将`warn_only`改为True；若有算子报错，应保留报错信息。
不修改F.interpolate的mode或align_corners。torch2.1.2下，严格算法会为相关CUDA双线性上采样反向选择确定性替代路径，因此计算轨迹可能不同。

目标运行的matmul TF32=False、cuDNN TF32=True，flash/memory-efficient/math SDPA后端均允许。
这些设置没有因本次改动被强制关闭，不要额外增加关闭全部TF32、改注意力实现等变化。

## 第三步 替换训练和验证的DataLoader构造

在`kvasir_train.py`原来创建`train_dataset`、`val_dataset`、`train_loader`和`val_loader`的地方，用以下代码替换。
这里的dataset_path、train_files、val_files、seed、args和两个transform工厂均沿用原脚本变量。

```python
train_dataset = PairedDataset(
    dataset_path, train_files, get_train_transform(args.img_size),
    seed, "train",
)
val_dataset = PairedDataset(
    dataset_path, val_files, get_valid_transform(args.img_size),
    seed, "valid",
)
train_loader = make_loader(
    train_dataset, args.batch, args.num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = make_loader(
    val_dataset, max(1, args.batch // 2), args.num_workers,
    pin_memory=torch.cuda.is_available(),
)
dataloaders = {"train": train_loader, "valid": val_loader}
```

不要再额外传shuffle=True；训练排列已由sampler负责。
训练仍为batch12、workers8、drop_last=True，每轮880张中取876张、73批。
训练过程中的验证仍为batch6、workers8、120张、20批。
保持`persistent_workers=False`，因为每轮要将更新后的dataset.epoch传给新建worker。

## 第四步 在每个epoch的每个阶段更新输入状态

在`train_model`中，找到原来的：

```text
for inputs, labels, _ in dataloaders[phase]:
```

只在它前面增加一行，缩进与这个for相同：

```python
set_loader_epoch(dataloaders[phase], epoch)
for inputs, labels, _ in dataloaders[phase]:
    # 后续保留原训练或验证计算
    ...
```

此调用位于外层epoch循环和train/valid阶段循环内，每个阶段开始遍历前执行。
epoch仍从0开始。不要只在训练开始时调用一次，也不要每个batch重置sampler。
原来的model.train/eval、loss.backward、optimizer.step、scheduler.step、指标计算和checkpoint选择顺序保持原样。

## 第五步 在独立评估脚本启用同一严格计算

在`kvasir_test.py`导入：

```python
from reproducibility_v1 import configure_runtime
```

在解析参数后、模型计算前增加：

```python
configure_runtime()
```

独立评估仍采用原loader与验证预处理，不使用训练增强。保持352输入、batch8、workers0、阈值0.5。
评估进程启动前也必须传入CuBLAS环境变量；不能只在训练脚本中设置。

## v1增强联动必须说明

在Albumentations 1.4.24下，每次`Compose.set_random_seed(seed)`会把同一种子传给子变换。
v1逐样本重设后，三个p=0.25的几何增强开关联动，并与默认p=0.5的CoarseDropout相关。
例如4096次既有合成检查中，水平与垂直翻转同时触发约24.805%，原机制约6.714%；四个增强开关只出现3种组合，原机制出现16种。
这是不同增强算子之间的联动，图像与mask仍共同变换、保持几何对齐。

所以这次修改既固定了输入，也改变了增强联合分布。它可能影响结果，但尚未证明它造成了0.882054。
本说明保留v1做历史对照；后续v2使用共享持续推进的随机流修复了联动，属于另一套输入协议。
若自行改用v2或其他增强固定方法，应另记实验，不能称为这次v1的原样复现。
只增加严格算法开关、而不加入上面的输入控制，也属于另一组实验。

## 保持原训练设置

- 网络：原SAM2 Hiera-L＋block Adapter32＋四层RFB64＋原U形Decoder，无NAF。
- 初始化：同一官方Hiera权重；Adapter和Decoder按seed初始化，不从已训练的P5.1 checkpoint继续。
- 数据：原Kvasir-SEG、原880/120划分、原mask>0口径，不改图像与mask文件。
- 输入352、batch12、workers8、epoch20；目标seed为2026。
- Structure主损失＋D2＋D3，辅助权重均1.0，boundary_weight=0。
- AdamW：encoder_lr=decoder_lr=1e-3、weight_decay=5e-4；CosineAnnealingLR，T_max=20、eta_min=0。
- 不新增AMP、EMA、warmup、梯度裁剪或模型初始化改动。
- checkpoint规则和三份权重命名保留原实现，以最佳验证IoU为主要结果。

关键环境是Python3.10、torch2.1.2、torch CUDA build11.8、cuDNN8700、timm1.0.15、Albumentations1.4.24、albucore0.0.23、NumPy1.26.4。
目标运行的torchvision运行时为0.16.2+cu121，这一实测组合没有为本次实验重装。
保留原环境，先比对版本，不需要照我们的requirements重新安装。

官方Hiera权重SHA256：

```text
7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c
```

## 在自己的分支运行

完成上述改动后，在自己的项目目录运行原来的`kvasir_train.py`，增加进程环境变量并将seed设为2026即可。
下面示例使用原作者Python和原数据路径，输出采用原作者自己的runs目录；不依赖研究分支任何启动脚本。
GPU3沿用目标运行，可按可用情况调整并记录。

```bash
(
    set -e
    export CUDA_VISIBLE_DEVICES=3
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONHASHSEED=2026
    export NO_ALBUMENTATIONS_UPDATE=1

    P51_RUN_DIR="/data/models/zsy/mslau-net/runs/p51_strict_v1_seed2026_$(date +%Y%m%d_%H%M%S)_$$"

    /home/zsy/miniconda3/envs/mslau-net/bin/python -u kvasir_train.py \
      --dataset /data/dataset/zsy/mslau-net/Kvasir-SEG \
      --train_txt configs/splits/kvasir/train.txt \
      --val_txt configs/splits/kvasir/val.txt \
      --output_dir "$P51_RUN_DIR/checkpoints" \
      --log_dir "$P51_RUN_DIR/logs" \
      --decoder_mode p5_sam2unet \
      --encoder_name sam2_hiera_large \
      --hiera_checkpoint /data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors \
      --loss structure \
      --img_size 352 --batch 12 --num_workers 8 --epoch 20 --seed 2026 \
      --encoder_lr 1e-3 --decoder_lr 1e-3 \
      --aux_d2_weight 1.0 --aux_d3_weight 1.0 --boundary_weight 0.0

    /home/zsy/miniconda3/envs/mslau-net/bin/python -u kvasir_test.py \
      --dataset /data/dataset/zsy/mslau-net/Kvasir-SEG \
      --split_txt configs/splits/kvasir/val.txt \
      --model "$P51_RUN_DIR/checkpoints/kvasir_best_model.pth" \
      --decoder_mode p5_sam2unet --encoder_name sam2_hiera_large \
      --img_size 352 --batch 8 --num_workers 0 --threshold 0.5 --debug false

    echo "本次输出：$P51_RUN_DIR"
)
```

`PYTHONHASHSEED`需要在Python启动前传入；在已启动的Python进程里修改os.environ不能重置其hash种子。
`CUBLAS_WORKSPACE_CONFIG`也应在进程启动前设置。原脚本已有全局seed设置，保留即可。

## 记录与结果解释

本次核心改动不要求采用我们的哈希日志、manifest、源码展开或目录整理脚本，原日志和权重输出可以保留。
建议至少记录：Git提交、环境版本、完整命令、seed、最佳epoch/IoU、Dice/Precision/Recall，以及本次输入协议确为v1。
若后续需要精确核对，再增加只读的初始state_dict、每阶段输入及权重内容哈希；这些审计不构成额外训练算法。

本文最小实现的核心输入函数与严格开关对应历史v1，但没有加入全部审计代码，完整20轮的数值等同性仍需实际运行验证。
不能保证对方环境运行一次就得到0.882054，也不能用单个seed确认稳定提升。
先试验目标seed2026并保留结果；若要评价这套训练协议的普遍效果，再与自己的原基线按相同种子进行比较。
