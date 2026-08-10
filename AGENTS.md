# MSLAU-Net 长期项目规则与实验记录

> 本文件是 Codex 的长期项目记忆。每次代码修改或实验完成后都必须更新。
> 本文件位于 Git 仓库根目录，是 Mac、`.120`、`.194` 三端同步时的唯一权威版本。
> `handover.md` 仅用于一次性交接本轮对话，不替代本文件。
> 最近更新：2026-08-10。

## 1. 长期目标

- 在 Kvasir-SEG、CVC-ClinicDB 上提高 MSLAU-Net 的分割 mIoU。
- 所有改动必须通过受控实验验证，不能只凭训练 Loss 或单次峰值判断有效。
- 明确区分 P0、P2、LFF、Dropout、数据划分和训练策略带来的影响。
- 保证代码、数据划分、随机种子、checkpoint 和日志可以复现。

## 2. 每次工作的强制规则

1. 操作前读取本文件；首次接手本轮项目时再完整读取 `handover.md`。
2. SSH 连接除非用户明确要求，否则保持开启。
3. 启动训练前检查 GPU 和 zsy 进程，禁止抢占他人 GPU。
4. 服务器工作树可能有用户修改；先执行 `git status --short`，不得 reset、checkout 或批量删除。
5. 用户说“只讨论”时，不修改代码、不启动训练。
6. 每个实验记录：假设、具体改动、控制变量、结果、结论、run/checkpoint 路径。
7. 失败实验同样记录，不能只保留正向结果。
8. 不同数据划分协议的结果不能直接比较或混排。
9. 只用验证集调参；方案锁定后才测试。已查看过的 test 结果不得反向用于调参。
10. 启动训练后至少确认样本数、网络开关、预训练加载、Epoch 0、GPU 利用率均正常。
11. 修改完成后运行 `git diff --check` 和相关 `py_compile`/冒烟测试。
12. 每次已验证的改动或训练结束后，按第 9 节格式追加记录并更新“当前最佳结果”。

## 3. 服务器与路径

### 服务器

```text
主要训练服务器：10.243.96.120
另一台服务器：  10.243.96.194
SSH 用户：       zsy
```

不要把 SSH 密码、GitHub 私钥写入代码或本文。

### `.120` 关键路径

```text
项目：           /home/zsy/projects/mslau-net
Python：         /home/zsy/miniconda3/envs/mslau-net/bin/python
数据根目录：     /data/dataset/zsy/mslau-net
模型根目录：     /data/models/zsy/mslau-net
训练输出：       /data/models/zsy/mslau-net/runs
通用预训练权重： /data/models/zsy/mslau-net/pretrained/best.pth
Kvasir：         /data/dataset/zsy/mslau-net/Kvasir-SEG
CVC：            /data/dataset/zsy/mslau-net/CVC-ClinicDB
```

GPU 检查：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -u zsy -o pid,etime,cmd | grep -E 'python.*(train|kvasir)' | grep -v grep
```

历史上 GPU 1、2 常被他人占用，GPU 0、3 较常空闲，但必须以现场检查为准。曾发生并行训练导致显存占满、GPU 利用率为 0；恢复单任务、batch32、`num_workers=8` 后正常。

## 4. 当前网络与训练语义

### 干净 P0

当前最强基线。前向过程：Encoder 四路特征 → `Conv_MLA` 自顶向下融合 → `MLAHead` → 可选 Dropout2d → `seg` → 上采样输出。

运行开关：

```text
--disable_edge_guidance
--fusion_mode fixed
--decoder_dropout 0.0
```

### P2 边缘引导

当前代码在 decoder 输出处生成 `coarse_logits`，将 `coarse_logits.detach()` 和输入图像传入边缘引导模块，再用同一个 `seg` 得到 refined logits。

当前问题：

- coarse logits 没有独立 auxiliary loss；
- coarse 与 refined 共用 `self.seg`；
- 现有 P2 实验未超过 P0。

如重启 P2，优先研究“独立 coarse head + coarse 辅助监督 + refined head”，不要先继续调学习率。

### LFF 当前语义

当前服务器代码中的 `--fusion_mode lff_scale` 已经是竞争式 softmax4，不再是最初的 `1 + delta`：

```python
fusion_logits = nn.Parameter(torch.zeros(4, mla_channels))
gamma = 4.0 * torch.softmax(fusion_logits, dim=0)
```

每通道四路权重和为 4，初始均为 1。复现早期加法式 LFF 时必须恢复对应旧代码，不能只复用当前参数名。

### 当前统一训练标准

```text
Loss：0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
Optimizer：AdamW
Base LR：1e-4
Scheduler：CosineAnnealingLR
Kvasir batch：32
num_workers：8
epoch：200
checkpoint：同时保存最低 val Loss 与最高 val IoU，以最高 IoU 为主结果
```

`kvasir_test.py` 测试时必须显式传入与训练一致的：

```text
--disable_edge_guidance
--fusion_mode
--decoder_dropout
```

否则可能用错误的 P0/P2/LFF 前向评估 checkpoint。

## 5. 数据划分协议

### Kvasir-SEG

```text
train：880
val：  120
split：/home/zsy/projects/mslau-net/configs/splits/kvasir
```

接手时重新检查 train/val/test 的交集，不要只凭文件名判断独立性。

### CVC 严格序列划分

按 `metadata.csv` 的 `sequence_id` 划分，同一视频不跨集合：

```text
train：428 张，20 序列
val：   92 张， 5 序列
test：  92 张， 4 序列
split：/home/zsy/projects/mslau-net/configs/splits/cvc_clinicdb
```

用于衡量跨视频泛化。单个 holdout 方差大，最终应采用 sequence-level 五折并报告均值±标准差。

### CVC 原仓库 CSV + fold 0

CSV：

```text
/home/zsy/projects/mslau-net/src/CVC_ClinicDB/test_train_data.csv
train category：551
test category： 61
```

原 `cvc_train.py` 在 551 张 train 中用 `GroupKFold(n_splits=5)`，固定 fold 0：

```text
实际 train：440
val：       111
test：       61
split：/home/zsy/projects/mslau-net/configs/splits/cvc_official_csv_fold0
```

这里的 group 是唯一 `image_id`，不是 `sequence_id`：train/val 的 29 个视频序列全部重叠，test 的 27 个序列也在 train 中出现。该协议用于复现原仓库和用户历史约 0.90 的结果，不能称为跨视频泛化。

## 6. 当前最佳结果

### Kvasir

| 模型 | 最佳 val mIoU | Epoch | 结论 |
|---|---:|---:|---|
| 干净 P0 | **0.860583** | 84 | 当前最佳 |
| P0 + Dropout2d(0.1) | 0.855916 | 107 | 下降 |
| 原加法式 LFF，LR 2e-3，无 Dropout | 0.852859 | 86 | 下降 |
| 原加法式 LFF，LR 2e-3，Dropout 0.1 | 0.853812 | 95 | 下降 |
| 原加法式 LFF，LR 5e-3，无 Dropout | 0.852145 | - | 下降 |
| softmax4 LFF，LR 2e-3，Dropout 0.1 | 0.849493 | 86 | 下降最多 |

最佳 Kvasir P0 checkpoint：

```text
/data/models/zsy/mslau-net/runs/kvasir_p0_only_bcedice_b32_e200_split880_120_gpu0_20260723_0947/checkpoints/best_iou_0.860583_epoch_84_0.085174.pth
```

### CVC

| 协议/模型 | val mIoU | test mIoU | 结论 |
|---|---:|---:|---|
| 严格序列划分，从通用 encoder 训练 P0 | 0.741435 | 0.7545 | 严重跨视频过拟合 |
| Kvasir 0.8605 P0 零样本评估严格 CVC val | 0.7849 | - | 比 CVC 重训高 0.0435 |
| 原仓库 CSV + fold0，干净 P0 | **0.891611** | **0.8746** | 接近用户历史 0.90 |

最佳 CVC 官方协议 checkpoint：

```text
/data/models/zsy/mslau-net/runs/cvc_clean_p0_bcedice_b32_e200_officialcsv_fold0_440_111_test61_seed1234_gpu0_nw8_20260731_084127/checkpoints/best_iou_0.891611_epoch_87_0.050696.pth
```

## 7. 代码与工具状态

已知长期改动涉及：

```text
kvasir_train.py          统一 BCE/Dice、最高 IoU 保存、seed、P0/P2、LFF、Dropout 参数
kvasir_test.py           增加 P0/P2、fusion、Dropout 配置参数
networks/mslau_net.py    P2 边缘引导、LFF、Dropout2d、softmax4 竞争权重
make_cvc_splits.py       按 sequence_id 生成严格 CVC 划分
loader.py                读取 images/masks 并二值化掩码
```

CVC 数据为复用 loader 已建立：

```text
CVC-ClinicDB/images -> PNG/Original
CVC-ClinicDB/masks  -> PNG/Ground Truth
```

旧 `cvc_train.py` 不应直接作为当前统一实验入口：默认纯 Dice、batch8、按最低 Loss 保存、默认 P2、不支持 LFF/Dropout CLI、只跑 fold0，且不能直接加载完整 0.8605 state_dict。可以借用其历史超参数做对照。

服务器工作树有未提交修改和历史杂项文件。不要清理；现场执行：

```bash
cd /home/zsy/projects/mslau-net
GIT_PAGER=cat git status --short
GIT_PAGER=cat git diff --check
```

GitHub 远程仓库：`chuifengbushuang/MSLAU_OPT`。zsy 已配置 GitHub SSH 密钥；不得读取或输出私钥。分支名需现场用 `git branch --show-current` 确认。

## 8. 改动与效果记录

### 8.1 服务器迁移与环境

- 操作：将 `.194` 的项目复制到 `.120`，代码放 `/home/zsy/projects/mslau-net`，数据和模型分别放 `/data/dataset/zsy/mslau-net`、`/data/models/zsy/mslau-net`；创建独立 Conda 环境。
- 效果：`.120` 可在空闲 GPU 上独立训练；Python 路径为 `/home/zsy/miniconda3/envs/mslau-net/bin/python`。
- 结论：环境、数据、代码、权重分离合理；后续不要把环境或大权重放入 Git 仓库。

### 8.2 P0 基线稳定化

- 操作：增加 `--disable_edge_guidance`，使 P0 明确绕过边缘引导；Loss 统一为 0.5 BCE + 0.5 Dice；同时保存最低 Loss 和最高 IoU；Kvasir 使用 880/120。
- 效果：seed1234 的干净 P0 达到 **0.860583**，Epoch84。
- 结论：这是后续模块实验唯一可信主基线。

### 8.3 随机种子实验

- 操作：P0/P2 先后补跑 seed42、1234、2026，用于观察初始化、shuffle、增强等随机因素。
- 效果：不同 seed 有波动，但本轮交接未保留所有精确数字。
- 结论：下一位 agent 必须从 `/data/models/zsy/mslau-net/runs` 日志重新汇总，不得凭记忆补数；重要结论至少基于 3 seeds。

### 8.4 P2 decoder 边缘引导

- 操作：在 `MLAHead` 后生成 coarse logits，将其 `detach()` 后与图像送入边缘引导模块，再输出 refined logits；位置与 0.8605 P0 的 decoder 输出位置对齐。
- 效果：P2 多次训练未超过干净 P0，并出现 coarse head 监督不足的疑问。
- 结论：当前 P2 路线无正收益；再次实验前先修辅助监督设计。

### 8.5 原加法式 LFF

- 操作：四路特征各增加 `(4,64)` 可学习通道权重，`gamma=1+delta`，零初始化保持 P0 起点；LFF 参数单独 optimizer group。
- 效果：LR2e-3 无 Dropout 为 0.852859；LR2e-3 + Dropout0.1 为 0.853812；LR5e-3 无 Dropout 为 0.852145。
- 结论：参数能学习，但没有超过 P0；仅调 LFF LR 价值有限。

### 8.6 LFF 独立学习率与余弦退火

- 操作：LFF 初始 LR 尝试 5e-3、2e-3，并随训练余弦退火；base LR 仍为 1e-4。
- 效果：2e-3 略优于 5e-3，但仍比 P0 低约 0.0077。
- 结论：失败原因不是简单的 LFF 学习率太小。

### 8.7 Dropout2d

- 操作：在 `MLAHead` 输出与 `seg` 之间加入 `Dropout2d(p=0.1)`；分别测试 P0 和 LFF。
- 效果：P0 + Dropout 为 0.855916，低于无 Dropout P0；LFF + Dropout 也未改善到基线以上。
- 结论：该位置实现简洁，但当前 p=0.1 对 Kvasir 无正收益。

### 8.8 softmax4 竞争式 LFF

- 操作：将 `1+delta` 改成 `gamma=4*softmax(fusion_logits, dim=0)`，使每通道四路权重和为4；保留初始全1；记录每路权重和约束误差。
- 验证：初始 min=max=1，sum error=0；256 个参数梯度有限且非零；训练中 sum error 约 4.77e-7。
- 效果：LR2e-3 + Dropout0.1 的最佳 val mIoU 为 **0.849493**，低于原 LFF 和 P0；最后20轮约0.83681。
- 结论：竞争约束实现正确，但限制了四路共同增强，当前方案应停止继续小幅调参。

### 8.9 CVC 数据下载与适配

- 操作：用 KaggleHub 下载 `balraj98/cvcclinicdb` 到 `/data/dataset/zsy/mslau-net/CVC-ClinicDB`；建立 images/masks 软链接；新增 sequence split 工具；给 `kvasir_test.py` 增加网络配置参数。
- 效果：612 组图像/掩码全部配对并可读取，PNG/TIF 各612组。
- 结论：数据本身和 loader 路径无错误。

### 8.10 CVC 严格序列 P0

- 操作：按 sequence_id 做 428/92/92，clean P0、seed1234、batch32、BCE/Dice。
- 效果：最佳 val 0.741435（Epoch36），最终 train 0.9618、val 0.6415；test 诊断为0.7545。验证 sequence10 仅0.0316，多数帧被预测为空。
- 结论：发生严重跨视频过拟合；单个严格 holdout 方差高。

### 8.11 Kvasir 最佳 P0 零样本测试 CVC

- 操作：加载 Kvasir 0.860583 完整 P0 checkpoint，不训练，直接推理严格 CVC val。
- 效果：mIoU 0.7849、Dice 0.8486，比 CVC 从通用 encoder 重训高0.0435。
- 结论：Kvasir 的多样性带来更好泛化；CVC 小样本重训发生窄域过拟合/灾难性遗忘。后续应尝试从完整 Kvasir P0 小 LR 微调。

### 8.12 下载原仓库 CVC CSV

- 操作：只下载 `test_train_data.csv` 到 `/home/zsy/projects/mslau-net/src/CVC_ClinicDB/`；校验612条、551 train、61 test，与数据文件完全匹配。
- 效果：成功复现旧 `cvc_train.py` 预期的数据入口。
- 结论：此前严格划分与用户历史0.90不可直接比较。

### 8.13 CVC 官方 CSV + fold0 干净 P0

- 操作：按原代码将551 train经 GroupKFold 取 fold0，得到440 train/111 val，61 test保留；其余配置与 clean P0 一致。
- 效果：最佳 val **0.891611**（Epoch87），CSV test **0.8746**，Dice0.9259；最后20轮 val 约0.88724，训练稳定。
- 结论：划分协议使 val 比严格序列划分高0.1502，基本解释用户历史约0.90。下一对照应保持该 split，改为旧参数 batch8 + 纯 Dice。

## 9. 后续每次追加记录的模板

```markdown
### YYYY-MM-DD：实验/改动名称

- 假设：为什么做。
- 具体操作：改了哪些文件、模块、公式或训练参数。
- 控制变量：数据集与 split、seed、batch、Loss、LR、epoch、网络开关。
- 结果：best val mIoU、epoch、test（仅最终方案）、异常现象。
- 对比：相对哪个同协议基线提高/下降多少。
- 结论：支持或否定什么假设，是否继续。
- 证据：run 目录、checkpoint、commit/branch。
```

## 10. 下一步优先级

1. **先补齐统计基线**：从 runs 汇总 Kvasir P0 seed42/1234/2026，报告均值±标准差。
2. **复现 CVC 历史0.90**：官方 CSV/fold0、clean P0、batch8、纯 Dice、最高 IoU保存；只改变旧训练参数。
3. **严格 CVC 提升**：新增完整模型 `--init_checkpoint`，从 Kvasir 0.860583 P0 初始化；先冻结 encoder，再以 encoder 1e-5、decoder 1e-4 微调；采用 sequence-level 五折。
4. **P2**：独立 coarse head + auxiliary loss 后再实验。
5. **LFF**：在未完成特征幅值统计和多 seed 前，不继续只调 LR/Dropout。
6. 每次实验完成后更新本文件，并将重要代码状态提交到清晰命名的 Git 分支；不要让 checkpoint 与代码语义错配。
