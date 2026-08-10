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

### P3 渐进式小波边界 Decoder

P3 位于分支 `kvasir-p3-progressive-wavelet`，保留 legacy P0/P2/LFF，通过以下参数启用：

```text
--decoder_mode progressive_wavelet
--progressive_channels 96
--disable_edge_guidance
--fusion_mode fixed
```

结构：Encoder 的 `64x64、32x32、16x16、8x8` 四层特征 → 多尺度 context →
`8→16→32→64→128` 渐进式选择性融合 → final head → 上采样到256。
两级固定 Haar 高频从反归一化 RGB 提取，并注入三个融合阶段；D3 coarse head 生成
detach 的 reverse attention；D2、D3 和边界 head 接受辅助监督，final head 独立输出。

当前 P3 损失：

```text
L = L_final + 0.2*L_D2 + 0.1*L_D3 + 0.1*L_boundary
每个 L 使用 0.5*BCE + 0.5*Dice
```

P3 optimizer：encoder LR `5e-5`，decoder LR `2e-4`，5 epoch linear warmup 后 cosine；
batch16、num_workers8、200 epoch、seed1234。legacy 模式默认训练语义保持不变。

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
| P3 reverse-only（无wavelet edge） | **0.859864±0.010269**（best 0.871511） | 39/82/57 | 当前三seed最高均值；2/3 seed提升 |
| 干净 P0 | 0.853004±0.010375（best 0.860583） | 100/84/51 | 历史可信主基线 |
| P3 渐进式小波 decoder | 0.859276 | 126 | 低于同 seed P0 0.001307；后期更稳定但小目标退化 |
| P3 去 wavelet edge/reverse | 0.856579 | 137 | 小目标恢复、后期最稳定，但最大目标明显退化 |
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
kvasir_train.py          统一 Loss/保存/seed；P3 辅助损失、分组 LR、warmup+cosine
kvasir_test.py           P0/P2/LFF/Dropout/P3 decoder 配置与 checkpoint 测试参数
networks/mslau_net.py    P2、LFF、Dropout2d、softmax4、P3 渐进式小波 decoder
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
- 效果：P0 seed42/1234/2026 分别为0.841179、0.860583、0.857249；均值0.853004，样本标准差0.010375，极差0.019404。
- 结论：单 seed 提高0.003～0.008不能直接判定有效；重要结论至少基于3 seeds。

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

### 2026-08-10：Kvasir P3 渐进式小波边界 Decoder

- 假设：P0 的四路特征过早对齐到64x64后累加，decoder 缺少逐级空间重建；LFF 只能缩放特征，不能修复该结构瓶颈。
- 具体操作：新增可选 P3 decoder；逐级融合四层 encoder 特征；加入两级 Haar 高频、多阶段选择性门控、D3 reverse attention、D2/D3 辅助 head、独立边界 head 和 final head。
- 控制变量：Kvasir 880/120、seed1234、0.5 BCE+0.5 Dice、无 Dropout、原 encoder 预训练、200 epoch；P3 batch16、encoder LR5e-5、decoder LR2e-4、warmup5。
- 验证：`git diff --check`、`py_compile`、随机 batch2 前向/反向、旧0.860583 P0 checkpoint严格加载、真实880/120一轮训练及P3 checkpoint推理均通过。
- 冒烟结果：真实数据1 epoch train IoU 0.5600、val IoU 0.6669；用于验证流程，不作为正式结果。
- 正式结果：35m12s完成。Best val IoU **0.859276@126**，对应 MainLoss0.092744；best MainLoss0.089036@40，对应IoU0.854277。最终 train/val IoU为0.9457/0.8489，最终泛化差距0.0968。
- 稳定性：P3最后20轮val IoU为0.848130±0.000898；P0同seed最后20轮为0.839600±0.001275。P3后期均值高0.008530且波动更小，但峰值仍比P0 0.860583低0.001307。
- 验证指标：mIoU0.8593、Dice/F1 0.9129、Precision0.9270、Recall0.9175、Accuracy0.9800、FPS141.29。P0对应Precision0.9197、Recall0.9245：P3更保守，假阳性更少但漏检更多。
- 逐图像分析：120张中P3胜65、负54、平1，中位差+0.00217，但10张下降超过0.05并拉低均值。按GT面积四分位，最小组下降0.03237；中间两组分别提升0.01229和0.01514；最大组基本持平。最大退步样本面积仅0.73%，IoU从0.8324降到0.1721。
- 门控参数：best checkpoint的edge scale在fuse3/2/1为0.3313/0.3087/0.1206，reverse scale为0.1000/0.1679/0.0747，说明模型更依赖较粗尺度的小波边缘。
- 诊断性推理：不重训、仅在内存关闭reverse得到0.86054；关闭edge得到0.86189；关闭edge+reverse得到0.86224，最小面积组从0.7613升至0.7910。该结果是在val上做的post-hoc诊断，不能当正式消融或新主结果。
- 证据：`/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_wavelet_c96_aux020_010_boundary010_b16_e200_seed1234_gpu0_nw8_20260810_0321`；best checkpoint为`checkpoints/best_iou_0.859276_epoch_126_0.092744.pth`；代码提交 `3869a0a`。
- 结论：完整P3未刷新P0峰值，不补跑其他seed。渐进式decoder/选择性通道融合具有正信号，但当前乘性wavelet edge与reverse attention伤害小目标；下一次应做正式的“保留渐进式decoder与channel gate、去掉edge/reverse”训练，而不是继续叠加模块。

### 2026-08-10：P3 去小波边缘门与反向注意力正式消融

- 假设：完整P3对小目标的退化主要来自wavelet edge与reverse attention乘性门控，而非渐进式decoder本身。
- 具体操作：为P3增加两个显式且默认关闭兼容性不变的CLI消融开关；本次关闭wavelet edge gate与reverse attention，保留progressive decoder、multi-scale context、selective/channel gate、D2/D3辅助监督及boundary head。
- 控制变量：Kvasir 880/120、seed1234、0.5 BCE+0.5 Dice、batch16、无Dropout、encoder LR5e-5、decoder LR2e-4、warmup5、200 epoch；与完整P3一致。
- 验证：语法与CLI检查通过；随机batch2前向/反向确认edge/reverse参数无梯度而channel gate有梯度；旧完整P3 checkpoint 874项strict加载通过；真实数据1 epoch train/val完整通过，val IoU 0.702868。
- 正式结果：34m49s完成。Best val IoU **0.856579@137**，对应MainLoss0.109669；best MainLoss0.100314@130，对应IoU0.854861。最终train/val IoU为0.9457/0.8552，泛化差距0.0905。
- 对比：比同seed P0低0.004004，比完整P3低0.002697；不满足0.865的多seed门槛，不补seed42/2026。
- 稳定性：最后20轮val IoU为0.855090±0.000793，共65轮达到0.85；完整P3分别为0.848130±0.000922和47轮，P0为0.839600±0.001308和12轮。该消融峰值降低，但后期最稳且泛化差距最小。
- 验证指标：mIoU0.856579、Dice/F1 0.912337、Precision0.931128、Recall0.916956、Accuracy0.973530。相对P0仍偏高Precision、低Recall，表现为更保守的分割。
- 面积分组：最小到最大GT面积四分位IoU为0.796212/0.876358/0.894080/0.859666；完整P3为0.761470/0.892878/0.895429/0.887326。关闭两门控修复了小目标，但最大目标下降0.027660，成为总分下降的主因。
- 逐图对比：相对P0胜73、负47，中位差+0.002999，但14张下降超过0.05、13张提升超过0.05，少量严重退步样本拉低均值；相对完整P3胜64、负56，中位差+0.000873。
- 阈值诊断：0.30到0.70扫描的最佳阈值为0.425，mIoU仅0.856916，比默认0.5提高0.000337，说明问题不是简单的输出阈值失配。
- 训练动态：channel scale在fuse3/2/1为0.3995/0.3435/0.1463，高于完整P3的0.3755/0.3123/0.1382，说明去掉空间门控后channel gate有所补偿。
- post-hoc方向证据：在完整P3 checkpoint上仅保留edge为0.860584，仅保留reverse为0.861951，两者都关为0.862309；这些仍是同一val上的诊断，不能当正式结果。按面积分组，edge更保护最大目标，reverse在总分和小目标间更均衡。
- 证据：run目录 `/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_noedge_noreverse_c96_aux020_010_boundary010_b16_e200_seed1234_gpu0_nw8_20260810_0426`；best checkpoint为`checkpoints/best_iou_0.856579_epoch_137_0.109669.pth`；代码提交 `7f7b261`。
- 结论：正式重训否定了“直接删除两门控即可提升峰值”的假设，但支持“乘性门控伤害小目标、同时帮助大目标”的机制判断。下一步不应继续调阈值或补seed，而应正式训练只保留reverse的一路消融。

### 2026-08-10：P3 reverse-only正式消融

- 假设：reverse attention可保留完整P3对中大目标的结构约束，同时关闭wavelet edge可减少小目标退化；post-hoc方向结果为0.861951，但必须正式重训验证。
- 具体操作：启用progressive decoder，关闭wavelet edge，保留reverse attention、selective/channel gate、D2/D3辅助监督和boundary head；无需新增代码。
- 控制变量：Kvasir 880/120、seed1234、0.5 BCE+0.5 Dice、batch16、无Dropout、encoder LR5e-5、decoder LR2e-4、warmup5、200 epoch；与前两轮P3一致。
- 验证：随机batch2前向/反向确认wavelet edge=False、reverse attention=True；edge scale无梯度，reverse/channel scale梯度非零，四个输出头尺寸正确。
- 正式结果：35m15s完成。Best val IoU **0.871511@82**，对应MainLoss0.091392；best MainLoss0.088864@72，对应IoU0.868192。最终train/val IoU为0.9453/0.8592，泛化差距0.0861。
- 对比：比同seed P0高0.010928，比完整P3高0.012235，比去双门控P3高0.014932；超过0.865门槛，应补seed42/2026。
- 稳定性：最后20轮val IoU为0.858555±0.000678，共117轮达到0.85；其后期波动、泛化差距和高IoU持续时间均优于P0、完整P3及去双门控P3。
- 验证指标：mIoU0.871511、Dice/F1 0.925828、Precision0.939108、Recall0.924487、Accuracy0.977467。相对P0，Recall基本持平而Precision提高约0.0194，说明提升主要来自减少假阳性，并未以明显增加漏检为代价。
- 面积分组：最小到最大GT面积四分位IoU为0.820686/0.875412/0.914351/0.875593；相对P0分别为+0.026870/-0.005141/+0.034087/-0.012025。reverse-only显著改善最小和中大目标，并将去双门控的大目标组从0.859666恢复到0.875593，但第二、第四组仍未超过P0。
- 逐图对比：相对P0胜72、负48，中位差+0.003500，15张提升超过0.05、7张下降超过0.05；相对完整P3胜66、负54，平均提升0.012235。提升不只是单个样本，但均值提升仍大于中位数，必须通过多seed确认。
- 门控参数：best checkpoint的channel scale在fuse3/2/1为0.3726/0.3066/0.1361；reverse scale在实际使用的fuse2/1为0.1730/0.0814，与完整P3接近。关闭的edge scale保持初值0.1且无梯度。
- 证据：run目录 `/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_reverse_only_noedge_c96_aux020_010_boundary010_b16_e200_seed1234_gpu0_nw8_20260810_0514`；best checkpoint为`checkpoints/best_iou_0.871511_epoch_82_0.091392.pth`。
- 结论：正式训练支持“wavelet edge是完整P3的主要负贡献，reverse attention具有正贡献”。reverse-only成为当前单seed最佳模型，但在seed42/2026完成前不能替代P0作为稳定主结论，也不在验证集上继续调阈值。

### 2026-08-10：P3 reverse-only seed42/2026补跑与三seed结论

- 目的：验证seed1234的0.871511是否可复现，并与P0三seed均值0.853004±0.010375公平比较。
- 控制变量：除随机种子外，与reverse-only seed1234完全一致；Kvasir 880/120、batch16、0.5 BCE+0.5 Dice、encoder LR5e-5、decoder LR2e-4、warmup5、200 epoch、wavelet edge关闭、reverse attention开启。
- 执行：seed42在GPU0、seed2026在GPU1，各num_workers8并行运行；两组均为880/120样本，预训练与网络开关确认正确，训练期间两卡计算正常。
- seed42结果：35m53s完成，best IoU **0.855969@39**，对应MainLoss0.093301；best MainLoss0.090549@22，对应IoU0.843605；最终train/val为0.9455/0.8424，最后20轮0.841360±0.000865。相对P0 seed42的0.841179提高0.014790。
- seed2026结果：36m45s完成，best IoU **0.852112@57**，对应MainLoss0.106723；best MainLoss0.099889@20，对应IoU0.843325；最终train/val为0.9465/0.8408，最后20轮0.843145±0.001561。相对P0 seed2026的0.857249下降0.005137。
- 三seed统计：reverse-only seed42/1234/2026为0.855969/0.871511/0.852112，均值 **0.859864±0.010269**；P0为0.841179/0.860583/0.857249，均值0.853004±0.010375。平均提高0.006860，配对差值为+0.014790/+0.010928/-0.005137，2/3 seed提升；跨seed标准差与P0几乎相同，不能宣称方差降低。
- 指标机制：三seed reverse-only平均Precision约0.933990、Recall约0.912873；P0约0.920165/0.918941。平均收益主要来自减少假阳性，但以约0.0061 Recall为代价。seed2026的Precision0.933935、Recall0.902603，相对P0的0.918695/0.928629，说明该seed失败来自明显欠分割。
- 逐图配对：seed42相对P0胜74、负45，中位差+0.005319；seed1234胜72、负48，中位差+0.003500；seed2026虽均值下降，仍胜65、负55且中位差+0.002078，说明少量严重退步样本拉低seed2026均值。
- 三seed面积组均值：最小到最大组reverse-only为0.797977/0.884796/0.899863/0.856820，P0为0.793642/0.881120/0.885840/0.851375，对应平均提升+0.004335/+0.003676/+0.014022/+0.005445。只有第三面积组在三个seed中均提升；其余组的正收益存在seed依赖。
- 参数稳定性：reverse scale的fuse2在seed42/1234/2026为0.1731/0.1730/0.1669，fuse1为0.1054/0.0814/0.0988，没有异常发散；seed2026的Recall问题更可能来自coarse/reverse空间图分布而不是标量scale。
- 证据：seed42 run目录 `/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_reverse_only_noedge_c96_aux020_010_boundary010_b16_e200_seed42_gpu0_nw8_20260810_0643`，best checkpoint为`checkpoints/best_iou_0.855969_epoch_39_0.093301.pth`；seed2026 run目录 `/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_reverse_only_noedge_c96_aux020_010_boundary010_b16_e200_seed2026_gpu1_nw8_20260810_0643`，best checkpoint为`checkpoints/best_iou_0.852112_epoch_57_0.106723.pth`。
- 结论：reverse-only是当前三seed均值最高的实验结构，满足“均值更高且至少2个seed提升”的预设判据，可升级为当前主模型候选；但n=3、一个seed下降且均值提升大于中位数提升，证据强度仍属中等，P0继续作为历史基线。

### 2026-08-10：P3 reverse-only 版本发布

- 操作：将达到单 seed 最高 **0.871511** 的精确代码状态固定为 GitHub 分支 `kvasir-p3-reverse-only-0871`，并增加 annotated tag `kvasir-p3-reverse-only-0.871511`。
- 代码状态：commit `791560e`；P3 保持 `progressive_wavelet` decoder、关闭 wavelet edge、保留 reverse attention。
- checkpoint：`/data/models/zsy/mslau-net/runs/kvasir_p3_progressive_reverse_only_noedge_c96_aux020_010_boundary010_b16_e200_seed1234_gpu0_nw8_20260810_0514/checkpoints/best_iou_0.871511_epoch_82_0.091392.pth`。
- 结论：P3 的代码、标签和最高 checkpoint 已形成可复现闭环；P4 在新分支开发，不覆盖 P3。

### 2026-08-10：P4 cascade reverse + 352 + final Lovasz + DINOv2 蒸馏

- 假设：P3 将同一 coarse reverse map 复用于后续层，可能造成局部纠错不足和欠分割；逐级生成 D3/D2/D1 预测并只引导下一层，可以保留反向注意力的 Precision 收益，同时提高空间纠错能力。352 输入改善小息肉细节；final Lovasz直接优化IoU代理；DINOv2多层特征蒸馏增强小数据集表征。
- 具体操作：在独立分支 `kvasir-p4-cascade-ra-352-dino-lovasz` 新增 `cascade_reverse` decoder；D3→D2、D2→D1、D1→final 使用 `1-sigmoid(detached logits)` 的逐级 reverse residual correction。纠错卷积末层零初始化，保留 identity 起点；不使用 wavelet edge。
- 监督：final 使用 0.35 BCE + 0.35 Dice + 0.30 Lovasz；D1/D2/D3/boundary 仍使用 0.5 BCE + 0.5 Dice，权重分别0.1/0.2/0.1/0.1。
- 蒸馏：冻结 `vit_small_patch14_dinov2`，取 block 2/5/8/11 的384通道特征；四级学生特征用1×1 adapter投影后做平均 cosine loss，权重0.1。教师只在训练出现，学生checkpoint和推理均不包含教师或adapter。
- 输入与优化：学生输入352；教师输入364（26×26 patch grid）；encoder LR5e-5、decoder/adapter LR2e-4、warmup5后余弦退火；计划 seed1234、batch8、200 epoch、Kvasir 880/120。
- 权重：服务器无法访问 Hugging Face，已从 Meta 官方直链下载原始 DINOv2 ViT-S/14 权重到 `/data/models/zsy/mslau-net/pretrained/dinov2_vits14_pretrain.pth`；两端 SHA256 均为 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。原始 `mask_token` 仅属遮挡预训练，加载时显式移除，其他174个键严格匹配。
- 验证：`py_compile`、CLI、随机batch2的352前向/五路监督/Lovasz反向通过；reverse correction梯度非零；P3 0.871511 checkpoint仍可strict加载。DINO教师保持eval且无梯度，四个adapter和学生encoder梯度非零。
- 真实数据冒烟：8 train/4 val、batch4、1 epoch在GPU0完成，train/val IoU为0.1799/0.1802；P4 checkpoint可由测试脚本严格加载并推理。该数值仅验证完整链路，不作为实验效果。
- 实验性质：这是冲击最高mIoU的组合实验，不是单变量消融。若有效，必须后续拆分 `P4 decoder`、`352`、`Lovasz`、`DINOv2` 才能归因。
- 当前状态：正式880/120训练待启动；完成后补写best val mIoU、epoch、稳定性与run目录。

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

1. **完成P4组合实验**：先跑完 `cascade reverse + 352 + final Lovasz + DINOv2` seed1234；以0.871511单seed峰值和reverse-only三seed均值0.859864同时比较。
2. **P4有效后拆分归因**：若明显刷新峰值，固定split/seed后依次去掉DINOv2、Lovasz或退回256；组合实验本身不能说明哪个模块有效。
3. **P4失败先查优化冲突**：对比MainLoss、DistillLoss、Precision/Recall、预测面积比与reverse map；先判断是蒸馏干扰还是cascade欠分割，不直接堆新模块。
4. **锁定reverse-only主候选**：保留P3分支、tag、三个checkpoint，不恢复wavelet edge；P0继续作为历史基线。
5. **更强证据**：论文级结论建议再加2个预注册seed或建立独立test；当前P3 n=3均值提升0.006860但配对结果含1次下降，不做显著性夸大。
6. **复现 CVC 历史0.90**：官方CSV/fold0、clean P0、batch8、纯Dice、最高IoU保存；只改变旧训练参数。
7. **严格 CVC 提升**：新增完整模型`--init_checkpoint`，从Kvasir 0.860583 P0初始化；先冻结encoder，再以encoder 1e-5、decoder 1e-4微调；采用sequence-level五折。
8. **LFF**：停止只调LR/Dropout；如重启，先完成特征幅值统计。
9. 每次实验完成后更新本文件，并将重要代码状态提交到清晰命名的Git分支；不要让checkpoint与代码语义错配。
