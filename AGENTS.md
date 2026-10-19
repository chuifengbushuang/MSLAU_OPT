# MSLAU-Net 长期项目规则与实验记录

> 本文件是 Codex 的长期项目记忆。每次代码修改或实验完成后都必须更新。
> 本文件位于 Git 仓库根目录，是 Mac、`.120`、`.194` 三端同步时的唯一权威版本。
> `handover.md` 仅用于一次性交接本轮对话，不替代本文件。
> 最近更新：2026-10-19。

## 1. 长期目标

- 2026-10-19 用户明确当前第一目标：设计NAF，使其在同环境、同协议的匹配对照中稳定优于baseline；历史0.878616的精确复现为第二目标，不作为新模块研究的前提。最终交付含none/NAF开关、完整配置和证据的同一套代码，由原实验者在其环境复验；不承诺跨环境必然提升。
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

### le 账号后续输出约定（2026-10-19 用户指定）

- 模型权重：`/data/models/le/mslau-net/checkpoints/<run_id>/`。
- 实验记录：`/data/models/le/mslau-net/experiments/<run_id>/`，训练日志放其中的`logs/`；启动命令、完整解析参数、代码commit/diff、环境/运行时、输入哈希、逐轮指标和评估结果均归此处。
- 可视化：`/data/models/le/mslau-net/visualization/<run_id>/`，曲线及固定样本预测对照按需生成。
- 三处使用同一唯一run_id，包含方案、seed、重复编号及时间；实验manifest记录三个绝对路径、权重选择指标/epoch/哈希及运行状态。汇总索引建议放`experiments/index.csv`。
- 用户已在服务器创建三个根目录。后续推荐入口`python -m tools.run_experiment`自动分配三类目录，生成manifest与加锁index.jsonl；透传旧训练/评估CLI，不改变模型及训练保存策略。直接调用旧脚本仍兼容，但不会自动统一布局。入口支持dry-run、独立repo/python、单卡预检、源码/环境/输入哈希及完成/失败记录；完整用法见`docs/experiment_outputs.md`。
- 所有运行（包括失败/负向）的轻量记录保留；关键基线、对照和最终方案各seed的最佳IoU权重保留，不只保留最高分一次。探索权重清理前先列清单并确认记录完整及引用关系。现有原代码可能重复保存同一最佳模型；核对模型内容后才制定去重操作，不能仅凭文件名或字节哈希判定。
- 原始预训练权重保持原路径，以路径及内容哈希引用。当前checkpoint为模型state_dict，不包含完整断点恢复状态；需要恢复训练时须另外实现optimizer/scheduler/epoch/随机状态等保存。
- 用户本轮未要求立即删除或搬迁旧文件；不得据此批量清理。后续无需重复激活环境或cd；每次要求用户运行下一步时附可执行命令。

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

### 历史 legacy 训练参考（不适用于P5.1/NAF）

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

2026-10-15 已读取用户回传的 P5.1/NAF2D 三 seed 训练、独立验证评估和真实特征诊断。
历史最佳结果保留在下表；当前环境的匹配对照为 P5.1 **0.869595±0.001532**、
四级 NAF2D **0.862984±0.001480**（样本标准差，seed42/1234/2026）。
NAF 三个 seed 均下降，平均下降 0.006610；后续应以当前匹配基线判断收益，不能把历史 0.878616 单次峰值作为唯一对照。
阶段消融 seed1234 已完成：仅 P1/P2 为0.866889，仅 P3/P4 为0.871696。
P3/P4-only 补齐 seed42/2026 后为 **0.867907±0.003383**，相对匹配 baseline 均值下降0.001687，
仅1/3 seed提升，未通过预设的初步支持条件；不升级为有效配置。
0.874656同权重微调对照已完成：none为0.875457，P3/P4 NAF为0.871617，NAF低0.003840。
这是独立的单seed微调结果，不与从官方Hiera开始的三seed均值混排，历史最佳三seed结果保留。

### Kvasir

2026-10-19 环境对照另行归档：同一原始代码、seed1234各重复2次，原环境均值0.873476±0.003455、当前环境0.871155±0.002638（重复运行样本标准差，非三seed统计）。原环境有探索性优势但范围重叠，未确认环境因果或复现历史0.878616；不得用这组同seed均值替换下表历史三seed统计。

| 模型 | 最佳 val mIoU | Epoch | 结论 |
|---|---:|---:|---|
| P5.1 官方式 SAM2-UNet | **0.872802±0.005118**（best 0.878616） | 10/17/9 | 当前三seed最高均值；3/3 seed超过P3 |
| P3 reverse-only（无wavelet edge） | 0.859864±0.010269（best 0.871511） | 39/82/57 | 上一代主模型；2/3 seed超过P0 |
| P3 reverse-only + 352（seed1234） | 0.860625 | 20 | 比同seed 256低0.010886；停止放大输入 |
| P4 cascade + Lovasz + DINOv2（seed1234） | 0.862562 | 139 | Precision升、Recall明显降；未超过P3 |
| P4 no-final-reverse（seed1234） | 0.860338 | 52 | Recall恢复但假阳性增多；P4路线暂停 |
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
- 正式结果：87m03s完成。Best val IoU **0.862562@139**，对应MainLoss0.277615；best MainLoss0.192077@17，对应IoU0.850508；最终train/val IoU为0.9496/0.8533，泛化差距0.0963。
- 稳定性：最后20轮val IoU为0.854000±0.000925；114轮达到0.85、仅6轮达到0.86。峰值比同seed P0高0.001979，但比P3 reverse-only低0.008949，且P3最后20轮为0.858555±0.000678。
- 验证指标：mIoU0.862559、Dice/F1 0.9177、Precision0.9435、Recall0.9053、Accuracy0.9771、FPS102.68。相对P3的Precision0.939108/Recall0.924487，P4进一步提高Precision但Recall下降约0.0192。
- 面积诊断：P4总预测面积仅为GT的92.30%，P3为95.27%；9张预测面积不足GT的80%，出现1张空预测。四个面积组IoU为0.809031/0.885418/0.890622/0.865166，仅第二组高于P3；最大面积组预测面积仅为GT的88.57%。
- 逐图对比：相对P3胜57、负63；7张提升超过0.05，14张下降超过0.05，平均差-0.008951、中位差-0.001341，少量灾难性欠分割进一步拉低均值。
- reverse参数：D3→D2、D2→D1、D1→final scale分别学习到0.1751/0.1468/0.1206；三级纠错叠加与欠分割现象相关，但尚不能证明因果。
- 蒸馏：DistillLoss从前10轮平均0.4904降至最后20轮0.2050；固定0.1权重在训练末期仍贡献约0.0205，相对MainLoss0.0547并不小，可能持续约束后期域内适配。
- 证据：run目录 `/data/models/zsy/mslau-net/runs/kvasir_p4_cascade_reverse_352_lovasz_dinov2_vits14_dw010_c96_aux010_020_010_bnd010_b8_e200_seed1234_gpu0_nw8_20260810_080951`；best checkpoint为`checkpoints/best_iou_0.862562_epoch_139_0.277615.pth`；代码提交`38c2e22`。
- 结论：P4组合未刷新P3峰值，不能升级为主模型；直接失败机制是Recall下降和预测面积收缩。第一优先级做单变量“关闭D1→final reverse”，其余P4配置不变。

### 2026-08-11：P4 no-final-reverse 单变量实验

- 假设：P4 的第三级 D1→final reverse residual correction 与前两级纠错累积，导致预测区域收缩；只关闭该级可能恢复Recall，同时保留cascade的逐级空间纠错。
- 具体操作：在独立分支 `kvasir-p4-no-final-reverse` 新增默认兼容的 `p4_use_final_reverse`/`--disable_p4_final_reverse` 开关；本实验仅绕过final reverse，保留模块参数以兼容旧P4 checkpoint。
- 控制变量：与P4完全一致；Kvasir 880/120、seed1234、352、batch8、200 epoch、final 0.35 BCE+0.35 Dice+0.30 Lovasz、DINOv2 weight0.1、encoder LR5e-5、decoder/adapter LR2e-4、warmup5。
- 验证：`py_compile`、CLI与diff检查通过；旧P4 0.862562 checkpoint strict加载；随机batch2确认前两级reverse与final head梯度非零、final reverse梯度为None；真实8/4样本含DINOv2的一轮训练和checkpoint推理通过。
- 正式结果：80m55s完成。Best val IoU与best MainLoss同时为 **0.860338@52/0.195151**；最终train/val IoU为0.9478/0.8446，泛化差距0.1032。
- 稳定性：最后20轮val IoU为0.844085±0.000624；仅20轮达到0.85、1轮达到0.86。比原P4峰值低0.002224、比P3 reverse-only低0.011173、比同seed P0低0.000245，后期退化显著加重。
- 验证指标：mIoU0.8603、Dice/F1 0.9156、Precision0.9226、Recall0.9214、Accuracy0.9747、FPS110.79。相对原P4，Recall提高0.0161但Precision下降0.0209，简单删除final reverse只是把偏差从欠分割移向假阳性。
- 面积与阈值：总预测面积/GT从原P4的92.30%升至97.57%，不足GT 80%的样本从9降至4、空预测从1降至0；四面积组IoU为0.821609/0.859122/0.891147/0.869431。验证集阈值0.575的post-hoc最佳IoU仅0.861025，无法追平原P4或P3，问题不只是阈值校准。
- 逐图对比：相对原P4胜63、负57、中位差+0.000807，但平均差-0.002232；多数样本微升、少数严重假阳性样本拉低均值。
- 结论：假设被否定。final reverse确实收缩预测，但也提供有效假阳性抑制和后期结构约束；不补其他seed，不继续删除前两级reverse。P4路线暂停，性能主线回到P3 reverse-only。
- 证据：run目录 `/data/models/zsy/mslau-net/runs/kvasir_p4_no_final_reverse_352_lovasz_dinov2_vits14_dw010_c96_aux010_020_010_bnd010_b8_e200_seed1234_gpu0_nw8_20260811_023028`；best checkpoint为`checkpoints/best_iou_0.860338_epoch_52_0.195151.pth`；代码提交`83d0f61`。

### 2026-08-11：P3 reverse-only + 352 输入单变量实验

- 假设：P3 reverse-only已具备最佳Precision/Recall平衡，仅提高输入分辨率可增强小目标与边界细节，而不引入P4 cascade、Lovasz和DINOv2的混杂影响。
- 具体操作：使用P3 `progressive_wavelet` decoder，关闭wavelet edge、保留reverse attention；唯一实验变量为输入从256改为352。分支为`kvasir-p3-reverse-only-352`。
- 控制变量：Kvasir 880/120、seed1234、batch16、0.5 BCE+0.5 Dice、D2/D3/boundary权重0.2/0.1/0.1、无Dropout、无DINOv2、encoder LR5e-5、decoder LR2e-4、warmup5、200 epoch。
- 验证：旧P3 0.871511 checkpoint在352实例上strict加载；随机batch16完整前后向峰值显存约8.9GiB，reverse与final head梯度非零；无需降低batch。完整880/120一轮冒烟train/val IoU为0.5628/0.6690，checkpoint推理通过，FPS128.12。
- 正式结果：53m03s完成。Best val IoU **0.860625@20**，对应MainLoss0.084584；最终train/val IoU为0.9461/0.8465，泛化差距0.0996。最后20轮val IoU为0.845325±0.001063，仅28轮达到0.85。
- 对比与机制：比同seed P3-256低0.010886，训练时间增加约50%；Precision从0.9391降至0.9261而Recall仅从0.9245降至0.9227。四个GT面积四分位IoU全部下降，预测面积/GT从0.9527升至0.9715，小中目标假阳性增加。阈值扫描最佳0.525仅提升约0.0001，排除简单校准问题。
- 证据：run目录 `/data/models/zsy/mslau-net/runs/kvasir_p3_reverse_only_352_noedge_c96_aux020_010_boundary010_b16_e200_seed1234_gpu0_nw8_20260811_064300`；best checkpoint为`checkpoints/best_iou_0.860625_epoch_20_0.084584.pth`。
- 结论：假设被否定。当前编码器/decoder不能有效利用额外分辨率，352放大背景纹理并加速过拟合；不补其他seed，停止单纯分辨率扩展。

### 2026-08-11：P5 SAM2 Hiera-L Adapter + P3 reverse-only

- 假设：P3的主要剩余瓶颈是小数据下的编码器表征和少数灾难性困难样本，而不是输入分辨率；冻结SAM2 Hiera-L并只训练轻量Adapter，可提供更强四尺度表征，同时保留P3已验证的reverse-only解码机制。
- 具体操作：新增`sam2_hiera_large`编码器和显式`p5_hiera_reverse`模式；Hiera-L主体冻结且始终eval，四级输出144/288/576/1152通道，各接一个零初始化瓶颈残差Adapter；随后接原P3 progressive decoder，强制关闭wavelet edge、保留coarse detach reverse attention及D2/D3/boundary监督。训练/测试CLI均支持P5 checkpoint语义。
- 权重：官方timm Hiera-L safetensors位于`/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors`，大小848661592 bytes，SHA256 `7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c`；加载时严格映射586个backbone键并忽略分类head。
- 控制变量：Kvasir 880/120、seed1234、输入352、batch16、num_workers8、150 epoch、0.5 BCE+0.5 Dice、D2/D3/boundary权重0.2/0.1/0.1、channels96、无Dropout；Adapter LR1e-4、decoder LR5e-4、warmup5后cosine。
- 验证：`git diff --check`、`py_compile`通过；旧P3 0.871511 checkpoint strict加载通过；随机batch16与真实Kvasir batch16前后向通过。冻结backbone无梯度，四级Adapter和decoder梯度非零；总参数214279297、可训练参数2130001；随机batch16峰值显存约4.26GiB。
- 正式运行：GPU0，PID `858742`；run目录 `/data/models/zsy/mslau-net/runs/kvasir_p5_hiera_l_frozen_adapters_p3_reverse_352_c96_b16_e150_seed1234_gpu0_nw8_20260811_0817`。Epoch0 train/val IoU为0.4977/0.6010，GPU利用率100%、显存约7.17GiB，已进入Epoch1。
- 中止结果：运行至Epoch88后判定失败并终止；当时best val IoU **0.8177@88**，Epoch66～85平均约0.7995±0.0090，train/val gap约0.10，从未达到0.82。相对同seed P3-256低约0.0538，相对P3-352低约0.0429。
- 失败机制：该版只在四个stage最终输出后放Adapter，无法修改Hiera内部注意力与特征形成；冻结主干导致训练IoU也仅约0.91，同时P3 decoder对固定特征过拟合。352是次要负贡献，不能解释主要差距。
- checkpoint：旧脚本仅在训练全部结束时统一保存；本次中止前`checkpoints`为空，因此没有可复用权重。后续已改为每次刷新best IoU原子覆盖保存`kvasir_best_model.pth`。
- 结论：该“stage-output Adapter + P3 reverse”设计停止，不补seed。代码分支`kvasir-p5-hiera-reverse`，提交`8d7b95f`。

### 2026-08-11：P5.1 官方式 SAM2-UNet 基准

- 假设：先严格验证SAM2 Hiera编码器在当前880/120协议上的价值，再决定是否接P3；上一版失败主要来自Adapter位置和decoder语义错配。
- 具体操作：按官方源码将48个Hiera block分别包装为prompt Adapter：`dim→32→dim`后与block输入相加，原Hiera参数全部冻结；四级特征经RFB压到64通道，使用经典U形decoder和两路深监督，不使用reverse、wavelet或独立boundary head。
- Loss与训练：加入官方structure loss（边界加权BCE+加权IoU），三路输出等权；Kvasir 880/120、352、batch12、num_workers8、AdamW LR1e-3、weight decay5e-4、cosine、20 epochs、seed1234。数据划分保持不变，但仍使用本项目现有增强。
- 可靠性修复：训练改为每次刷新best val IoU时，把CPU state_dict原子写入固定checkpoint，避免中断后没有权重；训练结束仍生成带指标的best loss/best IoU文件。
- 验证：语法与diff检查通过；旧P3 0.871511 checkpoint strict加载通过；官方Hiera权重严格加载；随机batch12与真实Kvasir batch12前后向通过。总参数216419043、可训练4269747，其中block Adapter 1714416；Hiera原参数无梯度，Adapter和decoder梯度非零；batch12峰值显存约14.32GiB。
- seed1234结果：21m24s完成，best val IoU **0.878616@17**，Dice0.9290、Precision0.9199、Recall0.9461、Accuracy0.9824、FPS35.64；最终train/val为0.9062/0.8766。相对同seed P3提高0.007105，相对P0提高0.018033。
- seed42/2026补跑：分别为 **0.868977@10** 和 **0.870813@9**；独立复评Dice为0.9210/0.9256，Precision为0.9234/0.9280，Recall为0.9384/0.9356。两组训练均完整结束且checkpoint可严格加载。
- 三seed统计：seed42/1234/2026为0.868977/0.878616/0.870813，均值 **0.872802±0.005118**；相对P3三seed配对提高+0.013008/+0.007105/+0.018701，平均提高0.012938且3/3 seed提升；相对P0平均提高0.019798。三seed平均Dice0.9252、Precision0.9238、Recall0.9400，收益主要来自缓解P3欠分割。
- 面积分组：P5.1三seed从最小到最大GT面积四分位IoU为0.800657/0.895469/0.907776/0.887301；相对P3分别为+0.002680/+0.010673/+0.007913/+0.030481。小目标绝对值最低且提升最小，下一步应补高分辨率局部细节，但不能破坏中大目标与全局结构收益。
- 证据：seed1234 run目录`/data/models/zsy/mslau-net/runs/kvasir_p5a_sam2unet_hiera_l_blockadapter32_rfb64_structure_352_b12_e20_seed1234_gpu0_nw8_20260811_0906`，最佳checkpoint为`checkpoints/best_iou_0.878616_epoch_17_0.304569.pth`；seed42/2026 run目录后缀分别为`seed42_gpu0_nw8_20260811_0941`和`seed2026_gpu1_nw8_20260811_0941`。
- 结论：P5.1已替代P3成为当前Kvasir主模型；分支`kvasir-p5-sam2unet-official-adapter`，核心代码提交`6602594`。后续以P5.1为冻结基线，新分支单变量改善小息肉。

### 2026-10-13：P5.1 四级独立 AFFE 代码与合成验证（未正式训练）

- 假设：在 RFB 后以输入自适应二维频域门控增强每级特征，可提供可消融的频域研究接口；本轮仅验证工程与数值，不检验分割收益。
- 具体操作：新增 frequency_enhancement.py 与共享 CLI；四级独立 AFFE、实数半谱门控、共轭约束、近恒等残差、可选中性分段正则；只接在 RFB 后，不改 Hiera/Adapter、Up、head、融合或 Structure Loss。默认 frequency_mode=none、frequency_reg_weight=0；补充严格/审计式 checkpoint 加载。
- 控制变量：未读取数据做训练/调参，split/增强/监督权重/优化器超参数不变；合成梯度检查仅3步。最终硬件为用户指定的共享4×RTX3090，需逐次协调可用卡。
- 结果：最终完整合成回归40 passed、1 skipped（真实checkpoint）、1依赖弃用警告；本地 CPU 模块/Decoder 检查通过；本地 RTX5070Ti 真实 Hiera 随机权重全链路、冻结参数全量不变、48个Adapter及RFB/AFFE梯度、FP16/BF16通过；旧提交Decoder的none输出逐元素一致，模拟旧state_dict迁移只缺20个AFFE键。实际旧checkpoint和预训练权重未提供，不能称已验证真实兼容。
- 对比：新增8708参数（每级2177），完整参数216419043→216427751；四级初始相对输出变化约0.07%–0.11%；FP32 FFT/IFFT最大误差≤9.54e-7。本地B=2完整模型前向46.24→50.02ms，前后向121.53→137.34ms，不代表3090性能。
- 结论：工程实现已具备本地合成证据，研究效果未知，当前最佳P5.1结果不变。3090环境/真实权重验收待补；连接le@10.243.96.120:22超时，未登录服务器或使用其GPU。不得启动正式训练。
- 证据：分支feat/p5-naf-frequency-enhancement，基线81b19e2；docs/frequency_enhancement/下method_design、interface_contract、verification_report、future_ablation_plan及原始JSON/XML；tests/、tools/verify_frequency.py。handover.md本地缺失已报告。

### 2026-10-13：AFFE 的3090目标环境验收（用户执行并回传）

- 假设：本地AFFE实现在目标torch2.1.2+cu118环境可运行，真实P5.1权重可审计迁移。
- 具体操作：用户在le账号的mslau-naf环境、RTX3090 GPU0运行604b1fd；40 passed/1 skipped的合成回归后，指定真实权重定向测试2 passed/5 deselected，补齐checkpoint验证。无正式训练。
- 控制变量：B=2、352、FP32性能测量，预热3次/重复10次；冻结主干、原Adapter/RFB/Head和损失语义保持不变。数据沿用/data/dataset/zsy/mslau-net/Kvasir-SEG与CVC-ClinicDB。
- 结果：none严格加载1180键；naf2d加载全部1180旧键，仅缺预期20个AFFE参数、无unexpected。预训练Hiera合成梯度/冻结检查通过；TypedStorage弃用警告未导致失败。
- 对比：前向57.7094→60.5151ms（+4.86%）；前后向145.3039→160.2846ms（+10.31%）；峰值增量分别8.56/27.20MiB。计时不含optimizer.step及真实数据加载，不代表正式训练batch显存。
- 结论：此前3090与真实checkpoint待验项已补齐，分割效果未知，当前最佳结果不变。用户更新GPU规则：现场确认空闲即可使用，无需再与同学协商。
- 证据：用户回传终端输出；远程父目录/home/le/WorkSpace/20261012_PolypSegmentation/中的affe_3090_tests_01.log、affe_3090_pretrained_tests_01.log、affe_3090_verification_01.json。完整JSON尚未传回本地，报告明确区分回传摘要与直接读取。

### 2026-10-13：3090完整证据归档

- 具体操作：实际读取用户同步的完整JSON和两份日志，原样归档至docs/frequency_enhancement，并生成evidence_3090.sha256；此前仅收到摘要的归档限制解除。
- 结果：参数量和1180旧键/20新键审计与终端摘要一致；全部数值有限、各级初始梯度非零。四级FFT最大误差1.4305e-6，小于2e-6容差；AFFE整体前向4.060ms、前后向12.033ms。
- 结论：本轮工程证据归档完成，无模型代码改动、无正式训练，当前最佳分割结果不变。详见verification_report.md。

### 2026-10-13：NAF2D最终定位审计与增量复测

- 假设：现有单尺度NAF已符合四级RFB后独立增强的最终设计，需要补齐职责、旁路与测量隔离证据，不重写算法。
- 具体操作：统一方法名Scale-specific NAF-inspired Adaptive Frequency Feature Enhancement和E/P/P'/D命名；模型仅局部变量更名，FFT/门控/残差/正则核心文件未改；修复verify_frequency诊断图滞留和单级输入梯度累积；新增21项回归，修订四份文档。
- 控制变量：无数据集训练、无val调参、无新监督或融合；Hiera/Adapter/RFB/Up/三个Head及默认reg=0不变。GPU使用按用户最新规则，现场空闲即可使用。
- 结果：CPU48 passed/14 skipped；本地5070Ti完整回归61 passed/1 skipped（本地无真实checkpoint），语法与diff检查通过。全部16种stage开关、train/eval阶段隔离、全关闭三头与基线精确一致、分割/正则梯度来源通过；核心满足最终研究设计。
- 对比：参数仍为每级2177/合计8708；本轮B2 FP32随机权重全模型前向46.058→49.541ms，前后向113.235→123.233ms；峰值增量32.04/23.11MiB。测量版本2清理额外图和梯度，旧结果保留但绝对显存有历史测量限制，不将耗时变化解释为算法加速。
- 结论：无分割/边界/去噪收益结论，当前最佳结果不变；历史3090真实1180键加载与20新键迁移证据继续有效，本轮未在3090重跑，新脚本目标环境复测待补。
- 证据：起点4199ebc，分支feat/p5-naf-frequency-enhancement；docs/frequency_enhancement/audit_v2_cpu.xml、audit_v2_5070ti.xml、audit_v2_5070ti.json和audit_v2_evidence.sha256；verification_report追加本轮，不覆盖原始证据；handover.md仍缺失。

### 2026-10-15：P5.1/四级 NAF 三 seed 与真实特征诊断归档

- 假设：RFB 后四级独立 NAF 能在不改 P5.1 其余训练配置的情况下提供稳定分割收益；本次同时检查退化是否伴随门控大面积饱和或残差系数异常增大。
- 具体操作：读取用户同步的六组完整训练日志、最佳 checkpoint 的独立验证评估、三个 NAF checkpoint 的 alpha 和 120 张验证图像逐图频域统计；本机未启动训练、未修改模型或随机性设置。
- 控制变量：Kvasir 880/120、352、batch12、workers8、Structure Loss 三头等权、AdamW（Adapter/decoder LR1e-3、weight decay5e-4）、cosine、无 warmup、20 epochs；NAF 四级启用、alpha 初始0.1、reg=0。训练代码49aa8b7；同环境、预训练及 split 哈希。Albumentations 增强序列不保证逐步一致，按用户决定以重复 seed 评价，不追求历史轨迹逐步复刻。
- 训练结果：seed42/1234/2026 baseline best IoU 为0.871332@16 / 0.869015@18 / 0.868437@19；NAF 为0.861288@8 / 0.863653@17 / 0.864012@12。均值与样本标准差分别0.869595±0.001532、0.862984±0.001480；逐 seed 差为-0.010044 / -0.005362 / -0.004425。独立评估的三 seed 平均 Dice 为0.9236→0.9188，Recall0.9448→0.9384；Precision0.9187→0.9167。均值指标据四位小数评估日志计算。
- 诊断结果：三个 NAF 模型均严格加载1200键、无 missing/unexpected，逐图统计覆盖相同120个不同样本且数值有限。alpha 范围0.1091～0.1272，未接近上界。P1/P2/P3/P4 的每 seed 平均相对输出 L2 变化范围依次为4.40%～4.88% / 3.53%～3.97% / 3.82%～4.12% / 1.73%～3.63%；已乘 alpha。门控权重距0.5/1.5边界小于0.01的比例，各阶段各 seed 的均值均≤0.288%，单图最大0.852%。P1～P3三个 seed 均表现为低频平均M<1、中高频平均M>1；P4三个频段平均M均<1。这些是未乘 alpha、非能量加权的门控均值，不等价于具体频段能量变化或阶段因果贡献。
- 结论：否定“当前四级配置有稳定收益”，暂不支持用大面积门控饱和或模块完全不起作用解释退化。先做仅 P1/P2（1,1,0,0）和仅 P3/P4（0,0,1,1）的两组 seed1234 探索性训练，唯一变量为启用阶段；仍从官方 Hiera 初始化，不从已训练 baseline/NAF checkpoint 续训。单 seed 只用于筛选，有候选收益后补 seed42/2026；恢复到 baseline 以下不算有效提升。尚未执行该阶段消融。
- 证据：本地父目录中的 naf_training_HYrMUMN0、naf_seeds42_2026_lhNT5iVg、p51_review_YvvoiuPi、naf_diagnosis_NcEnvNFa、naf_features_BGbBNFsn；服务器对应父目录为 /home/le/WorkSpace/20261012_PolypSegmentation。训练 checkpoint 均为各 run 下 checkpoints/kvasir_best_model.pth；seed1234 位于第一目录，seed42/2026 位于第二目录。当前用户环境为 le 账号的 mslau-naf，用户已在仓库根目录，后续命令无需重复激活或 cd；每次提出需用户执行的下一步须附具体命令。

### 2026-10-15：NAF 阶段消融 seed1234（用户执行并回传）

- 假设：四级全部启用可能不是适合当前任务的配置；比较仅 P1/P2 和仅 P3/P4，以阶段开关为唯一训练配置变量。
- 具体操作：读取 naf_stages_fSUUEZf4 两组完整20 epoch 训练日志和最佳 checkpoint 独立验证评估；本机未训练、未修改模型。
- 控制变量：训练代码49aa8b7、无代码差异、环境与前一轮相同；官方 Hiera 及 train/val split 三个 SHA256 值相同（哈希文件仅绝对/相对路径文本不同）。参数与先前 naf_all_seed1234 逐项核对，仅 frequency_enabled_stages、output_dir、log_dir 不同。seed1234、352、batch12、workers8、三头等权 Structure Loss、Adapter/decoder LR1e-3、AdamW weight decay5e-4、cosine、无 warmup、20 epochs、alpha初始0.1、reg=0 均保持一致；从官方 Hiera 初始化，无 init_checkpoint。
- 结果：P1/P2-only best IoU **0.866889@12**，Dice0.9199、Precision0.9180、Recall0.9417，最后5轮val IoU均值0.86356，最终train/val为0.9023/0.8655，21m47s。P3/P4-only best IoU **0.871696@15**，Dice0.9235、Precision0.9198、Recall0.9378，最后5轮均值0.86960，最终train/val为0.9017/0.8703，22m20s。两个 checkpoint 均严格加载1200键，启用开关与训练一致、无 missing/unexpected。最后5轮统计据四位小数日志计算。
- 对比：同 seed baseline best IoU0.869015、最后5轮均值0.86402，四级 NAF best0.863653。P1/P2-only 对 baseline 为-0.002126；P3/P4-only 对 baseline 为+0.002681、对四级 NAF 为+0.008043，最后5轮均值对 baseline 为+0.00558。P3/P4-only 的 Precision 对 baseline0.9253下降0.0055，Recall对0.9368升0.0010；不能宣称所有指标改善。
- 结论：P3/P4-only 出现值得复核的单 seed 正信号，不能据此断言浅层有害或证明因果机制；不立即组合正则、alpha/LR 等修改。下一步只补该配置的 seed42/2026，每组仍20 epochs。预先以三 seed 平均IoU高于匹配 baseline0.869595、至少2/3 seed逐项提升作为初步支持条件，同时报告样本标准差、Dice/Precision/Recall；n=3不等于统计显著或外部泛化。尚未执行补种子。
- 证据：本地 /home/lien/WorkSpace/PolypSegmentation/naf_stages_fSUUEZf4；远程 /home/le/WorkSpace/20261012_PolypSegmentation/naf_stages_fSUUEZf4；各组 command.txt、logs/kvasir_train.log、console.log、evaluation_command.txt、evaluation.log。最佳权重为各组 checkpoints/kvasir_best_model.pth。GPU快照中P3/P4启动时其他任务位于GPU1/2，不将其耗时与FPS差异解释为模块效果。

### 2026-10-15：原分支与当前 none 基线的语义复核

- 用户问题：原代码独立复现0.874656高于当前seed1234 baseline0.869015，是否接入NAF时改变了基线。
- 复核：直接比较81b19e2与49aa8b7。训练文件新增frequency参数、可选初始化和reg分支；none、reg=0且无init_checkpoint时不启用这些新增计算。loader、loss、split、Hiera/Adapter实现未改，优化器/调度/监督语义未改。两次启动参数的有效训练配置一致，pip freeze逐字一致、官方Hiera及split哈希相同；路径与新增关闭开关不同。
- 边界：源码文件有差异，不能称逐文件原封不动。none不实例化NAF、不新增参数或FFT；已有独立原版decoder测试验证相同初始化/权重和输入下三头逐元素一致，不能扩大成“完整训练轨迹已逐步验证一致”。输入探针已证实增强序列不固定，但不把两次最终差距全部归因某个已量化原因，也不因单次原分支结果较高就判定其期望性能更强。
- 开启NAF：RFB→NAF→原U形decoder，形状和接口兼容；训练时原有Adapter/RFB/decoder继续更新，NAF改变特征与反向梯度，Hiera主干继续冻结。训练后关闭NAF不能恢复一个独立训练baseline的权重。当前baseline确为frequency_mode=none，不是naf2d全阶段关闭。
- 数值：原分支独立复现0.874656比当前baseline高0.005641；P3/P4候选0.871696比该原分支运行低0.002960，因此其+0.002681只相对当前匹配baseline，尚不能称超过该原代码复现或历史0.878616。
- 下一步：不新增训练配置，用原分支checkpoint在当前none代码评估、当前none checkpoint在原分支代码评估，分别应与已有评估0.8747、0.8690核对。这项完整模型交叉评估尚未执行，不写作通过；已启动的补种子任务保持原配置。

### 2026-10-15：原分支/当前 none 完整模型交叉评估通过；规划同权重微调对照

- 假设：加入NAF接口未改变none模式下完整模型的评估指标；进一步探索从原分支复现0.874656的强起点加入NAF是否有收益。
- 具体操作与结果：读取p51_cross_eval_orkQm2hg两份日志。原分支0.874656权重在当前none代码下为IoU0.8747、Dice0.9275、Accuracy0.9774、Precision0.9306、Recall0.9381、F1 0.9275；当前baseline权重在原代码下为IoU0.8690、Dice0.9222、Accuracy0.9757、Precision0.9253、Recall0.9368、F1 0.9222。两次交叉评估各自与已有评估的六项均值和标准差在日志显示精度下全部一致；当前代码严格加载1180键、无missing/unexpected。
- 结论边界：支持这两份checkpoint在两套代码中的完整推理指标一致，不等价于逐图logits逐元素验证或完整训练轨迹一致。原先两次独立训练差距不能解释为“当前评估代码把相同权重算低”。
- 新协议（尚未执行）：按用户要求，以original_p51_seed1234_tclaP5zu/checkpoints/kvasir_best_model.pth为同一初始化，比较none继续训练与naf2d仅P3/P4（0,0,1,1）微调。两组seed1234、各额外20 epochs、Adapter/decoder LR均1e-4、cosine、无warmup、batch12、352、workers8、三头Structure Loss等权、weight decay5e-4；NAF alpha初始0.1、reg=0。1e-4是较小微调学习率的经验起点，未称最优。沿用官方Hiera参数及冻结方式，但init_checkpoint随后载入全部1180个已训练参数/缓冲项；仅NAF的20项为新初始化（实现仍实例化四级，启用两级）。
- 公平比较：none与NAF均从完全相同的0.874656 checkpoint初始化，额外训练预算与优化器/调度重建协议相同；init_checkpoint不恢复optimizer/scheduler，因此属于二阶段微调而非断点续训。分别比较微调对照和原始0.874656，不能把延长训练收益全部归于NAF，也不与从官方Hiera开始的三seed均值混排。单seed探索，结论仍需重复验证；原来的P3/P4补种子任务保持原配置。
- 证据：本地/home/lien/WorkSpace/PolypSegmentation/p51_cross_eval_orkQm2hg；服务器对应目录/home/le/WorkSpace/20261012_PolypSegmentation/p51_cross_eval_orkQm2hg；训练CLI现有--init_checkpoint与严格/审计式迁移接口可执行新协议，无需修改模型代码。输出计划为新建p51_ckpt_finetune_XXXXXXXX目录，原checkpoint保留。

### 2026-10-15：P3/P4-only 补齐 seed42/2026，三 seed 未通过

- 假设：检验seed1234的0.871696单次正信号能否重复；预设条件为三seed平均IoU高于baseline0.869595且至少2/3 seed提升。
- 具体操作：读取用户同步naf_p34_seeds42_2026_88uaX1ZS，核对两个训练各880/120样本、完整20 epochs、最佳checkpoint严格加载1200键且无missing/unexpected。代码49aa8b7、code_diff.patch为空、环境与前轮逐字一致，Hiera和split哈希相同；启动参数相对P3/P4 seed1234仅seed与输出路径不同。无init_checkpoint，均从官方Hiera预训练初始化。
- 控制变量：仅P3/P4开启（0,0,1,1）、alpha初始0.1、reg=0、352、batch12、workers8、三头等权Structure Loss、Adapter/decoder LR1e-3、AdamW weight decay5e-4、cosine、无warmup、20 epochs；本机未训练或改模型。
- 结果：seed42 best IoU **0.865188@19**，Dice0.9211、Precision0.9255、Recall0.9331，最后5轮IoU均值0.86250、最终train/val0.8996/0.8652，22m08s。seed2026 best IoU **0.866838@13**，Dice0.9229、Precision0.9226、Recall0.9368，最后5轮均值0.86490、最终train/val0.9046/0.8651，22m02s。结合seed1234的0.871696@15，三seed均值 **0.867907±0.003383**（样本标准差）。
- 对比：baseline seed42/1234/2026为0.871332/0.869015/0.868437，P3/P4-only逐seed差为 **-0.006144 / +0.002681 / -0.001599**，平均-0.001687（-0.1687个百分点），只有1/3提升。P3/P4均值高于四级NAF0.862984，但仍低于baseline0.869595。三seed平均Dice0.9236→0.9225、Precision0.9187→0.9226、Recall0.9448→0.9359（据四位小数独立评估日志）；均值表现更偏Precision、以Recall降低为代价，不能由汇总值断定小目标或边界是具体退化来源。
- 结论：未通过预设初步支持条件，seed1234的正信号没有得到另两seed支持；当前从官方Hiera训练的四级配置与P3/P4-only均无稳定正收益。不再扩展相同配置的重复种子或据此宣称改进；三seed观测波动较baseline大，不宣称稳定性提高。下一步优先回收此前已授权的0.874656同checkpoint两组微调对照，作为独立探索协议；其结果尚未同步，不假定已执行或完成，不据本次失败擅自停止已启动任务。
- 证据：本地/home/lien/WorkSpace/PolypSegmentation/naf_p34_seeds42_2026_88uaX1ZS；服务器对应/home/le/WorkSpace/20261012_PolypSegmentation/naf_p34_seeds42_2026_88uaX1ZS。seed1234在naf_stages_fSUUEZf4/naf_p34_seed1234。原始command.txt、logs/kvasir_train.log、console.log、evaluation.log、environment.txt、input_hashes.txt均保留；用户后续查询微调进度用只读命令，避免重复启动。

### 2026-10-15：0.874656同权重微调对照完成，当前NAF仍无正收益

- 假设：较强的完整P5.1初始化可能使新增P3/P4 NAF有正收益；以同checkpoint继续训练的none作为匹配对照，控制额外训练预算。
- 具体操作：读取用户同步p51_ckpt_finetune_NDtNP2kW。两组初始化路径相同，起点SHA256为975734e08eb664a4d40db703988d4e7ff9ab9b73c6f732e282869daf5ffdd08b；都加载1180旧键，none无missing，NAF仅缺预期20个新键、无unexpected。代码49aa8b7、diff为空、环境与上轮一致。启动参数只在frequency_mode和输出路径上不同，880/120样本、完整20 epochs，最佳checkpoint独立评估严格加载通过。
- 控制变量：seed1234、从original_p51_seed1234_tclaP5zu/checkpoints/kvasir_best_model.pth初始化、额外20 epochs、Adapter/decoder LR1e-4、AdamW weight decay5e-4、cosine、无warmup、352、batch12、workers8、三头Structure Loss等权；NAF仅P3/P4、alpha初始0.1、reg=0，优化器/调度重新初始化，Hiera主干继续冻结。
- 结果：none微调best IoU **0.875457@14**，Dice0.9283、Precision0.9301、Recall0.9402，最后5轮val均值0.87242、最后10轮0.87231，最终train/val0.9126/0.8709，24m27s。NAF微调best IoU **0.871617@0**，Dice0.9255、Precision0.9222、Recall0.9432，最后5轮0.87030、最后10轮0.86921，最终train/val0.9099/0.8707，24m16s。epoch0已包含一次完整训练轮次，不等于插入NAF后未训练时的评估；本次没有训练前的NAF初始评分。
- 对比：none对原始起点0.874656峰值略升0.000801（+0.0801个百分点）；NAF对匹配none为-0.003840（-0.3840个百分点），对原始起点为-0.003039（-0.3039个百分点）。NAF对none的Dice低0.0028、Precision低0.0079、Recall高0.0030（据四位小数评估日志），不支持用NAF带来整体提升。
- 结论：当前P3/P4配置在这一较强起点的联合微调中也没有正收益，不能把前述失败简单归为baseline起点低。单seed微调不作普遍无效或统计显著性结论；结合从官方Hiera训练的三seed失败，不再扩展相同配置训练。下一步只做现有NAF最佳checkpoint的推理旁路诊断：全部旁路、仅P3、仅P4，与已有P3/P4同时开启的0.871617比较，辅助定位门控当前直接作用；这不是独立重训消融，不能代替none训练结果，也不能单独证明退化原因。尚未执行旁路诊断，不宣称关闭后一定恢复。
- 证据：本地/home/lien/WorkSpace/PolypSegmentation/p51_ckpt_finetune_NDtNP2kW，服务器对应/home/le/WorkSpace/20261012_PolypSegmentation/p51_ckpt_finetune_NDtNP2kW；两组command.txt、logs/kvasir_train.log、console.log、evaluation.log；最佳checkpoint均为各组checkpoints/kvasir_best_model.pth。原始checkpoint与两组微调checkpoint均保留。

### 2026-10-15：微调checkpoint旁路结果与原论文机制复核

- 具体操作：读取p51_finetune_bypass_OPA0VuO1三次旁路和原P3/P4开启日志；使用同一NAF微调最佳checkpoint（已训练一轮后的epoch0），代码49aa8b7、严格加载1200键，参数开关为全旁路/仅P3/仅P4/两者开启。
- 结果：四组在日志显示精度下均为IoU0.8716、Dice/F1 0.9255、Accuracy0.9780、Precision0.9222、Recall0.9432；Recall标准差全旁路0.0884、其他0.0883，不能称逐图结果或logits完全一致。旁路后未在汇总指标上恢复原始0.8747或匹配微调baseline0.8755。
- 结论：不支持把微调退化直接归因于该checkpoint推理时的门控；也不能由四位小数汇总同分断言NAF完全无作用、权重未更新或门控恒为1。其他可训练权重/BN状态、训练轨迹与门控的联合适应均可能相关，原因未隔离。停止继续扩展同一配置训练，不自动改算法或启动新实验。
- 原文复核：完整阅读用户提供的Hu et al. IEEE TASE 2025论文主要正文，目视核对Fig.3/4/5与Table I/IV/V/VI/VII/VIII。NAF-MSNN是工业时序软测量整网，NAF子模块的核心为时间FFT、GAP→Linear→sigmoid→频率轴softmax、约90%概率抽样、尺度/频段相关分段L2约束、Linear/Padding/IFFT；整网还有多尺度卷积、随机跨尺度交换及GRU。现有二维方案改为连续有界软门控、幅值/坐标MLP、全频保留和近恒等残差；现有正则为对M偏离1的惩罚且正式训练lambda=0。它不是原算法的等价二维实现。
- 定位边界：最终定位附件50722e64明确仅借鉴单尺度过滤过程，禁止原跨尺度交换/GRU，并明确接受软门控和默认reg=0；这些是共同确认的工程/研究取舍，不是漏实现。当前实验检验的是新的二维适配假设，失败不等于否定原论文；不能以原整网效果背书当前NAF有效。后续先澄清可迁移机制和最小对照，再决定是否重设门控，不再把继续调开关当原机制复现。
- 原文效果：主比较热电MAE/RMSE/R²=0.052/0.058/0.871，聚丙烯0.029/0.038/0.881，PTA0.218/0.275/0.754。Table VII聚丙烯R²为MSNN+GRU0.547→固定去掉10%高频0.721→自适应Select0.825→加入交换0.881；PTA0.583→0.674→0.726→0.754。整网与单NAF贡献必须区分，不能将R²与本项目IoU混比。原文有尺度偏置文字相反、复数GAP及采样梯度/补零索引未完整定义等复现歧义，未获取作者可运行实现，不声称已复现原文结果。
- 证据：p51_finetune_bypass_OPA0VuO1；原PDF papers/Noise_Adaptive_Filtering_Neural_Network_Under_Multiscale_Features.pdf，SHA256 f67d4904456946e642a1d3d4c7a40d3398cc9dc692d7a5ed9d6d1d1c94dd58f3。方法说明补入docs/frequency_enhancement/method_design.md；本轮只更新记录/解释，无模型变更，无新增训练要求。

### 2026-10-16：历史 0.878616 原始训练日志与原分支 0.874656 复现审计

- 当前目标：用户优先复现原始 P5.1，暂不实施上一轮提出的 NAF 软选择/分段正则改造。已提供原分支 seed1234 独立重复3次的命令；尚未收到这3次结果，不宣称已经启动或完成。
- 证据：用户同步历史目录 `kvasir_p5a_sam2unet_hiera_l_blockadapter32_rfb64_structure_352_b12_e20_seed1234_gpu0_nw8_20260811_0906`，其中仅有 `train_console.log`、`logs/kvasir_train.log` 两个文件，checkpoint目录为空；无实际启动命令、运行环境快照或输入哈希。对照 `original_p51_seed1234_tclaP5zu` 的两份日志和已有command/environment/input_hashes。
- 操作：只读解析双方全部20轮；各自console与带时间戳日志的逐轮train/valid指标、学习率、最终best汇总一致；两次console的启动配置头逐字一致，两个参数组共40个学习率值全部一致且符合20轮无warmup的cosine序列。6602594至81b19e2的训练、模型、loader、loss、split及环境声明文件diff仍为空。本轮未修改模型或启动训练。
- 已确认：seed1234、880/120样本计数、sam2_hiera_large/p5_sam2unet、block_prompt瓶颈32、主干冻结、RFB64、输入352、aux D2/D3均1、edge=False、fusion=fixed、dropout0；总参数216419043、可训练4269747、Adapter1714416。样本计数相同不证明数据内容相同；batch12/workers8/Structure Loss由目录名和历史实验记录支持，原始日志未打印完整args，不能把所有启动参数均视为已被运行日志独立核实。
- 结果：历史best IoU **0.878616@17**、best主头val loss0.304569@17；复现best IoU **0.874656@14**、best主头val loss0.324911@18。峰值差0.003960（0.3960个百分点）。最后5轮val IoU均值分别 **0.87534/0.86888**，最后3轮0.877833/0.869733，均基于四位小数逐轮日志，不是跨运行统计。最后一轮train/val为0.9062/0.8766与0.9021/0.8679；历史14/20轮val较高，后期优势并非单轮孤立尖峰。
- 诊断边界：Epoch0已经经历完整一轮训练，不能用其差异证明初始权重不同。历史训练21m24s、复现28m04s，存在吞吐差异但不足以归因于GPU型号、卡数、库版本或AMP/TF32。日志没有记录这些运行时条件，亦没有记录实际预训练文件哈希、完整初始可训练参数、数据内容或增强随机状态；未定位唯一原因。此前历史checkpoint在当前环境复评仍为0.8786的证据继续有效。
- 下一步：保持原配置重复基线，并索取原实验环境实际导出和能找到的原始启动命令。若需要核对权重，建议在服务器CPU比较历史/复现checkpoint中排除prompt_learn后的`encoder.backbone.*`冻结参数/缓冲项内容指纹；无需上传checkpoint。整份训练后checkpoint哈希不同不能用来判定官方初始化不同。该指纹检查尚未执行。
- 产物：父目录 `analysis/p51_original_08786_vs_repro_08746_20261016/epoch_comparison.csv`、`comparison_summary.json`，含逐轮对照、两组学习率、原日志SHA256、已确认配置与未记录项；原文件保持原样。

### 2026-10-16：取得原实验者启动命令，有效训练参数与0.874656复现一致

- 新证据：用户提供原启动命令，工作目录`/home/zsy/projects/mslau-net`，解释器`/home/zsy/miniconda3/envs/mslau-net/bin/python`，CUDA_VISIBLE_DEVICES=0；命令输出目录仍以`<RUN_NAME>`占位，不视为逐字历史shell记录。
- 操作：从81b19e2源码的argparse定义安全构造解析器，补齐默认值后，与`original_p51_seed1234_tclaP5zu/command.txt`全部参数逐项比较。差异仅为train_txt/val_txt相对或绝对路径、output_dir/log_dir保存位置，以及disable_edge_guidance布尔值；其余解析结果完全一致，包括batch12、workers8、Structure Loss、352、20 epochs、seed1234、两组LR1e-3、D2/D3=1、boundary=0。未显式传入的warmup=0、fusion=fixed、dropout=0均与复现一致。分支内train/val内容SHA256与已有复现输入哈希一致；这不替代历史服务器文件的内容取证。
- 边缘标志：原提供命令没有`--disable_edge_guidance`，按源码属性应为True；历史原始日志却打印False，命令与日志有一处未逐字对应。P5模式创建edge_guidance=None，并在进入legacy边缘分支前直接返回，该标志只影响属性/日志，不影响P5模型参数、构建随机数或前向计算；不为其新增训练。复现命令的-u只改变日志缓冲。
- 当前结论：已无已知有效训练参数差异可以直接解释0.003960的峰值差。原环境与当前解释器分别为`/home/zsy/miniconda3/envs/mslau-net/bin/python`及`/home/le/miniconda3/envs/mslau-naf/bin/python`，实际依赖构建、cuDNN等尚未对照，不能据路径不同就断言环境是原因。下一步只读比较两套解释器的当前元数据；若原解释器不可访问，再由原实验者导出。当前导出不等于还原8月11日历史环境。
- 产物：`analysis/p51_original_08786_vs_repro_08746_20261016/reported_original_launch.txt`、`launch_comparison.json`。本轮未修改模型或启动训练；此前3次重复计划不增加新配置，未收到结果不宣称完成。

### 2026-10-16：两套环境对照完成，规划原始代码上的环境重复对照

- 证据：用户同步 `p51_env_compare_zX39oiez.log`；原解释器与当前解释器均可访问并成功导入torch。日志反映当前环境状态，不是2026-08-11运行快照。尚未收到此前3次重复训练结果及冻结主干指纹结果。
- 实际差异：原/当前Python为3.10.20(GCC14.3.0)/3.10.22(GCC14.4.0)；torch包标签2.1.2/2.1.2+cu118；torchvision0.16.2/0.16.2+cu118；PyTorch构建输出中的MKL为2022.1 Build20220311/2022.2 Build20220804。原环境同样CUDA可用构建，不能把无+cu118后缀理解为CPU版；仅靠标签也不足以确定全部二进制差异。
- 一致项：CUDA build11.8、cuDNN8700(8.7.0)、本次检查进程matmul TF32=False/cuDNN TF32=True；timm1.0.15、numpy1.26.4、albumentations1.4.24、albucore0.0.23、safetensors0.8.0、torchmetrics1.6.1、OpenCV-headless4.10.0.84、Pillow12.3.0、scikit-image0.25.2、scipy1.15.3；torch编译器GCC9.3、MKL-DNN3.1.1及其Git hash一致。USE_NCCL文本为ON/1，本训练为单卡且无分布式通信，不将该文本差异归为精度原因。
- 结论边界：环境存在差异，但没有证据将0.003960峰值差归因于Python补丁、MKL或发行包构建，也未发现CUDA/cuDNN/TF32默认值及列出增强库版本差异。此前输入探针已证实同seed样本相同而增强不同，训练波动仍是合理候选解释；不把两条曲线当作环境因果对照，不为接近0.8786而先改LR/epoch或重装依赖。
- 下一步（尚未执行）：同一原始clone(81b19e2)、同一空闲3090串行、官方Hiera初始化、seed1234及完整20轮原参数，仅切换两套绝对Python路径；若此前重复尚未开始，原/当前/原/当前各2次交替作为探索；若当前环境3次已启动/完成则全部保留，只补2次原环境，不重复开启当前组。所有新checkpoint统一用mslau-naf解释器和原分支kvasir_test.py评估，训练环境/参数/日志分别保存。小样本只提供探索信号，不能分离Python与MKL贡献或宣称显著性；不擅自停止已经启动的任务。
- 产物：`analysis/p51_original_08786_vs_repro_08746_20261016/environment_comparison.csv`、`environment_comparison.json`，含原日志SHA256、解析后的包/构建数据和对照协议；启动命令已做bash语法检查。本轮未改模型、未在本地或服务器启动训练。

### 2026-10-19：原/当前环境各2次对照完成，存在探索信号但不足以归因

- 具体操作与完整性：读取用户同步`p51_env_ab_zTFKdPrU`全部4次记录；原/当前/原/当前顺序串行、时间戳无重叠。代码81b19e2，保存的Hiera及split哈希与先前一致。训练命令除解释器和输出位置外完全一致，全部20轮、4组模型配置头和40值/次的学习率序列一致。两次原环境pip freeze逐字相同，两次当前环境亦相同。4次均用`/home/le/miniconda3/envs/mslau-naf/bin/python`与原分支`kvasir_test.py`评估，同120张验证集、参数除checkpoint路径外相同；独立评估与训练最佳值在4位小数精度下一致。
- 同步边界：第4次`console.log`未同步，但其`logs/kvasir_train.log`覆盖0—19轮、有训练完成及最佳checkpoint汇总，独立评估亦完成，因此纳入分析，无需重训。尚未收到冻结主干指纹结果。
- 单次结果：1_original **0.875919@14**，独立IoU0.8759/Dice0.9264/Precision0.9218/Recall0.9472，最后5轮IoU0.86976、20m57s；2_current **0.873020@18**，0.8730/0.9274/0.9269/0.9392，最后5轮0.86806、21m04s；3_original **0.871033@13**，0.8710/0.9261/0.9339/0.9314，最后5轮0.86702、21m01s；4_current **0.869290@13**，0.8693/0.9232/0.9207/0.9413，最后5轮0.86426、21m15s。
- 汇总：原环境 **0.873476±0.003455**，当前 **0.871155±0.002638**（各n=2、全部seed1234、样本标准差；不是跨seed结果）。均值差 **+0.002321**（+0.2321个百分点）。原环境范围0.871033—0.875919，当前0.869290—0.873020，范围重叠；原环境自身两次相差0.004886（0.4886个百分点），大于历史0.878616与先前0.874656的0.003960差距量级。相邻区组差+0.002899/+0.001743仅作描述，增强输入并未严格配对，不作配对显著性推断。最后5轮均值分别0.86839/0.86616。最好0.875919仍低于历史峰值0.002697（0.2697个百分点）。
- 新运行条件证据：本轮4次GPU前置快照驱动均为 **580.178.04**，而2026-10-14的0.874656复现快照为 **535.154.05**；8月历史0.878616的驱动未知。不能把先前复现混入本轮同条件环境组统计，或把驱动变化当精度原因。SMI显示CUDA13.0是驱动支持信息，各运行torch实际CUDA build仍为11.8、cuDNN8.7。四次GPU前置快照无计算进程，运行时均记录RTX3090。两环境本轮耗时均约21分钟，先前28分钟差异未重现。
- 环境边界：完整pip freeze还显示ImageIO2.37.4/2.38.0、Pydantic2.13.4/2.14.1、stringzilla4.6.2/5.2.0等差异；此前仅对列出的核心包确认同版，并未声称全部依赖相同。本实验检验环境整体，不分解Python/MKL或单库贡献；URL安装记录与==版本记录不同不自动表示版本不同。
- 结论：原环境两次探索性均值较高，但n=2且组内波动较大，支持继续小规模确认，不能认定环境是历史差距主因或断言环境无影响。训练重复波动已被直接观察到，不保证更换原环境就恢复0.878616。不回退驱动、改超参数或扩展NAF。
- 下一步（尚未执行）：反向顺序先current再original各补1次，同原代码、参数、同空闲3090和580.178.04驱动，达到每环境3次同seed重复；输出新`p51_env_ab_extra_...`目录，统一当前解释器评估，检查驱动/pip freeze延续上一轮。若驱动或依赖改变，保存输出再判断，不静默混组；不为未同步的第4次console补训。
- 产物：`analysis/p51_env_ab_zTFKdPrU_20261019/run_summary.csv`、`epoch_metrics.csv`、`environment_freeze_differences.csv`、`analysis_summary.json`，包含原文件SHA256、命令/日志审计、分组统计及驱动变化证据。本轮只分析/记录，无模型代码改动，无训练启动。

### 2026-10-19：环境补跑完成及后续输出布局

- 证据：`p51_env_ab_extra_E10YU4dN`，current→original各补1次。与前4次同代码81b19e2、Hiera/split哈希相同，两环境各自pip freeze逐字延续前轮；6次有效训练参数、模型配置头、学习率序列一致，全部20轮。统一当前解释器/原分支评估，评估IoU与训练最佳四位小数一致。
- 新结果：current **0.865311@19**（Dice0.9183，22m02s）；original **0.866661@16**（Dice0.9198，22m11s）。各3次均seed1234：original **0.871204±0.004631**，current **0.869207±0.003855**，样本标准差；均值差+0.001997（0.1997个百分点）。原环境范围0.866661—0.875919，当前0.865311—0.873020。这不是三seed结果。
- 运行条件边界：两次驱动仍580.178.04、RTX3090、CUDA build11.8、cuDNN8700，runtime均CUDA_VISIBLE_DEVICES=0。current补跑前整机无计算进程；original补跑前GPU1—3有rife/ffmpeg进程，选用GPU0显存0MiB但利用率47%，不能称整机空闲或证明训练全程无竞争。original记录GPU0 UUID，current的selected_gpu.txt未同步。不能将其他卡任务直接归因于精度下降。
- 结论：原环境保持小幅探索性优势，但范围明显重叠，组内波动大于均值差；两环境第三次均下降，未定位唯一原因。尚未复现历史0.878616，也没有证明环境等效或库版本为差距主因。不继续无上限重复以追峰值，不自动启动训练或更换驱动/依赖。
- 输出约定：用户指定后续权重/checkpoint、实验记录、可视化分别归`/data/models/le/mslau-net/checkpoints`、`experiments`、`visualization`，使用共同run_id分目录，详见第3节。本轮只记录约定，未改训练保存代码、删除/搬迁文件或创建远程目录。
- 产物：`analysis/p51_env_ab_six_runs_20261019/`中的run_summary.csv、epoch_metrics.csv、environment_freeze_differences.csv、analysis_summary.json；保留原始日志SHA256与验证项。

### 2026-10-19：输出入口统一、保留清单与NAF优先级更新

- 用户授权：统一后续输出，盘点旧文件但先不删除；第一目标改为NAF稳定超过同环境匹配baseline，历史最优精确复现为第二目标。停止将追到0.878616作为模块工作前提，不继续无限重跑/重装环境。
- 代码：新增tools/run_experiment.py（训练/评估包装、唯一run_id、三根目录、元数据/源码快照、GPU启动检查、日志与失败状态、带锁index.jsonl）；新增tools/inventory_outputs.py（只读扫描，CSV/JSON报告，跳过代码/数据/环境目录及软链接，不删除/迁移）；kvasir_train.py仅增加日志目录parsed_args.json，记录完整解析参数，无RNG/模型/损失/训练参数变更。保留原始checkpoint保存语义，尚未去重或添加续训状态。
- 清单：本地analysis/output_retention_20261019/retention_plan.csv覆盖19目录/文件组；local_inventory_v2逐文件209项，约1.41MB逻辑大小。唯一观察到的pth为0字节占位，不代表有效权重；服务器完整权重容量待只读盘点。不因本地未同步而判定远端缺失或估计可释放容量。历史0.878616、原0.874656源权重、关键匹配对照与诊断引用权重保护；重复命名副本、冒烟和旧PID仅列人工审阅候选。
- 研究协议：固定mslau-naf环境；新选择器先无正则S，再加分段正则SR，对照none基线B；如有信号补均匀正则SU辨别分段贡献。stage/位置和其他超参固定，不新增融合。先明确设计并做工程检查，再筛查seed1234、锁定候选后重跑42/1234/2026匹配对照，额外独立确认后才讨论稳定性。旧驱动/旧代码结果不替代新B。不保证方法必然有效。
- 验证：新增tests/test_experiment_outputs.py共7项通过，覆盖无副作用dry-run、禁止输出路径覆盖、评估路径与父run链接、GPU忙时不启动、子进程成功/失败记录、只读扫描；本地使用现有test-py311和/tmp/s2i_vitl_1015_deps中的pytest，无新安装。py_compile与git diff --check均通过。未启动训练、新NAF尚未实现。
- 交付说明：docs/experiment_outputs.md包含新布局、可执行dry-run、服务器只读盘点命令和保留规则。清单不等于清理授权；本轮未删除/搬迁文件，未推送或在服务器执行新入口。

### 2026-10-19：服务器实际盘点及新NAF软选择/分段正则提案（未实现、未训练）

- 用户范围：分析同步盘点报告，先给新设计和最小消融，不启动训练。本轮不修改模型/训练代码，也不删除或搬迁输出。
- 证据：experiments/output_inventory_20261019_032337_217355，创建时间03:23 UTC=11:23北京时间；251文件、60份非零权重，扫描无错误。20次训练每次3权重，命名best IoU/epoch与本地各run训练日志20/20一致；尚未加载tensor或验证副本内容。
- 容量：权重51968512116字节=48.39945GiB；其他191文件1262340字节=1.20386MiB。保留19份非冒烟固定名称最佳模型约15.326GiB；3份冒烟候选2.420GiB；19份命名best_iou副本待核对15.327GiB；19份best_loss按用途/引用复核15.327GiB（其中6次与best IoU同epoch，不等于已验证相同）。此前本地0字节best_loss占位在服务器报告为正常大小。新三根目录未记录到业务输出，本次盘点目录自身被排除；不解释为目录不存在。zsy历史/预训练及数据/源码目录不在盘点范围。
- 产物：父目录analysis/output_inventory_20261019_review/summary.json、run_checkpoint_summary.csv、weight_retention_review.csv（60权重逐项建议、源报告SHA256）。逻辑字节数不是已核实可回收物理容量；本轮无清理。
- 原文复核：提供PDF哈希f67d4904...9dd58f3与既有记录一致，本轮重新提取并目视检查第4/5页Eq.(6)—(9)。原文为频率softmax、90%概率硬抽样、V的分段L2；本提案为确定性可微预算选择、对丢弃量的分段惩罚，不是原算法等价复现。
- 新选择器提案：RFB后四级独立，保持原融合/decoder；64+3→16→64的逐频MLP，最终层无bias并零初始化，共8448参数（待实现核对）。每样本/通道在非DC频点内做频率softmax，半谱按共轭重数计数；用g=sigmoid((z-theta)/tau)、tau=1，求theta使sum(m*g)/sum(m)=rho=0.9。频率相互竞争仅发生在本级，不跨stage；softmax的共同平移会被theta吸收，真正选择约束来自预算，不将softmax单独包装为不可替代创新。
- 重建：DC增益固定1，其他M=g/rho，IFFT(M*Q)。均匀z产生g=rho、M=1，数值近恒等；不额外引入可关掉模块的alpha。增益0—1/rho，保留均值不等于保留90%能量或硬保留90%频点。初始化前层首步梯度为0、末层可学习，后续应验证梯度贯通；预算仍允许均匀选择的恒等退化。
- 可导性：theta必须隐式求导或用经梯度核查的可微求解，不可detach阈值后冒充固定预算梯度。标量代数检查预算0.9、恒等误差2.22e-16、隐式导数与有限差分误差1.15e-11；这不是完整二维模块/AMP/真实网络工程验收。
- 新分段正则提案：沿原径向1/3、2/3分三段，不预设stage高低频偏好。对丢弃量d=1-g施加平方惩罚，低/中/高w=(4,2,1)，按实际频点重数归一化使mean(w)=1；R=mean(m*w*d^2)/(1-rho)^2，再对样本/通道/stage平均。均匀选择时R=1；较强低频惩罚意味着保护低频免于丢弃，符号与直接惩罚保留分数V不同。此权重为待验证图像先验，不是原文系数复现，也不保证高频是噪声。
- lambda提案：实现后只用train.txt前48张、4个固定batch12、确定预处理、相同未训练模型eval模式校准，无optimizer step/BN更新；取末层选择器分割梯度与分段正则梯度范数中位数Gseg/Greg，lambda=min(0.01,0.05*Gseg/Greg)。退化/非有限梯度先修工程，不自动开训或用val搜索。lambda固定跨seed，均匀正则对照沿用相同lambda（其恒等初始梯度可能为0，不单独校准）。rho、温度、band权重及5%梯度目标均为设计选择，非最优实证。
- 最小消融：B=none；S=新选择lambda0；SR=新选择+分段；若SR有信号补SU=同选择+均匀正则，完整机制对照4组。首次固定四stage、官方Hiera初始化、352/b12/nw8/e20/Structure/双LR1e-3/D2D3=1/boundary0，seed1234仅初筛；锁定S或SR后补42/2026，并依据重复波动补独立确认。SR>S且SR>SU才初步支持分段分配；只优于B不能归因于分段。跨环境/跨数据收益尚未证明。
- 新匹配B与候选须同时使用相同的前瞻复现控制：隔离新模块初始化的RNG消耗、核对共同初始参数/缓冲、DataLoader使用独立generator且增强/worker显式设seed；不追溯历史每batch增强、不复用旧驱动/旧控制的基线作严格对照。预检包括奇偶尺寸共轭、恒等/预算/有限差分、末层与前层梯度、AMP及能量/频段诊断，以上均待后续实现。
- 下一步：用户先审阅设计；当前无需服务器执行新命令，不再补跑历史峰值。后续仅在落实新设计与工程验收后再给训练命令，本轮未生成启动训练任务。

### 2026-10-19：用户收敛实验路线，原环境副本克隆成功

- 用户新要求优先于先前兼容性/永久保留约定：可以删除重建NAF功能分支；每次实验复制独立必要源码，修改/新增文件明确标记，不再构建兼容全部候选的通用模型。以原始81b19e2及最接近历史值0.875919的原作者当前环境为新主线；先实验0三seed固定基线，再加入单个NAF候选并迭代。历史0.878616不再作为推进障碍，未达到不能证明只由偶然性导致。
- 用户澄清不强行凑9次实验；现有匹配基线/四级NAF/P3P4 NAF三组各三seed可作历史批次，新主线不得跨环境混成匹配对照。旧两版NAF只是相同算法的启用位置变化，新预算选择设计尚未实现。
- 清理授权已变化：保留有用的匹配实验结果，其他旁支/冗余可清理，迁移至用户三类输出根目录；不再默认保留所有失败产物。计划与实际操作分开，本轮尚未执行项目输出删除、迁移、代码重整或分支删除。原始数据/官方预训练/其他账号成果不在清理范围。
- 环境：原环境使用方式是用户在3090上直接调用/home/zsy/miniconda3/envs/mslau-net/bin/python，并非本地曾复刻。离线克隆因缺缓存失败；添加原cache仍失败；联网克隆一度被defaults ToS拦截。最终用户显式override channels为conda-forge/nvidia/pytorch、使用两份cache、--copy，成功克隆到/home/le/miniconda3/envs/mslau-p51。71个Conda包、13404个文件，pip check通过，torch/torchvision可导入。
- 回传运行版本：torch2.1.2、torchvision0.16.2+cu121、torch CUDA build11.8、cuDNN8700。原环境已存pip元数据torchvision0.16.2不等于已核实其运行时version.py标签；需一次原/副本运行时与pip freeze对照，不先重装torchvision或把克隆成功称完全一致。clone计划包含原来源的MKL2022.1及CUDA11.8构建；cuda-version12.9元包不替代torch实际CUDA build。
- 失败环境策略：用户明确失败就直接删，不再改名保留。此前失败副本用户已清理；最新版命令在独立bash内检查目标不存在后设置清理trap，克隆/基础检查失败或中断仅清理本次目标。成功mslau-p51保留，不删原环境/mslau-naf。粘贴多行曾丢换行，后续服务器命令优先独立单行或脚本文件，避免长多行依赖换行完整性。
- 下一步：只读核对原环境与新副本，之后固定副本并落实独立实验0代码/结果精简方案。不继续扩大环境因果审计，不启动训练；待具体运行脚本准备好再给用户执行。

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

2026-10-19 当前优先以用户最新收敛要求为准：回到原始81b19e2，从原作者当前环境复制的mslau-p51已在3090克隆成功（一次运行时/包清单对照待完成）。重整成独立实验目录，实验0运行原命令三seed各一次固定baseline，之后每个NAF候选独立复制修改；不再维持全部候选兼容、不追历史峰值、不强行凑9次实验。旧匹配实验保留必要主权重与记录，冗余旁支可清理，迁移至用户三个根目录；实际项目清理/重构尚未执行，失败环境不留备份。NAF稳定提升为第一目标。先前软预算/分段正则只是未实现提案，待新baseline及独立代码组织落实后选一个候选。下列P5.2/CVC等为历史储备，均排在当前主线之后，不自动启动。

1. **P5.2小目标主线**：以冻结的P5.1为基线，先分别做“aux D2/D3权重1.0/0.5/0.25”和“H/2零初始化RGB细节残差分支”两个seed1234单变量实验；有效后才组合并补三seed。
2. **预注册判断标准**：最小面积四分位至少从0.800657提高到0.810；总体三seed均值不低于0.872802，Precision不低于0.919、Recall不低于0.935，最大面积组下降不超过0.005。
3. **保持单变量**：第一版细节分支不加入reverse、wavelet、手工边缘、Dropout或新Loss；如轮廓仍差，再单独增加训练期boundary head，权重0.05且不参与乘性门控。
4. **停止旧失败路线**：不再补P3-352、不继续P4 cascade/no-final-reverse，也不继续固定0.1 DINOv2蒸馏或简单Lovasz叠加；P3/P0保留为历史基线。
5. **更强证据**：最终方案必须补seed42/2026或建立独立test；不在同一val上反复调阈值。
6. **复现 CVC 历史0.90**：官方CSV/fold0、clean P0、batch8、纯Dice、最高IoU保存；只改变旧训练参数。
7. **严格 CVC 提升**：新增完整模型`--init_checkpoint`，从Kvasir 0.860583 P0初始化；先冻结encoder，再以encoder 1e-5、decoder 1e-4微调；采用sequence-level五折。
8. **LFF**：停止只调LR/Dropout；如重启，先完成特征幅值统计。
9. 每次实验完成后更新本文件，并将重要代码状态提交到清晰命名的Git分支；不要让checkpoint与代码语义错配。
