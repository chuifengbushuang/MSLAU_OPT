# P5.1 严格v1基线代码改动与复现说明

严格v1沿用原P5.1网络、损失与训练配方，改变了数据输入随机序列的生成方式和CUDA确定性执行方式，并增加状态记录。
该版本在seed2026取得最佳验证IoU **0.8820540308952332**，对应epoch18，即20轮训练的第19轮。
复现这个历史结果需要使用`paired_aug_v1`的原实现，包括后来确认的增强开关联动行为；使用修正后的v2属于另一套输入协议。

原作者历史最高0.878616来自seed1234，严格v1的0.882054来自seed2026，差0.3438个百分点。
两者同时存在seed、输入随机机制、严格计算及历史环境状态的差异，不能据此将涨幅归因于单独一项代码改动。

## 对照版本和已观察结果

原代码基准是`kvasir-p5-sam2unet-official-adapter`分支提交`81b19e2c801efcf74826b8127f5ea712b0247ec5`。
严格v1实验定义为`exp00_baseline_paired_strict_v1`，产生目标结果时研究仓库提交为`f21c0276b3c4e43eee7a323a81a7b9e11f313c5d`。

| seed | 原作者原P5.1历史IoU | 严格v1最佳验证IoU | 严格v1 Dice | 严格v1 Precision | 严格v1 Recall |
|---|---:|---:|---:|---:|---:|
|42|0.868977|0.874576983|0.9285|0.9103|0.9582|
|1234|0.878616|0.853762156|0.9090|0.9065|0.9352|
|2026|0.870813|**0.882054031**|**0.9333**|**0.9321**|**0.9427**|
|三seed均值|0.872802|0.870131057|0.9236|0.9163|0.9454|

严格v1种子间IoU样本标准差为0.014661，原作者历史记录为0.005118。
严格v1的单次最高值更高，三seed均值并未超过原作者历史均值；该表不是单变量因果对照。
seed1234已完成两次独立20轮训练，输入、训练轨迹和最终选定权重内容一致；**seed2026的0.882054目前只有一次完整训练记录，尚待独立重复确认**。

目标运行目录为`exp00_baseline_paired_strict_v1_seed2026_20261027_103924_447984_r1`。
目标模型是该目录的`kvasir_best_model.pth`，或内容相同的`best_iou_0.882054_epoch_18_0.302606.pth`。
它不是最佳Loss权重；最佳Loss出现在epoch19，文件名为`best_loss_0.299703_epoch_19_0.880644.pth`。

## 实际修改了哪些文件

相对于上述原始提交，在原有14个训练与模型相关文件中，只修改两份原文件；网络、loader、loss及split逐字节不变。

| 文件 | 变化 | 是否影响训练轨迹 |
|---|---|---|
|`kvasir_train.py`|改用PairedDataset与独立采样器；每阶段设置epoch；调用严格运行时；添加输入、初值和阶段状态记录|输入算法和严格运行时会影响；记录用于核对|
|`kvasir_test.py`|评估前调用同一严格运行时并打印配置|可能影响计算路径；指标公式、预处理、阈值和模型参数不变|
|新增`paired_inputs.py`|按样本及epoch派生增强种子；独立shuffle和worker随机流；输入与初值哈希|前三项影响实际输入序列|
|新增`strict_runtime.py`|启用严格确定性；记录优化器、参数、buffer及RNG状态|严格算法影响计算路径；哈希只读|
|运行包装`run.py`及`config.json`|提供固定参数、进程启动前环境变量、输出布局与日志检查|环境变量影响运行；路径和记录属于组织方式|

`run.py`和`config.json`是研究仓库的新增包装文件，原作者的原始提交没有这两个文件。
仓库中的`changes.patch`同时包含对这个包装文件的修改，不能直接整份应用到缺少`run.py`的原作者目录。
转交包中的`minimal_overlay/original_to_strict_v1.patch`只修改原始`kvasir_train.py`、`kvasir_test.py`，配合两份新增helper可用于原代码副本。

## 按样本固定增强随机性

原代码对Python、NumPy及PyTorch设置全局seed，但没有完整控制Albumentations 1.4.24增强对象内部的随机状态。
所以相同seed可能得到相同样本顺序，却得到不同的增强图像和mask。

v1的`PairedDataset`继承原`loader.binary_class`，保留原图像读取、`mask > 0`转换及Compose增强调用。
在每次读取样本前执行：

```python
self.transforms.set_random_seed(stable_seed(
    self.seed, 'augmentation', self.phase, self.epoch, sample_id))
```

`stable_seed`不使用Python内置hash。它将以下列表按固定JSON格式编码为UTF-8，计算SHA256，取前8字节的小端整数并限制为63位：

```text
["paired_aug_v1", seed, "augmentation", phase, epoch, sample_id]
```

因此，同seed、同epoch、同phase、同样本ID的增强可重复；不同epoch改变增强。
repeat标签、输出路径和worker编号不参与增强种子。
这会产生一套新的确定输入序列，不保证复现原作者历史运行时未记录的增强序列。

原增强工厂没有改，顺序仍然是：

```python
A.Resize(352, 352)
A.HorizontalFlip(p=0.25)
A.VerticalFlip(p=0.25)
A.ShiftScaleRotate(shift_limit=0, p=0.25)
A.CoarseDropout()
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
ToTensorV2()
```

CoarseDropout等未显式给出的参数继续使用当时Albumentations 1.4.24的默认值，不能用其他版本的默认值代替。

## v1已确认的增强联动行为

Albumentations 1.4.24的`Compose.set_random_seed`会将相同种子传给各子变换。
v1每个样本都重置一次，使各子变换从相同随机数起点开始。三个`p=0.25`几何增强的触发开关因而联动，
并与默认`p=0.5`的CoarseDropout发生相关。这是不同增强算子之间的联动，图像和mask的几何对齐仍保持一致。

4096次真实库合成抽样的既有检查显示：

| 检查项 | 原增强未逐样本重置 | v1逐样本同种子重置 |
|---|---:|---:|
|水平和垂直翻转同时触发|6.714%|24.805%|
|三个几何增强同时触发|1.611%|24.805%|
|四个增强开关出现的组合数|16|3|

这些比例是合成检查观察值，不是目标训练日志逐样本统计。
原本以为只固定了随机性，实际上也改变了增强的联合分布；这是v1实现中已经确认的缺陷。
它可能影响训练效果，但尚不能认定它造成了0.882054。

复现历史v1时应使用已有v1 helper，保留上述行为并注明协议。
v2将Compose及其子变换改为共享持续推进的随机流，修复了这项联动；不得把v2 helper替换进去后仍称为v1原结果复现。

## 独立控制样本顺序与worker随机数

原训练使用`DataLoader(shuffle=True)`，样本排列依赖PyTorch随机流。
v1使用`EpochRandomSampler`，为每个epoch创建独立`torch.Generator`并调用`torch.randperm`：

```text
排列种子 = stable_seed(seed, "sample_order", epoch)
worker种子 = stable_seed(seed, "worker_base", phase, epoch)
```

worker初始化仍为Python和NumPy设置`torch.initial_seed() % 2**32`，但样本增强不再由worker编号决定。
DataLoader使用独立generator，`persistent_workers=False`；每个阶段开始前更新dataset.epoch和worker generator。
这些变化使输入顺序及增强不依赖模型初始化消耗了多少全局随机数，但也改变了原代码的具体采样序列。

训练仍然batch12、workers8、drop_last=True；880张中每轮使用876张，共73个batch。
训练中的验证仍为batch6、workers8、不丢尾批，共120张、20个batch。
独立评估仍为batch8、workers0。

## 严格CUDA执行的新增控制

原训练代码已经包含以下设置，不能将它们说成v1首次加入：

```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

v1关键新增项是：

```python
torch.use_deterministic_algorithms(True, warn_only=False)
```

并且在Python子进程启动前设置：

```bash
CUDA_VISIBLE_DEVICES=3
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=2026
NO_ALBUMENTATIONS_UPDATE=1
```

原脚本虽然在进程内部写过`os.environ["PYTHONHASHSEED"]`，这不能重设当前已启动解释器的hash种子；v1启动器在进程启动前传入。
`NO_ALBUMENTATIONS_UPDATE`仅关闭更新检查，不能解释为性能改进。
`configure_runtime()`同时用于训练和独立评估，严格算法遇到不支持的操作时直接报错，不降级为warn_only。

未修改`F.interpolate`的mode或align_corners参数。服务器torch2.1.2的严格执行会对相关双线性上采样反向采用确定性替代路径。
已有3090检查中，普通模式相同输入的5次梯度出现5种哈希，严格模式只出现1种；误差量级约1e-6。
因此开启严格模式会改变数值计算路径，不能将它视为纯日志开关，也不能认定这一个算子解释全部最终IoU差距。

目标运行保持matmul TF32=False、cuDNN TF32=True；flash/memory-efficient/math SDPA三个后端均允许。
没有为了确定性强制关闭全部TF32或强制切换注意力实现。

## 状态记录和输出的变化

训练前记录完整模型初始参数与CPU RNG；每个epoch的train/valid结束后记录参数、buffer、AdamW状态、学习率、scheduler及CPU/CUDA RNG。
每个阶段还记录样本顺序和归一化图像、mask的流式哈希，以及未舍入的指标。
这些函数读取状态而不更新参数、不消费随机数；CPU/GPU同步及哈希计算存在耗时，因此记录前后的运行时间不宜直接作为性能基准。

新增记录包括`initial_state.json`、`input_fingerprints.jsonl`、`epoch_metrics.jsonl`、`numerical_runtime.json`、`numerical_states.jsonl`和`checkpoint_tensors.json`。
20轮应有40条阶段记录。启动器只在输入/严格状态记录完整时标记completed。

仍保留原脚本的三份权重命名：`kvasir_best_model.pth`、`best_iou_...pth`、`best_loss_...pth`。
repeat只改变run_id标签，不改变seed或任何输入算法。

## 保持原样的网络和训练配方

| 项目 | 严格v1目标运行 |
|---|---|
|网络|SAM2 Hiera-L＋block Adapter32＋四层RFB64＋原U形Decoder|
|参数|总计216419043，可训练4269747；没有NAF或新融合模块|
|初始化|官方Hiera权重；Adapter和Decoder按seed初始化，不从已训练P5.1权重继续|
|图像与数据|352×352；原880/120划分；原loader及mask>0口径|
|损失|Structure主输出＋1.0×D2＋1.0×D3；boundary=0|
|优化器|AdamW，两组LR均1e-3，weight_decay=5e-4|
|调度与轮次|CosineAnnealingLR，T_max=20、eta_min=0，20轮编号0–19|
|训练|batch12，workers8，单卡，无AMP|
|其他|不新增warmup、EMA、梯度裁剪或模型初始化算法|
|checkpoint选择|以最高验证IoU为主要结果；保留最低验证Loss权重|

## 目标运行环境与核对指纹

目标运行在物理GPU3的RTX3090上，驱动580.178.04。运行环境由原作者环境克隆而来：

| 项目 | 目标运行记录 |
|---|---|
|Python|3.10.20|
|PyTorch|2.1.2|
|torchvision运行时|0.16.2+cu121|
|torch CUDA build|11.8|
|cuDNN|8700|
|timm|1.0.15|
|Albumentations / albucore|1.4.24 / 0.0.23|
|NumPy|1.26.4|
|OpenCV headless|4.10.0.84|
|scikit-image|0.25.2|

torchvision运行时后缀与torch CUDA build是目标环境的实测组合，复现时以原环境为准，不根据后缀自行重装。
转交包`reference/environment.txt`保留完整pip freeze，`runtime.txt`和`gpu_before.txt`保留当时运行信息。
pip freeze含本地构建路径，主要用于比对，不应直接当作可移植安装清单。

官方Hiera文件为`/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors`。
其SHA256应为`7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c`。

seed2026关键指纹：

```text
完整初始state_dict内容
a4ca9b0abb6f3a0e5f51e4acb314cd7b612ecd9df01652afb40b8b2d383fe319

epoch0 train样本顺序
0997cf8cf638a2f8527e9a86548a3bc6fe0ffcf25e9ea2718b1f5f304123ec86

epoch0 train增强图像
3597f5cf67225a04dc98afc23eca3a997344cccf5bf84ca94b1299dcbd4b7b85

epoch0 train增强mask
ca5da660e44ff2d6affcc8f934fdedf88f3172e92398221186a12e3cdde1308e

完整40阶段输入文件
25d75902473f27be9bcd9ef21faeae5dccd55f82eb5da775238dedbe5e063159

最佳IoU模型的state_dict内容
fe7a85813380ff80fc5a5876874b77329eb79c9f28e0c4d4ff9a55ada1b455ba
```

若模型初值不同，先检查代码、seed、timm/PyTorch和官方权重；若初值相同但输入不同，检查数据、split、Albumentations/helper和采样设置；
若输入与初值都相同而轨迹不同，再检查严格配置及计算环境。
不能只看到最终IoU相近就认定完整复现，也不承诺换环境后逐位一致。

## 原实验者执行方法

可以在研究分支仓库中直接使用原作者Python执行。`tools/run_experiment.py`及展开的`run.py`沿用当前`sys.executable`，
不会强制调用le的环境。输出根目录可指定为原作者自己的新目录。

先预览，不训练：

```bash
P51_OUTPUT_ROOT=/data/models/zsy/mslau-net/strict_v1_reproduction \
P51_REPEAT=2 \
/home/zsy/miniconda3/envs/mslau-net/bin/python -u tools/run_experiment.py \
    exp00_baseline_paired_strict_v1 --seed 2026 --gpu 3
```

确认预览后，由原实验者执行目标seed的一次独立重复：

```bash
P51_OUTPUT_ROOT=/data/models/zsy/mslau-net/strict_v1_reproduction \
P51_REPEAT=2 \
/home/zsy/miniconda3/envs/mslau-net/bin/python -u tools/run_experiment.py \
    exp00_baseline_paired_strict_v1 --seed 2026 --gpu 3 --run
```

这两条命令应在`exp/p51-independent`分支仓库根目录执行，工作树保持已提交状态。
使用的是原始输入v1的历史实验入口，不会调用当前修复后的v2基线或NAF模型。
GPU3沿用目标记录；更换卡后仍需按指纹判断实际重复性。

若使用独立转交包，解压到新的目录，在包根目录执行下面的预览；正式运行时末尾增加`--run`即可：

```bash
P51_OUTPUT_ROOT=/data/models/zsy/mslau-net/strict_v1_reproduction \
P51_REPEAT=2 \
/home/zsy/miniconda3/envs/mslau-net/bin/python -u source/run.py --seed 2026 --gpu 3
```

包中的完整源码可独立运行，不依赖其他实验目录。三类输出分别在输出根目录的checkpoints、experiments、visualization下，
没有图像导出时不会生成图像目录。实际命令由启动器打印并写入command.txt，全部训练参数与本说明的固定配方一致。

如果只采用最小补丁，请先复制一份准确的原始提交代码到新目录，再执行`git apply --check`和`git apply`，
将overlay内两份helper放在kvasir_train.py旁。直接运行修改后的训练脚本时必须在Python启动前传入：

```bash
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2026 \
NO_ALBUMENTATIONS_UPDATE=1 \
/home/zsy/miniconda3/envs/mslau-net/bin/python -u kvasir_train.py \
  --dataset /data/dataset/zsy/mslau-net/Kvasir-SEG \
  --train_txt configs/splits/kvasir/train.txt \
  --val_txt configs/splits/kvasir/val.txt \
  --output_dir /data/models/zsy/mslau-net/strict_v1_reproduction/checkpoints/direct_seed2026_r2 \
  --log_dir /data/models/zsy/mslau-net/strict_v1_reproduction/experiments/direct_seed2026_r2/logs \
  --decoder_mode p5_sam2unet \
  --encoder_name sam2_hiera_large \
  --hiera_checkpoint /data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors \
  --loss structure --img_size 352 --batch 12 --num_workers 8 --epoch 20 --seed 2026 \
  --encoder_lr 1e-3 --decoder_lr 1e-3 \
  --aux_d2_weight 1.0 --aux_d3_weight 1.0 --boundary_weight 0.0
```

直跑示例的输出目录须为新的空目录；若需要再次运行，请换新的目录名。
直接训练入口没有包装器的来源检查、环境归档和独立评估，优先采用完整源码中的run.py。
原0.878616代码目录和权重保留为历史参考。

## 实验3的P1和P2含义

实验3仅对原RFB输出的两路浅层、高分辨率特征做NAF：

| 层级 | 352输入时尺寸 | RFB后通道数 | 实验3处理 |
|---|---|---:|---|
|P1|88×88|64|NAF1|
|P2|44×44|64|NAF2|
|P3|22×22|64|直接进入原解码器|
|P4|11×11|64|直接进入原解码器|

浅层是相对于四级编码金字塔而言，不是输入图像或网络最前两层卷积。
独立NAF之间没有新增跨层融合，原Decoder仍从P4开始向上与P3、NAF2(P2)、NAF1(P1)融合。
严格v1的0.882054基线没有任何NAF，不能将实验3的模块改动混入这次复现。

## 复现后的记录交接

请提供新运行的manifest.json及logs中的初值、输入、指标、数值状态和checkpoint_tensors记录。
转交包附有原0.882054运行的同名记录，可用包内compare_reference.py按阶段比较。
其输出区分初值、输入、严格配置、训练轨迹和权重内容，便于定位差异；不要仅比较torch.save文件的档案SHA，
同内容权重可能因序列化文件名等因素具有不同文件SHA。

例如，在包根目录执行以下命令，将`<新run_id>`替换为训练结束时打印的实际记录目录名：

```bash
/home/zsy/miniconda3/envs/mslau-net/bin/python compare_reference.py \
    --run-dir "/data/models/zsy/mslau-net/strict_v1_reproduction/experiments/<新run_id>"
```

`matched`表示所检查的六份状态/指标记录一致；`different`会给出首个不同阶段和字段。
对照器只读，既不加载权重，也不训练。转交包不包含大权重或图像数据。

如目标seed成功重复，再讨论增强联动与严格计算各自的作用；当前不根据单次峰值认定严格v1是更优的通用基线。
