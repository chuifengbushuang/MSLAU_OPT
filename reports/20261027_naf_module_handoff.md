# 两版二维NAF模块的代码交接说明

供原实验者在自己的代码分支接入两版NAF。最简方式是从同一仓库的`exp/p51-independent`分支取出所需模块文件，按下文接入现有网络和训练损失。无需合并整个研究分支、采用研究分支的实验启动器或复制完整项目。

本文只交接模块、接口和相对原始基线的改动，不规定接收方的实验安排。两版是不同候选，分别接入；本文没有定义同时串联两版NAF的模型。

## 1 交接文件与版本

相对原始基线的参考提交是`81b19e2c801efcf74826b8127f5ea712b0247ec5`，即`kvasir-p5-sam2unet-official-adapter`分支对应的P5.1代码。以下模块以已推送提交`51947a6abd365172c1b229baaa464d0ff0cdc5a9`中的内容为准；交接文档和参考补丁在其后提交。

| 版本 | 模块源文件 | 对外类 | 模块新增参数，64通道 |
|---|---|---|---:|
| 实验3对应的独立NAF | `experiments/exp03_naf_p12_paired_strict_v2/added/networks/naf_select_reg.py` | `PyramidNAF2D` | 164，仅P1/P2 |
| 实验9对应的级联NAF | `experiments/exp09_naf_discrete_paired_strict_v2/added/networks/naf_discrete.py` | `CascadeDiscreteNAF2D` | 56,118 |

文件放在名称含`strict_v2`的目录，只表示其研究实验来源。**这两个模块本身不依赖增强v2、严格CUDA辅助文件或研究启动器**，只依赖PyTorch和Python标准库。它们也与实验3.1、9.1所用模型文件分别逐字节相同。独立NAF采用兼容严格CUDA的固定顺序累加；相对更早的实验3只改变该累加实现及浮点加法顺序，不改变公式和参数。

对应的纯NAF参考补丁：

- [独立NAF补丁](naf_module_handoff/exp03_model_only.patch)：修改`networks/mslau_net.py`、`kvasir_train.py`、`kvasir_test.py`；默认只启用P1/P2。
- [级联NAF补丁](naf_module_handoff/exp09_model_only.patch)：修改`networks/mslau_net.py`、`kvasir_train.py`；原`kvasir_test.py`直接构建修改后的网络，无需新增NAF专用参数。

补丁不包含新模块文件，模块需按下一节单独取出。补丁只处理NAF接入、训练正则及相关参数/日志；不修改数据读取、增强、DataLoader、随机种子策略、严格CUDA配置、学习率、优化器配方、划分或checkpoint保存规则。它们不同于实验目录下包含完整研究协议的`changes.patch`，不要混用。

## 2 在自己的分支取出代码

在接收方自己的仓库根目录执行，保持当前工作分支：

```bash
git fetch origin exp/p51-independent
NAF_REF=origin/exp/p51-independent
```

按需要取出模块和参考补丁。下列命令会写入所示目标文件；已有同名文件时应先比较并整合。

独立NAF：

```bash
git show "$NAF_REF:experiments/exp03_naf_p12_paired_strict_v2/added/networks/naf_select_reg.py" > networks/naf_select_reg.py
git show "$NAF_REF:reports/naf_module_handoff/exp03_model_only.patch" > /tmp/p51_naf_p12_model_only.patch
```

级联NAF：

```bash
git show "$NAF_REF:experiments/exp09_naf_discrete_paired_strict_v2/added/networks/naf_discrete.py" > networks/naf_discrete.py
git show "$NAF_REF:reports/naf_module_handoff/exp09_model_only.patch" > /tmp/p51_naf_cascade_model_only.patch
```

若仍是原始基线的对应代码，可以对所选补丁先检查再应用，以下两组选一组：

```bash
git apply --check /tmp/p51_naf_p12_model_only.patch
git apply /tmp/p51_naf_p12_model_only.patch
```

```bash
git apply --check /tmp/p51_naf_cascade_model_only.patch
git apply /tmp/p51_naf_cascade_model_only.patch
```

两份补丁都相对原始基线，不能依次叠加。已有MAFA或其他修改时，以下文接入点和补丁内容为参考，手动合入自己的代码；不要用整文件覆盖现有模型。补丁可应用并不代表MAFA组合的连接位置已核对。

## 3 共同的接入位置和张量接口

原始网络输入352×352时，Hiera的四级输出经过各自RFB后得到：

| 张量顺序 | 通道 | 空间尺寸 | 含义 |
|---|---:|---:|---|
| P1 | 64 | 88×88 | 浅层，高分辨率 |
| P2 | 64 | 44×44 | 浅层 |
| P3 | 64 | 22×22 | 深层 |
| P4 | 64 | 11×11 | 最深层，低分辨率 |

两版NAF都接收按`[P1, P2, P3, P4]`排列的四个`[B,C,H,W]`张量，输出同顺序、同尺寸的列表。模块不接收原始RGB图像，不直接接收通道为144/288/576/1152的Hiera输出，不负责创建RFB或解码器。

原始基线的具体接入点位于`SAM2UNetDecoder.forward`：

```python
features = [rfb(feature) for rfb, feature in zip(self.rfbs, encoder_features)]
# 在这里调用所选NAF，得到enhanced_features。
e1, e2, e3, e4 = enhanced_features
d3 = self.up1(e4, e3)
d2 = self.up2(d3, e2)
d1 = self.up3(d2, e1)
```

保留原来的`up1/up2/up3`、主输出头、D2/D3辅助头及所有输出插值设置。NAF是注册在解码器上的`nn.Module`成员，命名为`self.frequency_enhancer`，在构造函数中创建一次，不能在`forward`中临时创建。参考实现把它放在原有输出头初始化之后；沿用原代码在CPU构造完整模型后再搬至GPU的顺序。模块内部使用CPU RNG隔离区，避免新增层消耗原层和后续代码的全局初始化随机序列。

## 4 独立NAF的配置和调用

导入并在解码器构造函数中注册：

```python
from networks.naf_select_reg import PyramidNAF2D

# 放在SAM2UNetDecoder.__init__中，原有层初始化之后。
self.frequency_enhancer = PyramidNAF2D(active_stages=(1, 2))
```

在RFB之后调用：

```python
enhanced_features, naf_regularization, naf_stats = self.frequency_enhancer(features)
```

仅P1/P2各自做二维FFT、按幅度与二维频率坐标预测通道共享的频率评分，再产生软保留掩码，逆FFT后以`X + alpha * (filtered - X)`输出。P3/P4直接旁路，原解码器仍然使用全部四级特征。

| 配置 | 固定取值或含义 |
|---|---|
| 软预算 | 非DC完整频谱重数加权的掩码均值0.9，不是丢弃10%的离散频点 |
| 选择温度 | 0.1 |
| alpha | 可学习，初值0.1，上限0.25 |
| 正则对象 | 实际软掩码的平方，按二维径向频率三段加权 |
| 层级系数 | Pn三段分别为n^-1、n^-2、n^-3；P1三段同系数 |
| 正则归约 | 每层对batch及非DC重数加权频率取均值，最终`R=(R1+R2)/4` |
| 训练系数 | `lambda=0.001` |

选择器末层零初始化时，非DC掩码初始为0.9，结合alpha后对应约0.99的非DC增益；因此该版初始输出接近原特征，但不是严格恒等。`naf_stats`是已detach的诊断量，不能拿它代替可反向传播的`naf_regularization`。

参考补丁同时把`naf_stages`在训练、评估和模型构造中的默认值设为P1/P2；显式指定时也必须训练与评估一致。模块类自身的默认层级是四层，手动接入时务必写出`active_stages=(1, 2)`。

## 5 级联NAF的配置和调用

导入并在解码器构造函数中注册：

```python
from networks.naf_discrete import CascadeDiscreteNAF2D

self.frequency_enhancer = CascadeDiscreteNAF2D(channels=64)
```

调用接口：

```python
enhanced_features, naf_regularization, naf_stats, naf_fusion_stats = self.frequency_enhancer(features)
```

模块内部执行`F3=NAF3(P3,P4)`、`F2=NAF2(P2,F3)`、`F1=NAF1(P1,F2)`，返回`[F1,F2,F3,P4]`。每个双输入模块先把粗层特征上采样到当前层尺寸，再分别1×1投影；联合频谱描述预测两路频率评分，滤波后用GAP和中心化RMS生成逐通道两路融合权重，输出投影后加回当前层输入。没有独立NAF4。

| 配置 | 固定取值或含义 |
|---|---|
| 频率选择 | 每路保留`ceil(0.9*G)`个可选共轭频率对，全部自共轭锚点另行保留 |
| 训练选择 | 根据评分进行样本键控Gumbel-Top-K抽样 |
| 评估选择 | 根据评分稳定Top-K，无抽样 |
| 评分与近似梯度 | `V=sigmoid(selector)`、`P=softmax(V)`；离散前向，预算sigmoid的有偏直通近似反向，tau=0.1 |
| 残差 | 无内部alpha混合；保留外部当前层残差，输出投影零初始化 |
| 正则对象 | 评分V平方，不是离散0/1掩码平方 |
| 分段系数 | 三个二维径向频段；NAFi的local标签n=i，context标签n=i+1，系数n^-1/n^-2/n^-3 |
| 正则归约 | 每路对batch和可选共轭对取均值，六路求和除以6 |
| 训练系数 | `lambda=0.001` |

90%指频率对数量，包含固定锚点后总频率保留比例略高于90%，不表示保留90%能量。该模块与实验7的软频率级联版本不同，不能用`naf_cascade.py`替代。

**训练前向必须设置频率抽样上下文。**在创建模型后记录运行seed：

```python
model.naf_seed = int(args.seed)
```

训练循环保留样本ID，在每个训练batch的模型前向之前调用：

```python
for inputs, labels, image_ids in dataloaders[phase]:
    # 保留原来的设备搬移、标签转换及zero_grad。
    if phase == "train":
        model.progressive_decoder.frequency_enhancer.set_selection_context(
            model.naf_seed, epoch, image_ids)
    # 接原来的model(...)前向。
```

ID必须与本batch的图像顺序一一对应且稳定；原`binary_class`已经返回文件名，不需要改数据读取。使用实际运行seed和原epoch编号，不加入运行名称或重复标签。频率抽样使用自己的CPU Generator，不消费训练全局RNG；它不要求`dataset.seed`、`paired_inputs.py`或任何增强控制。`model.eval()`后采用稳定Top-K，不要求设置上下文。以上是原单卡代码的路径，接收方若改了模型包装层，应定位到实际注册的NAF成员。

## 6 训练损失和推理接口的共同改动

两版都需要在`return_aux=True`的模型输出字典中增加下列字段，保留原`logits`、`aux_d2`、`aux_d3`：

```python
"naf_regularization": naf_regularization,
```

`naf_stats`以及级联版的`naf_fusion_stats`可以供日志记录，但它们不参与训练损失。普通推理的`return_aux=False`仍然返回原形状的logits张量，推理前向也必须经过NAF。

模型创建后设置正则系数：

```python
model.naf_reg_weight = 0.001
```

在原来的主损失、辅助监督损失计算完之后，`backward()`之前，仅对训练阶段增加正则：

```python
if phase == "train":
    loss = loss + model.naf_reg_weight * model_outputs["naf_regularization"]
```

不能对正则调用`.item()`或`detach()`后再加入损失。不能把正则重复除以层数、batch大小，或在外部再算一次分段权重。验证损失和最佳checkpoint选择保持接收方原口径。

新增NAF参数必须在构造优化器之前注册，并进入原解码器参数组。原P5.1的解码器组收集所有非`encoder.`参数，自动包含它们，不需要新学习率；接收方若只列举MAFA或部分层的参数，需要确认NAF参数也被包含。

## 7 接入已有MAFA代码时的边界

交接所需接口是四级、统一通道、细到粗排序的特征列表，以及从模型输出传到训练损失的可微正则。若MAFA已有自己的融合路径，应保留其代码和输出头，在接收方选定的位置接入NAF，不用参考补丁中的整个`SAM2UNetDecoder`覆盖MAFA解码器。

这里没有MAFA分支的具体源码，因此不指定NAF在MAFA之前还是之后。两者顺序属于组合架构定义，需接收方结合实际代码明确记录；也不能仅凭模块能运行就认定与原NAF接法等价。需要核对：

- 输入仍是四级特征，顺序为细到粗；未经统一通道的Hiera输出不能直接送入此接口。
- 保持64通道可沿用现有参数量和配置。级联NAF通过`channels`构造投影，必须与实际输入一致；若各层通道不同，先明确适配，不能依靠广播或静默改维度。
- 保留MAFA自身的前向、损失项、辅助头和优化器成员，NAF正则只额外加入一次。
- 级联NAF内部已经有跨级融合；接入MAFA后会同时存在两套融合。该事实应在最终网络连接图中说明，不把它写成只增加单层滤波。

## 8 交接核对与参考来源

接收方只需核对所选模块已注册、四级输入输出尺寸不变、正则可反传且只加在训练阶段、优化器包含NAF参数、训练和评估构造相同网络，以及级联版训练前向已设置样本上下文。完整NAF权重在评估时按相同结构严格加载；官方Hiera初始化与已训练的完整P5.1 checkpoint是不同用途，不能用未匹配的完整模型权重静默跳过新层后称为同一次训练初始化。

本交接不要求接收方迁移我们的增强联动v1、修复v2、严格CUDA设置、输入哈希记录或三根输出目录。仅移植NAF时应保留接收方自己的训练协议；需要另一份可重复性修改时，参考独立的[严格v1说明](20261027_strict_v1_reproduction.md)，不要将其混入本次模块补丁。

模块SHA256，便于识别是否取错历史文件：

```text
naf_select_reg.py  6e98a18e3558562b0b2eb38d7ebcb42ab350d3181ec71f9f10d328f299091973
naf_discrete.py    8310afcfaa3d9b87c27c9d4e5e6b46521f95574adbc259f0b64a52404c11bf06
```

网络接入和完整研究实验的对照来源分别是`experiments/exp03_naf_p12_paired_strict_v2/`、`experiments/exp09_naf_discrete_paired_strict_v2/`；本文附的两个补丁只提取NAF相关部分。独立版补丁还把默认接入层级明确为P1/P2；级联版把运行seed直接存入`model.naf_seed`，以去掉对配对输入数据集属性的依赖，频率抽样键和算法不变。
