# 实验7：三级双输入NAF级联

从冻结的原始P5.1提交81b19e2直接构建。changes.patch只修改原网络接入、训练正则/诊断及评估模式检查；新增模型集中在added/networks/naf_cascade.py。
不引用实验1–5的模型文件，不修改冻结基线、旧NAF或已有结果。当前仅实现完整候选；融合对照、去正则对照待后续确认。

## 两条路径与尺寸

352输入，Hiera四级通道144/288/576/1152；原RFB统一为64通道，空间尺寸88/44/22/11。

```text
D4 = P4
F3 = NAF3(P3, P4)       D3 = 原UpBlock3(D4, F3)
F2 = NAF2(P2, F3)       D2 = 原UpBlock2(D3, F2)
F1 = NAF1(P1, F2)       D1 = 原UpBlock1(D2, F1)
```

三个模块分别输出22²/44²/88²；F3/F2同时进入后续NAF及Decoder。上下文不使用D3/D2，不detach。
P4原样进入Decoder，没有独立NAF4；P4在NAF3的context分支、上采样至22²后参与滤波。
原SAM2UNetUp、RFB、side1/side2/head定义及参数名不变；D3/D2辅助输出与352最终输出不变。

## 来源与适配边界

- [SPLF-SAM](https://arxiv.org/html/2508.19746v1)，§2.1、2.3：借鉴相邻层递进融合及以本级特征为残差的思路。UMFEB是融合模块，MAFA才是其频域适配器；本实验没有复现完整UMFEB、MAFA、Prompt Bank或其提示解码器。
- Hu et al., *Noise Adaptive Filtering Neural Network Under Multiscale Features*, IEEE TASE 22 (2025), DOI [10.1109/TASE.2025.3579694](https://doi.org/10.1109/TASE.2025.3579694)，§III-B/C、公式(5)–(11)：借鉴频率选择、分段约束及先滤波再按重要性交换融合。确定性双路门控不是原文的随机交换；不使用GRU。
- 当前六分支均处理二维特征。时间卷积尺度与图像层级不是同一个概念；层级编号的正则先验、全局频谱与边缘之间的关系均需实验验证。没有直接边缘监督，不预先宣称边界已改善。

## 单个双输入模块

1. 深层输入双线性上采样到本级大小，align_corners=True，与原Decoder一致。两路各用独立64→64无bias的1×1投影，不新增BN/Dropout。
2. FP32 rfft2(norm=ortho)。每路通道平均幅值除以非DC频率重数加权平均幅值，取log1p。拼接本级/上下文描述、两者绝对差、|2fy|、2fx，形成5通道描述。
3. 5→16→2的1×1卷积选择器（GELU，输出sigmoid）联合预测两路选择分数；温度0.1，分别分配g=min(1,c*p)的90%软预算。不是二值保留率或能量比例，不让两路竞争同一个预算。
4. rFFT内部列重数2，kx=0和偶数宽度Nyquist列重数1；DC预算权重0、掩码固定1。边界列共轭ky位置使用相同实数掩码，保留各自相位，irfft2指定原H/W。不直接相加两个复数频谱。
5. 每路滤波残差为X+alpha*(IFFT(g*FFT(X))-X)，alpha=0.25*sigmoid(a)，初始0.1。仅该滤波子步骤保持DC且不增加能量；整个融合残差没有此保证。
6. 每路提取GAP及centered_RMS=sqrt(mean((X-GAP(X))²)+1e-6)，拼成4C通道，经4C→16→2C的MLP生成两路逐通道softmax权重。仅GAP无法观察保留DC的滤波变化，RMS补充非DC能量统计。
7. 两路权重和为1，空间域加权后经64→64无bias的1×1输出投影，再加未经投影的Pi。选择器末层、融合门控末层、输出投影均零初始化，初始两路权重0.5，输出严格为Pi。

三个模块参数独立，共新增56,124个可训练参数（每模块18,708）。新增随机初始化在CPU RNG隔离区，不改变原参数或后续CPU随机状态。
零输出投影使第一步部分上游分割梯度为0，投影更新后会打开；不是永久冻结。合成测试验证多步后两路投影、选择器、融合门控均能获得非零梯度。
不同样本独立计算频率预算与融合权重；不建立88²×88²的空间注意力矩阵。
两路在22²/44²/88²滤波，比旧四层单路多处理约1.98倍空间采样点；参数少或模块数少不等于耗时更少。需要3090实测峰值显存，不预估实际速度。

## 正则：固定六分支归约

在受体网格上，用rho=sqrt((2fy)²+(2fx)²)/sqrt(2)，阈值1/3、2/3分三段。沿用1/n、1/n²、1/n³的形状，惩罚实际预算掩码g²。

| 模块 | local的n | context的n |
|---|---:|---:|
| NAF1 | 1 | 2 |
| NAF2 | 2 | 3 |
| NAF3 | 3 | 4 |

context中的F2/F3已混合层级，因此n=i+1只是明确的层级先验，不代表纯净的原层特征或物理尺度。更高频系数较小，不能描述成更强地惩罚高频。

```text
R_i,b = sum_batch,freq(w * coefficient(n,band) * g²) / (B * sum_freq(w))
R = sum(i=1..3, b=local/context) R_i,b / 6
L_train = 原Structure主损失 + 原D2损失 + 原D3损失 + 0.001*R
```

分母固定6，不使用旧实验的4，不把两个输入重新编号为n=1/2，也不按像素数直接相加。NAF1 local的n=1三段系数相同。
验证Loss不加正则，沿用原来的权重选择口径。每epoch按模块/分支记录三段掩码均值、alpha、R、频率内选择标准差、预算、batch内图像间掩码标准差；另记录上下文权重、通道/图像间门控标准差及残差RMS比。
诊断不参与梯度，图像间标准差是batch内统计、按样本数汇总，不是全数据集总体标准差；batch1记0。

## 训练与输出

原352/b12/nw8/e20、Structure、两组LR1e-3、AdamW与原调度、D2/D3权重1、boundary0、官方Hiera初始化、880/120划分及增强均不变。
实验7不加载旧分割checkpoint。三个种子1234/42/2026各一次，仍不能保证增强轨迹跨运行一致。
输出仍按P51_OUTPUT_ROOT下checkpoints、experiments、visualization三根目录组织，同run_id。
保留kvasir_best_model.pth、best_iou_*、best_loss_*三份原始命名权重；默认评估debug=false，无图时不创建图像目录。

先更新本分支，再做合成检查；以下只执行合成输入前向/反向及测试，不读取训练集、不启动正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 EXP07_FULL_SMOKE=1 EXP07_DEVICE=cuda EXP07_SMOKE_BATCH=12 /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp07_naf.py' -v
```

默认读取原官方Hiera路径；如已设置HIERA_CHECKPOINT则用该文件。完整检查打印batch12及峰值allocated显存，使用原Structure主/辅助监督验证反向；不在低显存时自动降低正式batch。

合成检查通过后，三个种子依次运行，失败即停止（这里是用户手动执行的正式训练命令）：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_SEED in 1234 42 2026; do
        /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp07_naf_cascade --seed "$P51_SEED" --gpu 0 --run
    done
)
```

不带--run只展开源码并预览，结束清理临时源码。模型定义无旧NAF兼容开关，严格加载本实验权重。

本地验证为PyTorch2.13 CPU：51项中49通过，旧NAF及实验7的官方权重完整Hiera检查各跳过1项；10个展开Python文件语法、14份冻结基线与原提交逐字节一致检查通过。本地没有读取服务器训练数据、加载其官方权重或执行3090 batch12检查。
