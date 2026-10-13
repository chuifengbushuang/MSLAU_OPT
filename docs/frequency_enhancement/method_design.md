# AFFE：NAF-inspired 二维适配设计

## Phase 0：基线和范围（2026-10-13）

基线为本地 `lien1012` 与本地 `origin/lien1012` 共同指向的
`81b19e2c801efcf74826b8127f5ea712b0247ec5`；工作树干净，功能分支
`feat/p5-naf-frequency-enhancement`。没有通过 fetch 确认远端是否更新。
完整阅读 AGENTS.md；其引用的 handover.md 在本地缺失。
实际源码与 PPT 的 P5.1 数据流一致：冻结 Hiera-L（48 个 block prompt adapter
可训练）→ 四个 RFB → 64 通道四级特征 → 三个插值/Concat/卷积 Up → 三路 head。
PPT 的 adapter 简写与实际 Sequential 有差异；实现以源码的两个 GELU 为准。
本轮仅在 RFB 后增加 AFFE，不修改融合、主干、adapter、head、原损失或数据划分。

## 论文阅读与公式审计

主文献：Hu et al., *Noise Adaptive Filtering Neural Network Under Multiscale
Features*, IEEE TASE 22 (2025), pp.17050–17062，DOI 10.1109/TASE.2025.3579694。
依据用户提供的 `papers/Noise_Adaptive_Filtering_Neural_Network_Under_Multiscale_Features.pdf`，
重点逐项核对 PDF 第 4–5 页（印刷页 17053–17054）的 Fig.3–5、Eq.(5)–(9)、Algorithm 1。
工作定位为 **Paper-based Reimplementation + NAF-inspired 2D Adaptation**，
不是官方代码的逐行复现。原任务是工业时序软测量回归，CNN 分支尺度是卷积核尺度，
不等同于 Hiera 的空间金字塔 stage。

| 原论文 | 本轮实现与偏差 |
|---|---|
| Eq.(5) 时间轴 DFT | 在 H/W 上 rfft2，norm=ortho，batch/channel 不参与变换 |
| Eq.(6) GAP→Linear→sigmoid→频率维 softmax | 实值 log1p 幅值、全局幅值统计、二维频率坐标→两层 1×1 MLP→有界软门控；不对复数直接做 GAP/Linear |
| Eq.(7) λ L2(V)/n、/n²、/n³ | 用实际 FFT 坐标划分径向频段，对门控相对 1 的偏离做分段二次惩罚，各 stage 相同规则；不用 stage 编号代替 n |
| Eq.(8) 依概率随机选 ceil(0.9M) 个频点 | 用确定、可微的实数软掩码乘原复谱；不做硬采样、不承诺 90% 稀疏率 |
| Eq.(9) Linear→Padding→IFFT | 不丢弃频点，无需恢复索引或 Linear/Padding；irfft2 显式传回原 H/W |
| Fig.3 / Algorithm 1 | 只适配频域过滤链；不实现跨尺度交换、多尺度 CNN 或 GRU |

歧义与解释：

- Eq.(5) 产生复数，论文却用实空间记号；未说明 GAP 如何处理复数，或 Linear 是否复值。
  这里只由幅值生成实数门控，保留原相位。GAP 的精确轴也未给出。
- Eq.(6) 的正文**明确** softmax 沿 Fourier component dimension；但多维张量排列、
  复数处理和 GAP 后维度没有完整工程定义。不能称归一化轴完全未知。
- Eq.(8) 随机选点未说明有放回与否、梯度估计方法；硬索引不保证概率网络可导。
  若将来研究 Gumbel/straight-through，应另做消融，不在本版猜测作者实现。
- Eq.(9) 未定义 Linear 的轴、复值处理、Padding 的频率索引恢复方式和共轭约束。
- 第 3 页称小尺度偏低频、大尺度偏高频；第 4 页 Eq.(7) 前的文字反向描述。
  Fig.4 和 Eq.(7) 对 n≥1 给出同一系数：n=1 三段相等，n>1 高频惩罚更小。
  Fig.5 画的是 1/n、1/n²、1/n³ 的**系数衰减**，不是实测门控；第 5 页的
  “小尺度约束弱”句子与公式在小 n 系数较大的事实不一致。L2 是否平方亦未唯一确定。
  Algorithm 1 从 n=0 开始，与 Eq.(7) 的分母有索引歧义。因此本版不用原尺度偏置。

第二篇 `papers/2508.19746v1.pdf` 是 *SPLF-SAM*，不是 NAF 的另一个版本。
其 §2.2 / Eq.(1) 的 MAFA 对 8×8 patches 做 FFT、乘可学习核并重建，
与这里由每次输入产生门控不同。其 UMFEB、prompt bank、融合与 decoder 不在本轮范围。

## 预先确定的二维方案

对每级独立计算 Q=rfft2(F)，A=log(1+|Q|)。门控输入包含 C 通道 A、
一个该样本 A 的全局均值广播通道、|2fy| 与 2fx 坐标，保留所有频点位置。
两层 1×1 Conv（C+3→16→C，SiLU）跨通道建模，不跨 stage。
M=1+0.5 tanh(g)，范围 (0.5,1.5)，因此无随频点数量增长的 softmax 幅值衰减。
最后一层权重 N(0,0.02)、bias=0.01；不全零初始化，保证首步前层有梯度路径。

仅在半谱 fx=0 和偶数 W 的 Nyquist 列，将 M[ky] 与 M[-ky mod H] 平均。
这两列必须满足行共轭约束。其余列由实数逆变换隐式补全对应的负 fx，
无需把所有列都强制为行对称。正实权重不改变相位。

F_filtered=irfft2(MQ,s=(H,W),norm=ortho)，
F_out=F+sigmoid(a)(F_filtered-F)，初始 alpha=0.1，各 stage 独立 a。
有效谱增益为 1+alpha(M-1)，始终在 (0.5,1.5)，初始化输出接近输入且梯度非零。
实际数值及梯度以 verification_report 为准，不能把随机非恒等变化解释为去噪。

AMP 区域内部关闭 autocast：FP16/BF16/FP32 输入在 FP32 做 FFT、门控和残差，
FP64 输入使用 FP64；返回输入 dtype。模型参数显式转换到工作 dtype 的操作仍可导。

频率半径 r=sqrt(fy²+fx²)/sqrt(0.5²+0.5²)，边界默认 (1/3,2/3)，
不按本图最大可达频率重新缩放（奇数尺寸也使用相同物理定义）。
各段 R_b=mean_b((M-1)²)，平均采用半谱列重数：DC/Nyquist=1，其余=2。
R_stage=Σ w_b R_b / Σ有效段w_b，默认 w=(1,1,1)；空段不参与分母。
四级 R 取启用 stage 均值，全部关闭时为零。总损失为原分割损失+λ_freq R，λ 默认 0。
这是中性的分段能量约束，抑制过强修改；本身偏好恒等，**不保证防止常数门控塌缩**。
自适应性由任务损失驱动，后续必须监测门控方差和输入敏感性。
诊断只在请求时生成并 detach，无模块级计算图缓存。
