# 实验九局部残差语义门控

2026-10-29用户确认执行。宿主为实验九离散级联NAF＋原Decoder，**不是实验三独立NAF**。

原三级连接为F3=NAF3(P3,P4)、F2=NAF2(P2,F3)、F1=NAF1(P1,F2)。本次只改各NAF内部的原始局部残差：

- 原版：Y=P+R(P,C)。
- LocalGate：Y=M(P)P+R(P,C)。
- DeepGate：Y=M(Up(C))P+R(P,C)。

保留原投影、离散90%频率选择、样本键Gumbel、STE、GAP/RMS两流融合、零初始化输出投影、
训练评分正则0.001、RFB/三UpBlock/三头及全部训练数值协议。没有替换跨层UpBlock。

门控借鉴FAME-Net Fig.5的mask生成器：普通3×3与膨胀3×3求和，经1×1，另加输入的1×1投影，再1×1生成单通道mask。
本次隐藏宽16、dilation2是明确的工程选择，论文未给出这些默认值。最终卷积权重和偏置均零初始化，
使用2*sigmoid使初始mask=1；这不同于论文直接sigmoid，且允许增强和抑制。没有完整SGG的cross-attention、MoE或BC损失。
Local/Deep参数量及完整初值一致，仅mask输入不同。所有新门控初始化在独立RNG上下文，保持原层初值和外部RNG。

固定seed1234/42/2026各20轮。原NAF9 seed1234复用本机旧完整运行；补原NAF9 seed42/2026及两门控各三seed，共8次新训练。
三组真实b12 smoke全部通过才进入正式队列；每次正式训练均重新加载官方Hiera，不继承smoke状态。
先完成所有固定seed再比较，不因首seed高低选择继续。原无模块B和仅外围融合H均复用现有三seed。

环境协议：5070Ti、BF16/NAF核心FP32、352、train12/valid6、workers2/CPU4、paired_aug_v2、严格CUDA；
AdamW两组lr0.001/weight_decay0.0005、Cosine20逐轮、Structure主/D2/D3各1。阈值FP32 sigmoid>0.5。
880训练/120验证，验证集兼用于选权重，**无独立测试集**。

新增每次对齐旧基线40条输入，保留40阶段模型/缓冲/AdamW/调度器/全RNG哈希，三份完整最佳权重及严格重载复评。
旧NAF9 seed1234没有完整初值/阶段状态哈希，不反推不存在的历史证据；通过其源码、原层初值、环境、输入和权重审计后复用。
本批未安排同seed完整重复训练，不能把严格配置或CPU短轨迹重复等同完整训练重复性证明。

入口：tools/run_sgg_residual_study.py默认只预览，--run才smoke和训练；独立输出：
../work/sgg_residual_study_20261029/{experiments,checkpoints}；审计../analysis/20261029_sgg_residual_study。
完成后自动分析各seed/均值/配对差、末5/10轮、逐图IoU、小息肉、4px边界/带外错误、两张既定失败图，
并保存各层mask前景/背景均值与空间SD，不通过挑图或调阈值解释增益。

实验三建议：旧0–3批次均值0.873464，确为当批最好；严格v2为0.870246，对应基线0.871383，
2/3 seed提高但均值下降0.1137百分点。暂未有本机5070Ti实验三匹配三seed，不把跨环境旧最高值当当前最优。
后续若验证独立NAF路线，先补 unchanged 实验三本机三seed，再选择单输入可适配的FreqSelect；
FDFFL的FFT混合已有空间等价性，优先级低。MFDA双流、深层SGG、级联融合/跨层beta均会改变其独立性质，不强行移植。
本次没有自动授权启动实验三、F_eq、AdaDR/AdaKern或超参数搜索。
