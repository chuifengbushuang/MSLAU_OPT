# 当前服务器补跑原版实验三

2026-10-29用户授权：固定seed1234、42、2026各20轮，保持现有训练配置不变，比较已有基线B、H及深层语义门控。

直接展开`exp03_naf_p12_paired_strict_v2`，使用其中未修改的完整模型及NAF文件。仅P1/P2独立处理：
Y=X+alpha*(filtered(X)-X)，90%软频率预算、temperature0.1、alpha初值0.1/上限0.25，DC保留；
原有固定顺序累加沿用严格v2源码。正则是实际软mask的平方按原尺度加权，(R1+R2)/4，训练权重0.001。
新增参数仅原实验三的164个，不接入SGG/FreqSelect/MFDA/FDFFL/跨尺度连接，不调整模块超参数。

沿用当前5070Ti协议：官方Hiera-Large每次新初始化，Kvasir-SEG 880train/120valid，352，
BF16且NAF核心FP32，train12/valid6、workers2、CPU4、paired_aug_v2及严格CUDA；
AdamW两组lr0.001、weight_decay0.0005，CosineAnnealingLR T_max20/eta_min0，逐epoch更新。
Structure主/D2/D3各1，验证不加NAF正则；阈值为FP32 sigmoid>0.5。880train中drop_last实际每轮876张。
归档源码config保留历史设置用于审计；实际本机训练配方以每次effective_config.json为准。

先真实b12六步smoke，核对原模型接入、P3/P4旁路、P1/P2无跨层依赖、初始预算/alpha、各层梯度、重复推理与严格权重重载。
正式三次从官方权重重新初始化，不继承smoke。所有seed固定运行，不按首seed成绩选择继续；不重跑B/H/Deep。

每次与旧基线核对40条输入，记录40阶段完整模型/缓冲/AdamW/调度器/全RNG哈希；
共享原层初值、环境和初始全RNG/runtime与已有参照核对；保留三份完整最佳权重，严格重载复评。
验证集用于选择权重及比较，没有独立测试；本批没有同seed完整重跑，不将严格配置等同完整重复性证明。

启动器`tools/run_exp03_independent_study.py`默认只预览，`--run`才执行；使用锁和唯一目录，不自动重试失败运行。
完成后自动比较12个模型的每seed/均值/配对差、末5/10轮、学习曲线、逐图预测、小息肉/4px边界及既定两张失败图。

独立输出：`../work/exp03_independent_study_20261029/{experiments,checkpoints}`。
审计和表格：`../analysis/20261029_exp03_independent_study`。
图像：`../visualization/exp03_independent_study_20261029`；报告：本目录RESULTS.md。
