# FreDFT频域核心替换：三seed完成报告

当前5070Ti同环境同协议下，F有小幅平均增益，但最佳权重的配对收益有seed依赖。F平均IoU0.872085，较基线提高0.4208个百分点，较同参数量S提高0.2081个百分点；两项均2/3seed获胜。支持保留该适配作为候选，不宜宣称已稳定解决NAF无增益或已证明频域机制的普遍优势。

## 三seed最佳完整权重复评

每组固定20轮，以原验证IoU选择最佳epoch，再严格重载完整权重独立复评。SD为三个固定seed的样本标准差。

| 模型 | IoU均值±SD | Dice均值 | 相对基线（百分点） | 相对S（百分点） | 可训练参数 |
|---|---:|---:|---:|---:|---:|
|原始基线|0.867876±0.005966|0.921437|+0.0000|-0.2127|4,269,747|
|F：频域适配|0.872085±0.003012|0.924229|+0.4208|+0.2081|4,678,755|
|S：空间域对照|0.870004±0.006601|0.923517|+0.2127|+0.0000|4,678,755|

| seed | 基线IoU | F IoU | S IoU | F－基线（百分点） | F－S（百分点） |
|---|---:|---:|---:|---:|---:|
|1234|0.865559723|0.875174915|0.869863872|+0.9615|+0.5311|
|42|0.874653785|0.871922460|0.876673545|-0.2731|-0.4751|
|2026|0.863415663|0.869156824|0.863473679|+0.5741|+0.5683|

S相对基线3/3方向为正，但seed2026仅+0.0058个百分点，实际接近持平。F的配对差值SD分别为0.6314个百分点（对基线）和0.5920个百分点（对S），相对平均差值不小；三个seed仅作描述性探索，不作显著性推断。

现有本机旧NAF+原Decoder仅seed1234可直接比较：IoU0.864684848；F同seed为0.875174915，提高1.0490个百分点。没有补跑旧NAF另两个seed，也没有混用3090历史结果构造NAF三seed均值。

## 完整曲线与末5轮

| 模型 | 三seed末5轮IoU均值 | 对基线（百分点） | 对S（百分点） |
|---|---:|---:|---:|
|原始基线|0.864050795|+0.0000|-0.2351|
|F：频域适配|0.867984870|+0.3934|+0.1583|
|S：空间域对照|0.866401802|+0.2351|+0.0000|

F的末5轮均值在三个seed均高于各自基线；seed42虽然最佳IoU低于基线，末5轮均值仍高0.2551个百分点。对S的末5轮比较仍为2/3获胜。该结果说明平均改善并非只有某次单epoch峰值，但不等于独立数据集泛化或所有seed最优checkpoint均获益。末5轮包含收敛趋势，其轮内SD不能当成跨seed方差。

![各seed完整学习曲线](/home/lien/WorkSpace/PolypSegmentation/visualization/local_fredft_study_20261028/confirmation_curves.png)

![三seed配对比较](/home/lien/WorkSpace/PolypSegmentation/visualization/local_fredft_study_20261028/confirmation.png)

## 执行、验证与资源

- 已按确认计划完成两组真实数据smoke及8次新训练（各20轮，共160轮）；旧seed1234基线和NAF复用，无重复挑选运行。新增正式训练及复评合计93.02分钟，不含实现、smoke和逐图推理。
- 首轮seed1234确实同时超过基线和S，才触发seed42/2026本机基线/F/S六次确认；seed42的负向结果完整保留。
- 352/b12/valid6/BF16/nw2/CPU4/严格CUDA/paired_aug_v2、原880/120划分、原mask>0、三头Structure权重1、AdamW两组lr0.001、weight_decay0.0005、20轮Cosine沿用。
- 每次从同一官方Hiera重新初始化；保留原Decoder/RFB/三级UpBlock/三输出头，以及NAF三级级联/外围投影/融合门控/零输出残差。选择器、离散频率预算及评分正则删除。
- F/S均4,678,755可训练参数，完整初值逐位一致；每seed的原层初值/RNG与基线一致，40条输入指纹完全匹配。
- 35项CPU/工作流检查通过；真实smoke的初始输出恒等、三级任务梯度、重复eval、序列化和完整严格重载通过。
- 全部九个基线/F/S记录的87项产物检查通过，涵盖源码/driver快照、输入helper、环境和27份完整权重档案哈希；各次训练的最佳权重已严格重载复评，原指标误差<1e-6。
- F/S首轮smoke峰值分配显存13.425/13.181GiB；两组含更多多尺度卷积，均高于原NAF的4,325,865可训练参数。比较旧NAF还包含移除其评分正则的差别。

## 机制与收益边界

这是FreDFT核心的跨尺度适配，未复现整篇检测系统：MFDA使用同路Q/K无共轭频谱乘法响应调制异路V，随后接多尺度FDFFL；外围仍沿用NAF门控，而非论文最终concat融合。未加入LFEM、CGMM或额外跨尺度直连。实现来源和边界见README.md。

FDFFL的FFT与IFFT之间仅有通道拆分/拼接，其数学等价空间实现已用FP64/梯度验证；原TF32会放大少量FFT舍入误差，正式协议未修改。F/S对比体现全局频谱乘法响应与空间逐点响应的差别，也含FDFFL浮点舍入差；不能排他地证明FFT计算形式本身或特定频段选择带来收益。

四组seed1234最佳权重的逐图纯推理已复现原指标。F相对基线55胜65负，均值由部分较大的改善带动；4px边界带内IoU为0.597965，基线0.598190，未观察到边缘指标提高。带外FP/FN减少较明显，不能将此直接解释为已证明去除背景纹理的因果机制。这项区域诊断仅覆盖seed1234。详细错误计数、成功和失败案例见首轮报告。

本研究仍共用120张验证图像选epoch，历史test文件与其相同。三个seed是优化随机性的重复，不是三个独立数据集。下一步优先固定候选在独立数据上检验；本批没有自动追加新模型、新seed、训练轮数或阈值搜索。

## 可复核产物

- [首轮报告](/home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/studies/fredft_frequency/RESULTS_SEED1234.md)
- [完整结果与审计JSON](/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_fredft_study/confirmation_summary.json)
- [三seed逐运行CSV](/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_fredft_study/confirmation_runs.csv)
- [首轮逐图CSV](/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_fredft_study/per_image.csv)
- [核心实现](/home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/studies/fredft_frequency/fredft_core.py)、[训练入口](/home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/tools/fredft_study.py)、[条件监督器](/home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/tools/confirm_fredft_study.py)、[分析入口](/home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/tools/analyze_fredft_study.py)
- 每个CSV中的records指向完整日志/源码快照目录；工作空间checkpoints下同run_id目录保存三份完整权重。
