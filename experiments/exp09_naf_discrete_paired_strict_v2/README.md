# 实验9：离散NAF，同协议v2对照

当前暂缓：用户要求优先直接检验[原实验3](../exp03_naf_p12_paired_strict_v2/README.md)，以下六次命令保留用于后续机制分析，目前不执行。

对应历史实验9的同协议重测，不是新架构。原实验0–9、v1/v2基线定义和旧结果不改。
直接从冻结基线81b19e2构建，NAF模型文件与旧实验9逐字节相同。

保留三级双输入级联和原Decoder；没有独立NAF4，没有内部alpha。每路保留ceil(0.9G)个共轭频率对和全部自共轭锚点，训练使用样本键控Gumbel-Top-K、评估使用稳定Top-K。评分V的分段平方正则为六路均值求和/6，lambda=0.001，仅训练加入。梯度仍为原实验9的有偏预算sigmoid近似，不借本次迁移改变选择器、温度或正则。

本次只迁移到已验证的paired_aug_v2与strict_cuda_v1，训练/评估都严格执行。选择噪声键仍为naf_discrete_v1，不改原频率抽样序列。352/b12/nw8/e20、官方初始化、原Structure三头监督、AdamW两组1e-3、Cosine及checkpoint选择规则不变。新增56,118参数。

输入helper与v2基线仅record_initial_state不同：额外记录剔除新模块后的baseline_state_sha256。与同seed基线及仅融合的样本顺序、增强图像/mask哈希应逐轮相同；共享原层及融合层初始化/RNG保持匹配。新增记录不改变训练算法。

固定三组：已有exp00_baseline_paired_strict_v2、本实验、exp08_cascade_fusion_paired_strict_v2。
基线已有1234/42/2026，各按一次计入均值，1234第二次仅用于确认重复性。其IoU为0.8682272643/0.8691121638/0.8768093199，均值0.8713829160、样本SD0.0047201860；不以较低的旧基线混入本批对照。

主要比较预先固定为每个seed的最佳验证IoU之差及三seed等权平均差，同时报告样本SD、最后5轮/最后一轮、Dice和增益方向，不挑最高重复或删掉不利seed。验证120用于选择和报告，并非独立测试集。三seed是探索性证据，不等同显著性或跨数据集验证。

NAF与仅融合对比同时移除了频率选择和正则，衡量二者组合收益。若有稳定正向迹象，下一步才考虑同协议去正则，以区分选择和约束的贡献；若不及仅融合，就不把融合收益写成频率收益。

先执行[仅融合说明中的严格GPU合成检查](../exp08_cascade_fusion_paired_strict_v2/README.md)，通过后在仓库目录顺序运行六次：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    export P51_REPEAT=1
    for P51_SEED in 1234 42 2026; do
        for P51_EXPERIMENT in exp08_cascade_fusion_paired_strict_v2 exp09_naf_discrete_paired_strict_v2; do
            /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py "$P51_EXPERIMENT" --seed "$P51_SEED" --gpu 3 --run
        done
    done
)
```

不重新激活环境、不重装依赖。GPU占用仅提示；错误则停止。权重、实验记录和图像继续按run_id进入三个既定根目录，保留原三权重命名。代理不代启动正式训练。
