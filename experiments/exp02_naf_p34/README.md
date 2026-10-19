# exp02_naf_p34：仅P3/P4位置消融

相对冻结实验0构建，源码差异来自同一份experiments/shared_naf_v1；不引用其他实验结果或权重。构建后为独立全量源码。
本配置训练与独立评估均固定naf_stages=3,4，只实例化启用层的模块。其余训练参数与实验0一致。
层级固定P1=1、P2=2、P3=3、P4=4，不重编号。R_n是单层按batch及非DC频点重数加权的g²分段惩罚均值。
正则统一为R=sum(启用层R_n)/4，lambda=0.001；不按启用模块数量重新平均。
因此每个保留层的损失系数始终为0.001/4，移除层不放大其他层的正则。验证和权重选择不加入正则。
P1的三段系数均1；它的正则没有高低频偏向，但会约束掩码偏离均匀分配。

输出按checkpoints/experiments/visualization同run_id保存，保留三份原始命名权重，默认不导出预测图。
完整方法在仓库experiments/shared_naf_v1/README.md。展开目录自带run.py/config.json，可独立运行。

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/run_experiment.py exp02_naf_p34 --seed 1234 --gpu 0 --run
```

旧exp01_naf_select_reg（73e38b8及更早）是两层且分母2，不等同于本轮新协议。旧输出不覆盖、不改名冒充新结果。
