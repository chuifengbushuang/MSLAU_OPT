# exp04_naf_all_no_reg：四层NAF去正则

对应对照为exp01_naf_all。相对该配置仅改变id和naf_reg_weight=0.0；模型、启用层级、90%预算、残差参数及训练和评估设置完全相同。
继续从冻结实验0＋shared_naf_v1构建，不从其他实验的权重微调，不修改已完成实验的代码。
正则R仍可用于日志诊断，但乘以0，不参与训练目标；验证/权重选择原本就不加入正则。
每次仍保存三份原始命名权重，输出到约定的三个根目录。

这是分段正则的消融，不是新的NAF设计；用三seed与原配置逐seed比较。冻结代码的Albumentations增强轨迹不保证同seed逐次相同，因此小差异需重复确认。

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/run_experiment.py exp04_naf_all_no_reg --seed 1234 --gpu 0 --run
```
