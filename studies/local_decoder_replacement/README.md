# 当前机器上的 Decoder 替换比较

用户于2026-10-28授权在当前机器实际训练，固定seed1234，先smoke，再检验无NAF基线是否在0.8786±0.005内，之后比较两个NAF模型。历史3090结果只作参考，不能与当前硬件/数值环境混成配对结果。

三组均直接从官方Hiera初始化，保留原编码器及BlockAdapter、四个RFB和三个输出头：

| 组别 | 路径 |
|---|---|
| baseline | 原RFB → 原三个UpBlock → 原主/辅助头，无NAF |
| naf_decoder | RFB → 实验9离散级联NAF → 原三个UpBlock → 原主/辅助头 |
| naf_replacement | RFB → 同一实验9离散级联NAF → 原主/辅助头，删除三个UpBlock |

替换版主头读F1，D2头读F2，D3头读F3；没有闲置的UpBlock参数。它比较的是整条解码路径，不是等参数量对照。两个NAF组的模块、频率选择及正则一致：训练按样本键Gumbel-Top-K，评估稳定Top-K，六路V正则均值，lambda0.001。NAF零输出投影、两组共享初始化与原实现相同；替换版初始输出不等于原基线，因它移除了原UpBlock。

三组使用原352尺寸、原增强操作及概率、880/120划分、原mask>0、Structure主+D2+D3（均1）、AdamW两组lr0.001、weight_decay0.0005、20轮Cosine至0。采用已验证paired_aug_v2控制相同样本/epoch的增强，严格CUDA执行，保留训练drop_last。新环境无法保证复现旧3090的单次最高值。训练batch、worker、CPU线程及FP32/BF16由smoke决定，正式比较三组保持一致。

入口为`tools/local_decoder_study.py`。它通过现有构建器展开基线v2源码，保存到每次记录目录的source/，另复制原实验9模块和本工具快照，不编辑冻结基线。保存共享原层初值/RNG哈希、完整环境、源文件哈希、逐轮输入指纹、梯度审计、时间、最佳IoU及Loss权重和最佳IoU的严格重载复评。

数据/权重/输出默认取仓库父目录，支持命令参数覆盖：

```bash
NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ../work/local_train_env/bin/python -B -u tools/local_decoder_study.py \
  --mode smoke --variant baseline --batch 12 --workers 2 --threads 4 --precision bf16
```

将mode改为train执行20轮训练，将variant改为naf_decoder或naf_replacement执行对应架构。正式运行前使用完成smoke后的共同设置。每个进程都从初始权重开始；smoke不保存正式权重。输出在父目录experiments/local_*、checkpoints/local_*下；默认不导出预测图。

本研究有意义的结构检查在tests/test_local_decoder_study.py：共享层初值/RNG、追加版零投影输出与原基线一致、替换版实际移除UpBlock且三头确实读取F1/F2/F3、纯任务损失传到三个NAF选择器/门控及四级输入、完整state_dict严格重载和非零滤波分支的确定性推理。
