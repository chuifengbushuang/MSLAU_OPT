# 零初始化跨尺度跳连接消融

已完成20轮：最佳权重复评IoU0.856671427，低于旧替换版0.861666300；完整四组对比、beta和审计见[实际结果](CROSS_SCALE_SKIP_RESULTS.md)。

用户确认在旧NAF替换版上增加三条跨尺度直接路径，并实际执行smoke和20轮训练。旧三组结果保持不变，作为本机对照，不重跑。

```text
D4 = P4
D3 = NAF3(P3, D4) + beta3 * Up(D4, 22×22)
D2 = NAF2(P2, D3) + beta2 * Up(D3, 44×44)
D1 = NAF1(P1, D2) + beta1 * Up(D2, 88×88)
主头读D1，D2/D3辅助头读D2/D3；各头输出插值到352×352。
```

每个NAFi本来已包含Pi残差，不能再重复加Pi。新增beta1/2/3均为无约束可学习FP32标量、初值0；只增加3个参数，可训练参数总数3,993,324。上采样为bilinear、align_corners=True，新增路径FP32计算后恢复原特征dtype；每一级相加后的Di同时送对应输出头及下一级NAF。无新卷积、无UpBlock、无detach，保留原NAF投影、门控、90%选择、六路正则和初始化。

模块源为本目录naf_cross_scale_skip.py，包装原实验9的stages，保留其原参数名字/初值，zeros初始化不消费RNG。入口是tools/local_decoder_study.py的新变体naf_replacement_skip。完整源文件及其哈希随每个运行归档。

共同设置保持seed1234、BF16、352、batch12、验证batch6、2worker、CPU4线程、严格CUDA、paired_aug_v2、AdamW两组lr0.001、weight_decay0.0005、Cosine20轮至0、原三头Structure监督及NAF正则0.001。官方Hiera重新初始化，smoke更新不进入正式训练。

## 已完成的smoke

2026-10-28真实Hiera、真实数据、batch12、BF16 smoke通过：

- beta=0时，四级特征、正则和全部NAF诊断量与原替换级联逐元素完全一致。
- 三个beta在第一次反向、尚未更新且值为0时均有有限非零梯度；更新后能离开0。
- 第7个训练步的编码器适配层、RFB、主头、NAF选择器/门控/输出投影及beta梯度全部有限非零，无缺失参数。
- 推理输出形状正确、重复eval逐元素相同、参数保存读取前后输出一致。
- 稳态中位约0.335秒/训练步、峰值分配9.268GiB、预留10.123GiB；预计完整训练/验证/存盘约10–13分钟。

smoke目录：父目录experiments/local_smoke_naf_replacement_skip_seed1234_20261028_013411_88572。

新增4项CPU测试通过，覆盖非零残差下零beta前向/RNG/原参数一致、非零beta确实传到后续NAF和三头、纯任务损失对零beta和三层频率分支可反传、非零beta的完整权重严格重载。旧模型专项4项及工作流25项也通过，共33项。

正式命令（已授权实际执行）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lien/WorkSpace/PolypSegmentation/work/local_train_env/bin/python -B -u \
  /home/lien/WorkSpace/PolypSegmentation/MSLAU_OPT/tools/local_decoder_study.py \
  --mode train --variant naf_replacement_skip --batch 12 --workers 2 --threads 4 --precision bf16 --seed 1234 --epochs 20
```

每轮额外记录三个beta，最终记录最佳IoU权重中的beta。对比时必须核对40条输入指纹、共享层初值/RNG、有效配置及原模型/NAF代码；driver新增了变体/检查/日志，因此不能要求整个driver文件哈希仍与旧版相同。应单独审查训练/损失/优化计算是否保持一致。结果为单seed同验证集比较，不声称稳定收益。
