# 2026-10-28 跨尺度跳连接消融结果

新增三条零初始化跨尺度跳连接的版本已完成20轮，并严格重载最佳IoU权重复评。在本次固定seed1234、BF16、batch12、相同20轮预算下，没有改善：IoU0.856671427，较原NAF替换版下降0.4995个百分点，较原始baseline下降0.8888个百分点。

| 模型 | 最佳权重IoU | Dice | 末5轮IoU均值 | 末5轮IoU样本SD | 最佳轮次（从1计） | 训练/复评分钟 |
|---|---:|---:|---:|---:|---:|---:|
| 原始baseline | 0.865559723 | 0.918729888 | 0.864486756 | 0.000884040 | 19 | 8.64 |
| NAF+原Decoder | 0.864684848 | 0.917038641 | 0.861039123 | 0.002480600 | 16 | 10.32 |
| NAF替换Decoder | 0.861666300 | 0.916309873 | 0.859946155 | 0.002080449 | 20 | 9.81 |
| NAF替换+跨尺度跳连接 | 0.856671427 | 0.913853868 | 0.852304102 | 0.005565980 | 20 | 9.87 |

前三组直接复用已完成结果，没有重跑或挑重复。四组均采用相同的120张验证集及最佳权重选择方法。新增版本末5轮均值比旧替换版下降0.7642个百分点；末5轮波动也较大，但这些数值包含收敛趋势，不能解释成跨seed稳定性结论。两个替换版最佳值均在最后一轮，本结论只针对相同20轮预算。

## 实际结构与beta

```text
D4=P4
Di=NAFi(Pi,D{i+1})+beta_i*Up(D{i+1})，i=3,2,1
```

新增路径绕过本级频率筛选，逐级相加后的Di继续进入下一级NAF和本级输出头。保留原NAF内Pi残差、RFB、三头；无UpBlock或新卷积。只增加3个标量，可训练参数3,993,324。beta初值均为0，未使用非负约束或额外正则。

最佳IoU权重中的beta：

| 参数 | 初值 | 最佳权重值 |
|---|---:|---:|
| beta1 | 0 | -0.107307635 |
| beta2 | 0 | 0.112187989 |
| beta3 | 0 | 0.049804002 |

beta能从0学习不等于性能提升；负值表示该直连分支作为有符号残差参与相加，不能仅凭符号推断具体图像语义或错误原因。

## Smoke与公平性核对

smoke和33项CPU/工作流测试全部通过。真实Hiera/BF16/batch12下，beta=0的四级特征、正则及诊断量与旧替换级联逐位相同；第一次反向三个beta均获得有限非零梯度，数次更新后NAF选择器/融合门控/输出投影均有任务梯度。推理、重复eval、完整权重严格重载正常。smoke稳态约0.335秒/步，峰值分配9.268GiB；正式20轮训练及复评9.87分钟。

正式结束后全部核对通过：

- 40条train/valid输入指纹与旧替换版逐条一致。
- 所有共享原层初始张量、CPU RNG一致，仅多3个参数。
- 有效配置除variant外完全一致，完整pip freeze一致，原网络/NAF源文件哈希一致。
- driver增加了新变体、检查和beta日志。移除新增变体的日志分支后，训练函数的AST与旧driver完全一致；StudyDecoder、损失、优化器、抽样上下文和autocast函数的AST也完全一致。
- 最佳IoU完整state_dict严格加载，复评与训练最佳IoU相差小于1e-6；固定最佳IoU与带指标最佳IoU权重档案哈希相同，并另保留最佳Loss权重。

共同配置仍为352、seed1234、BF16、batch12/验证6、2worker/CPU4线程、严格CUDA、paired_aug_v2、原三头Structure、AdamW两组lr0.001、weight_decay0.0005、Cosine20轮至0、NAF正则0.001。数据、官方Hiera、旧模型和历史结果不变，没有调整阈值。

## 产物

运行ID：`local_train_naf_replacement_skip_seed1234_20261028_013534_88998`。

- 新权重：`/home/lien/WorkSpace/PolypSegmentation/checkpoints/local_train_naf_replacement_skip_seed1234_20261028_013534_88998/`
- 完整记录：`/home/lien/WorkSpace/PolypSegmentation/experiments/local_train_naf_replacement_skip_seed1234_20261028_013534_88998/`
- 四组汇总：`/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_cross_scale_skip/summary.json`
- CSV与核对/绘图脚本：同一analysis目录的runs.csv、summarize.py。
- 四模型曲线：`/home/lien/WorkSpace/PolypSegmentation/visualization/local_cross_scale_skip_20261028/four_model_learning_curves.png`
- beta轨迹：同一visualization目录的beta_trajectories.png。

历史0.8786参考门槛仍未复现；此前用户已明确接受记录差距并完成本机同协议消融。本次结果支持在当前预算下保留原Decoder，不支持仅增加这三条直连来提高精度；不能据一个seed排除所有跳连接设计。
