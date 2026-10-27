# 2026-10-28 本机三模型比较结果

在固定seed1234、相同20轮训练预算下，NAF替换原Decoder的三个UpBlock没有改善结果。原始无NAF基线最好；NAF+Decoder较基线下降0.0875个百分点，NAF替换Decoder较基线下降0.3893个百分点、较NAF+Decoder下降0.3019个百分点。

| 模型 | 最佳权重复评IoU | Dice | 最佳轮次（从1计） | 训练及复评分钟 | 可训练参数 |
|---|---:|---:|---:|---:|---:|
| 原始基线，无NAF | 0.865559723 | 0.918729888 | 19 | 8.64 | 4,269,747 |
| 离散级联NAF + 原Decoder | 0.864684848 | 0.917038641 | 16 | 10.32 | 4,325,865 |
| 同一NAF替换三个UpBlock，保留RFB/三头 | 0.861666300 | 0.916309873 | 20 | 9.81 | 3,993,321 |

三组正式训练共28.78分钟，不含依赖安装、smoke和额外FP32诊断。替换版比NAF+Decoder减少332,544个可训练参数（7.69%），但本次IoU降低；完整训练用时仅少约0.51分钟。不能仅凭相似的渐进融合拓扑认定三个UpBlock多余。替换版最佳值出现在最后一轮，当前结论只针对相同20轮预算，不能排除更长训练或其他设计的可能性。

## 资源与共同配置

- 当前机器：RTX5070Ti 16GB、Ryzen9600X 6核12线程、约16GB主机内存；当前PyTorch2.13.0+cu130/torchvision0.28，保留现有CUDA环境。
- 独立依赖环境：`/home/lien/WorkSpace/PolypSegmentation/work/local_train_env`。timm1.0.15、Albumentations1.4.24、NumPy1.26.4等固定，完整pip freeze保存在每个运行目录。
- 352×352、batch12、验证batch6、2个数据worker、4个PyTorch计算线程、interop1，OpenCV额外线程关闭，OpenBLAS1。
- BF16混合精度；NAF内部FFT/选择/融合保持其原实现的FP32计算，概率与IoU阈值计算FP32。三组精度一致。严格CUDA算法、cuDNN deterministic、关闭benchmark。
- 原增强操作/概率、原mask>0口径，paired_aug_v2逐样本/epoch控制，训练880张（drop_last后每轮876）、验证120张。val/test划分镜像，不是独立测试集。
- Structure主+D2+D3损失，各权重1；AdamW两组lr0.001、weight_decay0.0005；20轮Cosine至0。仅NAF两组训练时加原六路评分正则0.001，验证不加。
- 所有模型均从同一官方Hiera文件重新初始化，不沿用smoke更新、训练好的baseline或其他实验权重。

## Smoke与核对

先测FP32小batch4，再测FP32 batch12、BF16 batch12以及CPU线程/worker组合。选定配置下，baseline中位训练步约0.275秒、NAF+Decoder约0.360秒、NAF替换约0.332秒；峰值PyTorch分配显存约8.94/9.29/9.16GiB。FP32 batch12峰值分配13.98GiB、预留14.93GiB；调整至4线程/2worker后约0.643秒一步。BF16提供显存余量并保留原batch12。

全部三个完整Hiera真实数据smoke通过有限非零梯度、正常优化更新、输出形状、重复eval逐位相同及训练参数序列化检查。专项CPU测试4项和既有工作流25项通过；专项测试额外覆盖纯任务损失传到三个NAF选择器/融合门控/四级输入、删除UpBlock、三头读取实际F1/F2/F3及完整state_dict严格重载。

完整正式训练后核对全部通过：

- 三组所有保留原层的初始张量和CPU RNG完全一致。
- 三组20轮train/valid共40条输入指纹（样本顺序、图像、mask、batch边界）完全一致。
- 三组driver、原模型源文件、NAF源文件哈希及共同有效配置一致。
- 三组最佳完整模型权重均严格重载，复评IoU与训练记录误差小于1e-6；各保留原始三份权重。

## 历史参考差距

当前baseline低于0.8786参考值1.3040个百分点，未满足最初±0.005要求。核查发现本次完整初始模型SHA256为`8b679cb55de63ac980c3342e777496dbd5e0a2523fb40ff00cacb4cfa600319d`，与历史严格v2 seed1234一致；全部40条历史输入指纹亦一致。那次同协议历史基线IoU为0.868227，不能把0.8786当成同协议已验证结果。

用户随后明确授权“记录差距，继续三个模型同环境消融”。额外FP32定位运行已中止、保留部分记录，明确排除正式三组比较；没有完整FP32对照，不能把差距归因于BF16、GPU或单一算子。

## 产物

正式运行目录：

```text
baseline       local_train_baseline_seed1234_20261028_004839_75754
NAF+Decoder    local_train_naf_decoder_seed1234_20261028_010212_79638
NAF替换        local_train_naf_replacement_seed1234_20261028_011314_82790
```

每个run_id对应项目父目录下的`experiments/<run_id>/`和`checkpoints/<run_id>/`。记录包含有效配置、完整环境、源码及driver快照、40条输入指纹、逐轮指标、梯度审计、复评结果、console.log和结果文件；权重包含固定最佳IoU、带指标最佳IoU、最佳Loss三份。

- 汇总和核对：`/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_local_decoder_study/summary.json`
- 表格：`/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_local_decoder_study/runs.csv`
- 图：`/home/lien/WorkSpace/PolypSegmentation/visualization/local_decoder_study_20261028/learning_curves.png`
- 可重新核对/绘图脚本：`/home/lien/WorkSpace/PolypSegmentation/analysis/20261028_local_decoder_study/summarize.py`

这是一个seed、一个验证划分的探索性比较。它支持本次预算下保留原Decoder，不证明跨seed稳定性或独立测试集上的优劣；原基线和旧实验定义均未修改。
