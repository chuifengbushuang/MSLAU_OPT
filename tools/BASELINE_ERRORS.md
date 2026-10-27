# v2基线错误分析（仅推理）

目的：先确认基线的困难样本和误差位置，再判断频域方法是否针对真实缺陷。
不修改实验定义、模型、损失、增强或训练配置，不调用训练入口，不保存/覆盖权重。

## 输入

- 已完成的strict-v2基线seed1234、42、2026，每seed只取预先选定的r1。
- `checkpoints/<run_id>/kvasir_best_model.pth`及`experiments/<run_id>/manifest.json`。
- 原数据`/data/dataset/zsy/mslau-net/Kvasir-SEG/{images,masks}`及冻结的val.txt（120张）。
- 权重是完整state_dict，严格加载全部参数，因而无需另外加载官方SAM初始化文件或联网下载。
- 权重和数据留在服务器，不必同步大文件。本次不需要NAF权重或1234重复权重。

## 执行

当前已在服务器仓库中，无需重新激活环境。默认沿用本批物理GPU3，可按空闲情况修改`--gpu`。

```bash
git pull --ff-only origin exp/p51-independent
```

```bash
PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 \
/home/le/miniconda3/envs/mslau-p51/bin/python -u tools/analyze_baseline_errors.py \
  --gpu 3 \
  --output-root /data/models/le/mslau-net \
  --runs \
  exp00_baseline_paired_strict_v2_seed1234_20261027_112812_507785_r1 \
  exp00_baseline_paired_strict_v2_seed42_20261027_151645_765712_r1 \
  exp00_baseline_paired_strict_v2_seed2026_20261027_154836_826819_r1
```

顺序对3个权重各推理120张，不反向、不更新参数。自动展开临时源码，结束后清理临时目录。
使用原版`kvasir_test.py`的loader、预处理和指标函数：352、batch8、workers0、阈值0.5、FP32、无TTA或后处理；保持strict_cuda_v1设置。

先核对三份权重的文件SHA256、已完成的v2协议、网络/loader/评估/严格运行时源码及split哈希。
推理后对照原记录的IoU和Dice（原记录保留4位小数，容差0.000051）。不匹配则标记failed并停止，不输出成功结论。
同时记录原图/mask文件SHA256和逐图输入哈希，三个seed的评估输入必须完全一致。
新分析目录默认带时间和进程号；已有目录拒绝覆盖。

## 输出

同一个`baseline_errors_v2_<时间>_<进程号>`下：

```text
/data/models/le/mslau-net/experiments/<analysis_id>/
  summary.json                    状态、来源、版本、逐seed/逐尺寸分组统计
  dataset_files.csv               240个原图/mask文件的哈希
  size_groups.csv                 每seed每尺寸组的样本量、IoU、Recall、低召回数
  cross_seed_review.csv           按平均IoU排序的120个病例及人工复核空白栏
  <run_id>/per_image.csv          逐图指标、误差分解、输入哈希、图像相对路径
  <run_id>/review_candidates.csv  各类候选图片的排名
/data/models/le/mslau-net/visualization/<analysis_id>/<run_id>/
  cases/*.jpg                    每张图的四面板（原图轮廓、概率、GT、错误叠加）
  worst_iou.jpg                   最差IoU候选
  small_low_recall.jpg            小面积组低召回候选（有样本才生成）
  far_background_fp.jpg           远离GT边界的FP候选（有错误才生成）
  boundary_error.jpg              边界F1较低候选（有错误才生成）
```

同步`experiments/<analysis_id>`和`visualization/<analysis_id>`即可；不新增checkpoints文件。
图像颜色：GT轮廓青色、预测轮廓紫色；漏分FN红色、误分FP黄色。预测使用float32概率计算，热图仅用于显示。

## 指标解释与边界

- 面积组固定为GT占352×352网格的`<1% / 1–5% / 5–20% / >=20%`，诊断中的“小面积”为前两组。
  这是图像内相对面积，不是临床直径，也不宣称与历史图的分组定义相同。每组同时报告样本量。
- 低召回候选：整张GT的Recall<0.5；另统计8连通GT区域中Recall<0.5的个数。
  连通区域不等于临床独立息肉，Recall<0.5不是经过验证的临床漏诊阈值。
- GT边界为mask减去3×3腐蚀，图像外部视为背景。用欧氏距离定义边界带，分别报告2/4/8像素结果。
  对每个尺度，总FP+FN恰好分为：边界带FP+FN、带外内部FN、带外背景FP。
  报告按所有错误像素汇总的比例及逐图比例均值，避免只按大息肉像素数下结论。
- 边界F1：预测轮廓中距离GT轮廓不超过容差的比例，与反方向比例的调和均值。
  边界距离均值/95分位数为双向轮廓距离合并后的统计；不是所有论文的HD95实现约定。
  单侧或双侧轮廓为空时距离为null；单侧为空时边界F1=0，双侧为空时为1。
- 另统计与GT无重叠的预测连通域。带外FP不一定是独立假目标；也可能连接真实预测的过分割部分。
- 这些候选排序并不互斥；例如小息肉漏检可能主要表现为边界带错误。不能把三个候选类别当作互斥病因。
- 远离边界的FP只说明误分位置。是否由皱襞/纹理、反光、低对比、标注疑点等导致，必须看原图复核。
  `cross_seed_review.csv`中的`suspected_cause`和`review_notes`保持空白，不自动生成病因标签。
- 使用的是选最佳权重所用的验证集，且冻结的test.txt与val.txt相同；结果是开发诊断，不是独立测试结论。
  验证图像的背景区域可以分析像素误分，但不能替代独立正常/无息肉图像上的假阳性评估。
- 即使发现大量边界错误，也不直接证明频域方法适合；后续仍须验证频率选择改善了哪些病例、是否引入新错误，
  并通过同协议仅融合对照区分融合和滤波的贡献。本工具不为此启动新训练。

## 本地验证

`python -m unittest discover -s tests -p 'test_baseline_errors.py' -v`

覆盖错误区域分解、空预测、触边目标、面积边界、聚合分母、错误权重/源码/协议拒绝、跨seed输入拒绝，
以及原始预处理/指标下小网络的真实CPU推理和图像导出（参数和梯度不变）。
本地不具备服务器的完整训练权重和真实验证数据；实际Hiera复评一致性由服务器命令检查。
