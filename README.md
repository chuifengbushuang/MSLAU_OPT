# P5.1：冻结基线与独立实验差异

研究分支exp/p51-independent，直接基于原实验者最新提交81b19e2。
目标：原始三seed baseline → 单个NAF候选 → 匹配比较和迭代，不追历史峰值。

```text
experiments/
  exp00_baseline/       原始14文件＋run.py/config.json/说明
  exp01_.../           后续创建：config.json、changes.patch、added/、README.md
tools/
  build_experiment.py   基线＋单个实验差异 → 完整源码
  run_experiment.py     临时展开后运行，默认只预览
  organize_legacy_results.py
.work/                 临时源码，结束/失败后清理，不进Git
```

## 实验定义

实验0的训练、网络、损失、loader和split保持原始字节，哈希见source_manifest.json。
新实验使用自己的config.json（id为目录名，base_commit为81b19e2完整提交号）。
对原文件的修改放标准Git文本补丁changes.patch，例如a/networks/mslau_net.py与b/networks/mslau_net.py。
新文件放added/，保持相对路径，例如added/networks/naf.py；不能借added/覆盖原文件。
每个实验直接相对冻结基线，不在运行时引用其他实验；必须改变基线时另建批次。
运行前提交；工作树有未提交改动时统一入口拒绝训练。当前只有实验0，没有新NAF实现。

## 环境和运行

固定Python：/home/le/miniconda3/envs/mslau-p51/bin/python，已核对原/副本运行版本和pip freeze一致。
数据：/data/dataset/zsy/mslau-net/Kvasir-SEG。
初始化：/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors。
原命令设置：352/b12/nw8/e20、Structure、两组LR1e-3、D2/D3均1、boundary0，其余原默认。
不要重新安装原requirements/environment文件，不改原增强或初始化。

仓库根目录仅预览：

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/run_experiment.py exp00_baseline --seed 1234 --gpu 0
```

加--run才训练并独立评估；入口显示并记录所选GPU占用，已有进程、显存占用或利用率不会阻止训练。
GPU占用查询失败也仅提示，是否能使用指定卡由PyTorch的CUDA检查确认。
三seed按1234、42、2026各一次，不挑最高重复拼基线。

## 输出与完整导出

```text
/data/models/le/mslau-net/checkpoints/<run_id>/kvasir_best_model.pth
/data/models/le/mslau-net/experiments/<run_id>/   命令、环境、runtime、日志、manifest
/data/models/le/mslau-net/visualization/<run_id>/ 有图才输出，实验0默认不导出预测
```

保留原脚本的三份权重及原始命名：

- `kvasir_best_model.pth`：固定名称的最佳验证IoU模型，独立评估默认使用它。
- `best_iou_<IoU>_epoch_<epoch>_<Loss>.pth`：同一最佳IoU模型的带指标名称版本。
- `best_loss_<Loss>_epoch_<epoch>_<IoU>.pth`：最佳验证Loss模型，可能对应不同epoch。

不再在训练结束后删除任何一份。
manifest保存研究提交、基线、补丁/新增文件/源码哈希、配置和reproduce_from_repository重跑命令。
临时源码会删除，重跑使用仓库入口，而非记录中的临时路径。

按需导出，目标目录必须不存在：

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/build_experiment.py exp00_baseline --output releases/exp00_baseline
```

导出目录可移到仓库之外，使用所需环境的python run.py --seed 1234 --gpu 0 --run独立运行，无需Git或补丁工具。
输出根目录可用P51_OUTPUT_ROOT指定，数据和初始化仍按项目路径。导出不包含数据/权重。
展开仅发生在启动前，batch循环不应用补丁或动态分派模型；有一次源码构建/校验开销，未测启动耗时。

## 迁移清理旧输出

默认只显示计划：

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/organize_legacy_results.py
```

添加--apply才执行已授权的迁移和删除，固定范围见tools/legacy_results_plan.json。

- 保留旧mslau-naf的baseline、四级NAF、P3/P4 NAF各三seed，9次运行的三份原始命名权重及必要记录标history_*。
- CPU检查state_dict，复制SHA256一致且全部迁移成功后才清理旧目录。
- 其他环境重复、微调、旁路、冒烟和冗余权重留小型历史汇总，再删除清单中的旧输出。
- 新文件/大小变化/软链接/进行中的Kvasir任务会阻止操作，不扩大删除范围。
- 不删代码clone、Git、数据、环境或原作者成果。真实模型迁移只在服务器执行--apply时发生。
- 可重复执行，已迁移run会复核权重哈希；首次迁移三份权重需临时空闲空间约21.78GiB。

旧匹配均值：baseline0.869595、四级NAF0.862984、P3/P4 NAF0.867907；不能与新环境混作匹配对照。
原环境最好观察值0.875919，原作者历史最好0.878616；旧NAF源码保留在Git历史/归档tag中。

历史整理和命名修复工具只手动使用，不参与训练流程。
基础检查：python -m unittest discover -s tests -v，不导入模型、不用GPU、不训练。

已执行旧版单权重迁移的结果，使用tools/restore_checkpoint_names.py预览，加--apply才修复命名。
它恢复固定名称和带指标的IoU名称（新建IoU副本使用硬链接，不额外复制大文件）；
最佳Loss只从仍存在的真实原文件恢复。已经删除且没有其他副本的最佳Loss无法还原，manifest会明确记录缺失。
该工具不训练、不删除旧输出，不要为修复命名重新执行迁移清理。
