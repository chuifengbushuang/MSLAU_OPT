# P5.1：冻结基线与独立实验差异

研究分支exp/p51-independent，直接基于原实验者最新提交81b19e2。
目标：原始三seed baseline → 单个NAF候选 → 匹配比较和迭代，不追历史峰值。

历史候选实验9 `exp09_naf_discrete`：保留实验7三级级联，训练按样本抽取90%共轭频率对、评估确定性Top-K，去掉内部alpha，对评分V施加分段平方L2正则，固定六项均值归约。目前先修复输入协议，不继续安排旧实验9训练。
实验9接入已有paired_aug_v1输入协议，对照应使用`exp00_baseline_paired_aug_v1`；旧实验0–8保持原样，旧NAF结果只作参考。
实现边界、合成检查和正式三seed命令见[实验9说明](experiments/exp09_naf_discrete/README.md)。本地检查不替代3090完整Hiera检查，不代表已有训练收益。

当前暂停扩展模型实验，先检查配对输入基线的CUDA数值可重复性。服务器已确认两个上采样用例在普通模式下梯度有微小差异，在严格模式下5次逐位相同；尚不能据此解释完整训练的IoU波动。
完整基线短步对照及严格v1的seed1234两次完整训练已完成：两次20轮状态/指标/最佳权重一致，最佳IoU均0.853762。
2026-10-27发现逐样本Compose.set_random_seed使子变换同起点，引入增强联动；此前仅检查重复性，遗漏联合分布。当前应使用独立`exp00_baseline_paired_strict_v2`的共享推进随机流，保持采样顺序、原训练配方和严格CUDA执行不变。修复依据、CPU检查与单次训练命令见[v2说明](experiments/exp00_baseline_paired_strict_v2/README.md)；旧定义/结果保留，暂缓尚未启动的v1补训。代理不代启动训练。
完整基线短步重放使用独立工具，不修改任何实验定义：

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python -u tools/check_baseline_numerics.py --gpu 0 --seed 1234 --mode strict --steps 3
```

工具在两个全新进程中各执行epoch 0前三个真实batch的诊断更新，保留352/b12/nw8、原AdamW/Structure/辅助监督/学习率；只启用严格确定性及CuBLAS配置，保留TF32与注意力后端现状。直接使用原增强/损失定义，避开训练脚本中无关的torchmetrics/Matplotlib导入。它不会运行完整epoch、验证或保存权重，临时展开源码结束即删除。
小型JSON和日志保存在`/data/models/le/mslau-net/experiments/p51_numeric_replay_strict_seed1234_…/`，含初始完整参数与CPU/CUDA随机状态、逐batch输入、前向输出/损失、每个参数的梯度和更新后参数/BN缓冲区哈希，以及首次差异路径。`summary.json`的`matched`只代表观察到的三个更新一致，不承诺整次训练一致。失败记录保留具体异常，不能把严格模式报错改成警告后视为通过。
需要排查普通模式时可显式设置`--mode default`；当前只要求先运行严格模式，不启动新一批正式训练。

```text
experiments/
  exp00_baseline/       原始14文件＋run.py/config.json/说明
  exp00_baseline_paired_aug_v1/ 原增强不变，独立控制输入随机性和记录
  exp00_baseline_paired_strict_v1/ 同一输入算法＋严格CUDA执行，完整重复验证
  exp00_baseline_paired_strict_v2/ 修复增强子流联动，采样顺序和严格CUDA协议不变
  exp01_naf_all/        四层完整模型配置
  exp02_naf_p34/        仅P3/P4配置
  exp03_naf_p12/        仅P1/P2配置
  exp04_naf_all_no_reg/ 四层NAF去正则对照
  exp05_naf_p12_no_reg/ P1/P2 NAF去正则对照
  exp07_naf_cascade/    三级双输入NAF级联，独立补丁和模型文件
  exp08_cascade_fusion/ 实验7的仅级联融合对照，独立源码，频率及正则关闭
  exp09_naf_discrete/   三级级联＋离散频率选择＋评分V正则，输入采用paired_aug_v1
  shared_naf_v1/        NAF实验共用的changes.patch、added/和方法说明
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
仅接入位置不同的实验用source_definition=shared_naf_v1复用同一份冻结差异源码。构建时复制实际代码，导出后独立，不引用别的实验目录。
运行前提交；工作树有未提交改动时统一入口拒绝训练。当前实验0为原始基线，实验1四层，实验2仅P3/P4，实验3仅P1/P2。方法和归约见experiments/shared_naf_v1/README.md。
实验1–3固定R=sum(启用层R_n)/4，lambda=0.001；实验4/5仅将lambda改为0。
实验7为三级双输入级联，六个分支固定求和除以6，lambda=0.001。旧exp01_naf_select_reg是分母2的历史版本，不混入新协议。
实验8保留实验7的投影、GAP+RMS门控、残差和级联，频率处理旁路且无正则；共享层初始参数逐元素一致。

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

服务器更新代码后，先执行新版完整合成检查（三种接入配置）：

```bash
CUDA_VISIBLE_DEVICES=0 NAF_FULL_SMOKE=1 NAF_DEVICE=cuda /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp01_naf.py' -v
```

首轮实验0–3的三个种子依次执行（本批已完成），任一次失败即停止；GPU按实际选择修改：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    P51_GPU=0
    for P51_EXPERIMENT in exp00_baseline exp01_naf_all exp02_naf_p34 exp03_naf_p12; do
        for P51_SEED in 1234 42 2026; do
            /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py "$P51_EXPERIMENT" --seed "$P51_SEED" --gpu "$P51_GPU" --run
        done
    done
)
```

共12次正式训练/独立评估；若实验0三seed已完成，可从列表移除exp00_baseline，复用现有结果。旧NAF结果不能替代新协议的实验2。


## 实验0–5已完成

2026-10-20首轮12次已完成，报告见reports/20261020_naf_v1/README.md。当前匹配IoU均值：baseline0.864155、四层0.866403、P3/P4 0.871437、P1/P2 0.873464。
四层3/3seed改善；P1/P2均值最高但2/3改善，尚未确认稳定增益。实验4/5也已完成，不需要重复执行下方历史批量命令。
exp04_naf_all_no_reg对照实验1，exp05_naf_p12_no_reg对照实验3，只将lambda改为0。共享模型源码/补丁完全相同，无需重复完整模型GPU检查。
实验4均值0.867181，实验5均值0.863899：四层去正则小幅提高0.0778百分点，P1/P2去正则三个seed均下降、均值下降0.9565百分点。新旧批分别使用GPU0/2、增强轨迹未固定，不能据此宣布确定因果或稳定增益。

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_EXPERIMENT in exp04_naf_all_no_reg exp05_naf_p12_no_reg; do
        for P51_SEED in 1234 42 2026; do
            /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py "$P51_EXPERIMENT" --seed "$P51_SEED" --gpu 0 --run
        done
    done
)
```

## 实验7完整模型（已完成）

三个独立双输入NAF，先分别滤波再门控融合；无独立NAF4，保留原Decoder和训练配置。
F3=NAF3(P3,P4)，F2=NAF2(P2,F3)，F1=NAF1(P1,F2)。F3/F2/F1同时进入原Decoder，P4原样进入最深解码路径。
双路选择器使用联合频谱描述；融合使用GAP+centered RMS；输出投影零初始化，初始输出与基线一致。新增56,124参数。
具体流程、论文借鉴边界及正则归约见experiments/exp07_naf_cascade/README.md；旧实验保持冻结。仅融合对照现为实验8，实验7单独去正则对照暂不新增。
三seed IoU为0.871073/0.876448/0.871200，均值0.872907±0.003067；较当前基线+0.8752百分点、3/3提高，较实验3低0.0557百分点、1/3提高。三seed探索结果不证明稳定增益或边缘改善；已有实验7无需重复训练。

服务器拉取后，先完成实验7合成检查，包含官方Hiera和正式batch12的一次前向/反向，不执行正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 EXP07_FULL_SMOKE=1 EXP07_DEVICE=cuda EXP07_SMOKE_BATCH=12 /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp07_naf.py' -v
```

合成检查通过后，三个种子顺序运行，任一次失败即停止：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_SEED in 1234 42 2026; do
        /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp07_naf_cascade --seed "$P51_SEED" --gpu 0 --run
    done
)
```

## 下一步：实验8仅级联融合

独立定义exp08_cascade_fusion。投影与融合共享层初始参数及全局CPU随机状态与实验7一致，新增可训练参数55,728；删除396个频率选择器/alpha参数，运行时没有FFT/IFFT或频率正则。
配置相对实验7仅改变id和naf_reg_weight=0；实际损失直接沿用原训练损失，不添加零值占位正则。方法及检查见experiments/exp08_cascade_fusion/README.md。

先执行服务器合成检查，包含官方Hiera、正式batch12的前向/反向及峰值显存，不执行正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 EXP08_FULL_SMOKE=1 EXP08_DEVICE=cuda EXP08_SMOKE_BATCH=12 /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp08_fusion.py' -v
```

通过后，三个种子依次训练/评估，失败即停止：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_SEED in 1234 42 2026; do
        /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp08_cascade_fusion --seed "$P51_SEED" --gpu 0 --run
    done
)
```

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
基础检查：python -m unittest discover -s tests -v，包含CPU数值和展开模型的合成检查，不执行正式训练。
完整Hiera检查只在显式设置NAF_FULL_SMOKE、EXP07_FULL_SMOKE或EXP08_FULL_SMOKE时启用；需要有torch的环境才能执行模型数值检查。

已执行旧版单权重迁移的结果，使用tools/restore_checkpoint_names.py预览，加--apply才修复命名。
它恢复固定名称和带指标的IoU名称（新建IoU副本使用硬链接，不额外复制大文件）；
最佳Loss只从仍存在的真实原文件恢复。已经删除且没有其他副本的最佳Loss无法还原，manifest会明确记录缺失。
该工具不训练、不删除旧输出，不要为修复命名重新执行迁移清理。
