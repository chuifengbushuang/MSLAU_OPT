# 基线输入可重复性：paired_aug_v1

只验证原P5.1基线的输入与重复运行，不接入NAF，不自动启动训练。
基础提交81b19e2c801efcf74826b8127f5ea712b0247ec5；原14文件和旧实验目录保持不变。
这是原基线加输入控制补丁，不是源码逐字节不变的历史复现，也不以达到0.8786为成功标准。

## 源码与训练设置

- `changes.patch`只修改展开后的`kvasir_train.py`及记录启动器`run.py`。
- `added/paired_inputs.py`复用原`loader.binary_class`，原图读取、二值mask转换不改。
- 原`get_train_transform/get_valid_transform`函数逐字保留：Resize、H/VFlip(p=.25)、ShiftScaleRotate(shift_limit=0,p=.25)、CoarseDropout、Normalize、ToTensorV2。没有关闭增强或更换插值配置。
- 网络、损失、原评估脚本、数据划分完全未改。config与实验0只有id和顶层input_protocol不同。
- 352/b12/nw8/e20、原Structure主/辅助损失、两组LR1e-3、AdamW(weight_decay5e-4)、原Cosine、warmup0、D2/D3各1、boundary0、官方Hiera初始化。Hiera骨干冻结，Adapter32、RFB64、原Decoder，无NAF。不是checkpoint微调。
- 正式使用原作者当前解释器`/home/zsy/miniconda3/envs/mslau-net/bin/python`；不更新或写入该环境。硬件为单卡3090，默认GPU0；占用只提示。

## 随机性协议

每个样本的增强种子为SHA256派生的非负63位整数，输入是协议版本、主seed、augmentation流标签、phase、epoch、样本ID。不包含实验编号、worker编号或重复标签。每次调用原读取器前设置Compose.set_random_seed；图像与mask仍由原代码联合增强。

训练采样器用独立torch.Generator，由主seed、sample_order流标签、epoch生成randperm。DataLoader工作进程使用另一独立generator；验证使用SequentialSampler。不受模型的随机数消耗影响。

每轮创建DataLoader迭代器前设置epoch；固定persistent_workers=False，与原默认一致。保留8个worker。不同epoch/seed产生不同增强，完全相同seed/epoch/样本应得到相同输入。

保留原cudnn.deterministic=True、benchmark=False。启动训练子进程时显式设置PYTHONHASHSEED，但种子派生不依赖Python hash。未额外开启torch.use_deterministic_algorithms，也未改变原CUDA算子。输入可重复不能保证GPU训练逐位相同；初始参数与输入均相同但训练仍分歧时，才进一步定位CUDA数值非确定性。

## 服务器执行

在研究仓库目录执行，不需要激活环境：

```bash
git switch exp/p51-independent
git pull --ff-only origin exp/p51-independent
```

先只读复核解释器与依赖（不安装包）：

```bash
/home/zsy/miniconda3/envs/mslau-net/bin/python -m pip check
CUDA_VISIBLE_DEVICES="" NO_ALBUMENTATIONS_UPDATE=1 /home/zsy/miniconda3/envs/mslau-net/bin/python -c 'import sys,torch,torchvision,albumentations,numpy; print(sys.executable); print("torch",torch.__version__,"torchvision",torchvision.__version__,"CUDA",torch.version.cuda,"cuDNN",torch.backends.cudnn.version()); print("albumentations",albumentations.__version__,"numpy",numpy.__version__)'
```

真实数据一致性检查：

```bash
CUDA_VISIBLE_DEVICES="" NO_ALBUMENTATIONS_UPDATE=1 /home/zsy/miniconda3/envs/mslau-net/bin/python -u tools/check_paired_inputs.py --seed 1234 --workers 8 --epochs 2
```

检查自动展开源码；不导入网络、不加载权重、不调用GPU、不执行优化步骤。四个独立子进程分别为8-worker、8-worker重复、0-worker以及变更seed；刻意扰动全局RNG。比较完整两轮train/valid的顺序/图像/mask哈希，并对固定样本检查epoch/seed确实改变增强。输出PASS后将带源码哈希、输入指纹和版本的小型JSON写入`/data/models/le/mslau-net/experiments/paired_aug_v1_input_check_<时间>_<PID>.json`。失败返回非零；不会自动启动训练。数据检查不能替代实际模型初始化/GPU数值重复检查。

只有检查通过后，独立训练两次seed1234；每次20轮，随后自动原脚本评估。任一次失败停止，不覆盖输出：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_REPEAT in 1 2; do
        P51_REPEAT="$P51_REPEAT" /home/zsy/miniconda3/envs/mslau-net/bin/python -u tools/run_experiment.py exp00_baseline_paired_aug_v1 --seed 1234 --gpu 0 --run
    done
)
```

P51_REPEAT仅标记run_id及manifest，不进入随机种子。去掉--run则只预览原训练参数及实际完整命令；源码临时展开到研究仓库`.work/exp00_baseline_paired_aug_v1_*/source/`，结束或失败后清理，只保留定义及记录，不永久复制整份代码。

## 输出与判定

沿用三个输出根目录，同一run_id：

- checkpoints：原命名三份权重，kvasir_best_model.pth、best_iou_*、best_loss_*。
- experiments：原manifest、命令、环境、GPU快照、训练/评估日志；额外在logs保存initial_state.json、input_fingerprints.jsonl、epoch_metrics.jsonl。
- visualization：仅实际出图时创建；原debug=false。

initial_state.json按排序后的state_dict名称、dtype、shape及张量内容生成SHA256，并记录CPU RNG、参数量、cuDNN设置。不是直接比较torch.save文件字节。

input_fingerprints.jsonl每轮train/valid各一行，共40行；包括样本/批次数与顺序、图像和mask汇总哈希。哈希包含batch边界，训练880样本中按原drop_last实际使用876个；验证使用120个、batch6。

epoch_metrics.jsonl额外记录未四舍五入的各轮loss/main_loss/IoU，避免原四位小数日志掩盖细微分歧。记录不改变训练计算，但CPU端种子设置和哈希有额外耗时，未作服务器性能基准。

两次运行首先要求initial_state.state_sha256一致、全部输入记录一致；再比较高精度指标轨迹及最佳结果。若参数/输入不同先修管线；若只有指标不同则报告剩余数值波动并定位，不再归因于未固定增强。固定输入可能让结果上升或下降，不能直接与旧协议结果混合统计，也不代表所有随机性已消失。

本轮不改实验7/8，也未验证它们已经采用此协议；未来接入后需要同样核对共享初值和跨模型输入。

## 本地完成的检查

完整unittest共72项，69通过、3项既有官方Hiera/GPU检查跳过；本实验新增11项全部通过。使用本地CPU PyTorch2.13和真实Albumentations1.4.24、NumPy1.26.4、skimage0.25.2；包含8个worker、独立进程、原训练循环的小型合成用例和三权重/运行记录。测试依赖放在本地/tmp，没有改服务器环境。14份冻结文件与原提交逐字节一致，旧实验定义未改，展开源码语法与预览通过。

没有访问服务器真实数据、加载官方权重或启动3090正式训练；上面的真实数据检查仍需用户在服务器执行。数据哈希跨不同库版本并不保证相同，因此正式比较必须使用同一环境。
