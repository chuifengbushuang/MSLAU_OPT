# 实验输出、保留规则与当前研究优先级

2026-10-19 用户确认：新 NAF 在同环境、同训练协议下稳定优于 baseline 为第一目标；历史
0.878616 的精确复现为第二目标，不作为 NAF 工作的前置条件。不承诺任何未经实验的方法必然提升。
原实验者最终接收同一套 baseline/NAF 开关代码、配置与环境说明，在其环境重新对照。

## 后续唯一推荐入口

在仓库根目录使用 `python -m tools.run_experiment`，实际训练/评估仍调用原有 Kvasir 脚本。
该入口不更改模型、损失、增强、seed或权重保存语义；直接调用原脚本的旧CLI仍兼容，但不自动采用新布局。

```text
/data/models/le/mslau-net/
  checkpoints/<run_id>/               原训练入口生成的模型权重
  experiments/<run_id>/
    manifest.json                    状态、命令、三类绝对路径、结果、权重哈希
    command.txt                      可读启动命令
    commit.txt / status.txt / code_diff.patch
    source.zip                       已跟踪和未忽略的未跟踪源码快照，不含权重
    environment.txt / runtime.txt    pip freeze及PyTorch运行时
    gpu_before.txt / selected_gpu.txt
    input_hashes.json                显式传入的split/预训练/初始化/评估权重哈希
    console.log
    logs/kvasir_train.log
    logs/parsed_args.json            当前分支训练入口解析出的完整参数，含默认值
  experiments/index.jsonl             加锁追加的全局索引；各manifest为权威记录
  visualization/<run_id>/             评估debug图像；曲线等后续按需写入
```

run_id自动包含方案名、时间和随机后缀；`--name`应含方案、seed、重复编号。
不允许在透传参数里再指定`--output_dir/--log_dir/--debug_dir`。
训练与独立评估分别有自己的run_id；评估时用`--parent-run`链接训练run_id，避免覆盖日志。
`--repo`可指向独立原分支clone，`--python`可指定解释器。原分支本身不会生成新增的parsed_args.json；其默认参数可由source.zip核查。
CPU仅供工具检查；正式训练显式单卡`--gpu 0`等。入口检查选用GPU计算进程、显存和利用率；
这只是启动快照，不是GPU预约服务，不保证运行全程无人使用。启动前仍应现场检查。
入口不自动生成全部预测或曲线；评估`--debug true`才输出预测。

### 仅预览、不启动的完整P5.1示例

```bash
python -m tools.run_experiment \
  --name p51_none_seed1234_r1 --gpu 0 --dry-run -- \
  --dataset /data/dataset/zsy/mslau-net/Kvasir-SEG \
  --train_txt configs/splits/kvasir/train.txt \
  --val_txt configs/splits/kvasir/val.txt \
  --encoder_name sam2_hiera_large --decoder_mode p5_sam2unet \
  --hiera_checkpoint /data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors \
  --loss structure --img_size 352 --batch 12 --num_workers 8 \
  --epoch 20 --seed 1234 --encoder_lr 1e-3 --decoder_lr 1e-3 \
  --aux_d2_weight 1.0 --aux_d3_weight 1.0 --boundary_weight 0.0 \
  --disable_edge_guidance --fusion_mode fixed --decoder_dropout 0.0 \
  --frequency_mode none
```

`--dry-run`只打印计划，不创建目录、不导入模型、不检查GPU、不启动训练。
当前不需要删除该选项去补跑baseline；先完成新NAF设计与工程验证。

### 独立评估布局示例（先替换训练ID）

```bash
TRAIN_RUN_ID='替换为训练manifest中的run_id'
python -m tools.run_experiment \
  --name p51_none_seed1234_r1_eval --kind evaluate --gpu 0 \
  --parent-run "$TRAIN_RUN_ID" --dry-run -- \
  --dataset /data/dataset/zsy/mslau-net/Kvasir-SEG \
  --split_txt configs/splits/kvasir/val.txt \
  --model "/data/models/le/mslau-net/checkpoints/$TRAIN_RUN_ID/kvasir_best_model.pth" \
  --encoder_name sam2_hiera_large --decoder_mode p5_sam2unet \
  --img_size 352 --batch 8 --num_workers 0 \
  --disable_edge_guidance --fusion_mode fixed --decoder_dropout 0.0 \
  --frequency_mode none --debug false
```

评估NAF必须传与训练相同的全部模块开关；上述none示例不能用于NAF权重。

## 保留与清理

- 所有成功、失败和负向实验的命令、完整配置、环境、日志、指标、输入哈希和源码快照保留。
- 历史0.878616、原分支0.874656及被用作微调起点的权重保护；不清理其他账号的模型或预训练输入。
- 匹配baseline、关键消融、最终方案保留各seed/重复的最佳IoU权重，不只挑最高一次。
- `kvasir_best_model.pth`为当前主引用；`best_iou_*.pth`可能保存同一模型，逐tensor核对后才考虑去重。
- `best_loss_*.pth`仅在同模型内容核查、评估用途与引用审计后决定；不能仅凭epoch相同就删除。
- 冒烟权重、过期PID、可重建的大特征缓存及重复全量预测是人工审阅候选；不提供自动删除命令。
- 当前state_dict不是完整续训包；没有自动保存optimizer/scheduler/RNG。此次不改变保存策略，避免混入训练对照。
- 本地同步目录不代表服务器完整内容：已发现一个0字节pth占位文件，不是可加载模型。
  本地容量统计不是服务器可释放容量。盘点工具不加载模型，也不跟随目录软链接，不推断tensor重复。

本地实际清单：项目父目录`analysis/output_retention_20261019/`。
`retention_plan.csv`按现有目录列建议；`local_inventory_v2/files.csv`逐文件盘点。
同一份目录保留清单已纳入仓库：[output_retention_plan.csv](output_retention_plan.csv)。

### 服务器只读盘点（仅新建报告）

```bash
AUDIT_DIR="/data/models/le/mslau-net/experiments/output_inventory_$(date +%Y%m%d_%H%M%S)_$$"
python -m tools.inventory_outputs \
  --root /home/le/WorkSpace/20261012_PolypSegmentation \
  --root /data/models/le/mslau-net/checkpoints \
  --root /data/models/le/mslau-net/experiments \
  --root /data/models/le/mslau-net/visualization \
  --output-dir "$AUDIT_DIR"
printf '盘点报告：%s\n' "$AUDIT_DIR"
```

扫描旧项目父目录时排除代码clone、环境、论文和数据目录。CSV含逻辑字节数，不等于去重后的可释放物理容量。
原实验者历史权重位于zsy目录，不在这条命令的清理范围。请同步报告，不必同步大权重。

## NAF主线的后续实验纪律

1. 固定当前mslau-naf环境；历史峰值复现降为辅助工作，不继续为追峰值无限重复或重装环境。
2. 新NAF仍在RFB后逐级独立处理，不增加跨尺度融合。先完成“频率竞争/软选择”和“分段正则”
   的明确数学定义、二维共轭一致性、初始化与梯度验证，再实现独立新模式；保留旧naf2d及none兼容。
3. 初筛最小组为B：none；S：新频率选择、reg=0；SR：同S+分段正则。
   架构位置、stage、训练参数保持固定，不同时搜索多种插入位置/alpha/LR。若SR有信号，再补
   SU：同S+总权重匹配的均匀正则，区分“正则有用”与“分段分配有用”。正则定义和尺度先校准再定lambda，不能照抄原文数值。
4. seed1234仅筛查工程/负向信号，不能证明有效；候选锁定后在42/1234/2026下重跑匹配B与候选，
   报告每seed差值、均值及样本标准差。已有旧驱动/旧代码baseline作背景，不替代新的匹配对照。
5. 预先约定三seed均值提升且至少2/3 seed提升仅为初步支持；欲称稳定优于baseline，还需额外
   未参与选型的种子/重复确认，并报告失败例、Dice/Precision/Recall、面积分组和开销。
   如收益小于重复波动，增加独立确认而非挑最佳一次；小样本不宣称统计显著或保证跨环境收益。
6. 模块开关导致初始化RNG消耗不同、增强未独立固定等应记录并尽量隔离，降低无关噪声；
   不要求逐batch追溯历史随机轨迹。任何新复现控制同时用于B和候选，不混用旧基线结果。
7. 最终交付同一commit的B/NAF启动配置、模块单元/集成验证、每次完整实验manifest、环境快照和
   结果表，由原实验者在其环境重跑匹配对照；本环境正收益不能预先保证另一环境同幅度提升。

本轮不实现新NAF、不自动训练；下一步先锁定新选择器/正则设计，再完成工程验证。
