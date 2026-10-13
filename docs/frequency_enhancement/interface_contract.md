# P5.1 / AFFE 接口契约

## 数据流

RGB [B,3,352,352] → 冻结 Hiera-L + 可训练 block prompt adapters →
E1/E2/E3/E4 → 原 RFBModified → P1/P2/P3/P4 → 独立 AFFE →
原 SAM2UNetUp 三次融合 → 原 main、D2、D3 heads。

| 级别 | Hiera 通道 | RFB/AFFE 输入与输出 | 半谱 H×Wf |
|---|---:|---|---|
| P1 | 144 | [B,64,88,88] | 88×45 |
| P2 | 288 | [B,64,44,44] | 44×23 |
| P3 | 576 | [B,64,22,22] | 22×12 |
| P4 | 1152 | [B,64,11,11] | 11×6 |

独立 AFFE 支持任意正 H/W（包含奇数、非方形），dtype、device、通道/阶段顺序不变。
完整 P5.1 的输入尺寸规则仍由原模型决定，不能由 AFFE 的通用性推导完整模型支持任意尺寸。

## 配置与默认值

训练和测试共用 `networks/frequency_config.py`。

| CLI / 模型参数 | 默认 | 含义 |
|---|---|---|
| frequency_mode | none | none 完全保留基线；naf2d 仅允许 p5_sam2unet |
| frequency_enabled_stages | 1,1,1,1 | CLI 逗号分隔 0/1；Python 为四个 bool |
| frequency_reg_weight | 0.0 | 只用于训练损失，不改变 forward 的预测 |
| frequency_band_edges | 1/3 2/3 | 规范化径向频段边界 |
| frequency_band_weights | 1 1 1 | 非负分段正则权重，不能全零 |
| frequency_alpha_init | 0.1 | 每级独立 alpha=sigmoid(alpha_logit) 初值 |

示例为配置片段，**本轮未执行正式训练**：

```text
--encoder_name sam2_hiera_large --decoder_mode p5_sam2unet --img_size 352
--frequency_mode naf2d --frequency_enabled_stages 1,1,1,1 --frequency_reg_weight 0
```

保留原 P5.1 的实验参数：Structure Loss、D2/D3 各 1、boundary=0、
原 LR、batch、epoch 等应从既有 P5.1 命令恢复。项目 CLI 的旧默认值并不自动等于
P5.1 正式实验超参数，本次没有改变这些默认值。

## Forward、正则化与诊断

```python
from networks.frequency_enhancement import AdaptiveFrequencyEnhancement2D, FourStageFrequencyEnhancer

# 单级契约：Tensor -> 同形同 dtype Tensor
module = AdaptiveFrequencyEnhancement2D(channels=64)
y = module(x)

# 给未来 Fusion 的稳定接口；只增强，绝不融合或重采样。
enhancer = FourStageFrequencyEnhancer(enabled_stages=(True, True, True, True))
p1, p2, p3, p4 = enhancer.enhance_features([p1, p2, p3, p4])

# P5.1 普通输出接口保持原语义。
logits = model(images)  # Tensor[B,1,352,352]
outputs = model(images, return_aux=True)  # logits, aux_d2, aux_d3
outputs = model(images, return_aux=True, return_frequency=True)
loss = existing_segmentation_loss + lambda_freq * outputs['frequency_regularization']

# 显式诊断；默认不产生这些字典，也不在 module 内缓存图。
outputs = model(images, return_aux=True, return_frequency_diagnostics=True)
stage_diagnostics = outputs['frequency_diagnostics']
```

`forward_with_details` / `enhance_features(..., return_regularization=True)`
返回 `(增强特征, 可导正则标量, 可选诊断)`。默认仅返回特征。
四级正则按启用级数取均值，不会随启用数机械放大。
诊断包含 detach 的门控半谱、三个频段均值和全谱等效计数、输入/滤波后/残差后范数、
alpha、正则值和半谱重数加权的相对频谱变化。空段均值记 0 并配合 count=0 解读。
关闭 stage 原样返回同一个输入 tensor，诊断为 None；全部关闭时正则为零。
`none` 不实例化 AFFE，不添加任何参数或 buffer，不改变旧 state_dict 键。

训练按显式输出传递正则，无全局缓存或“上一次 forward”的状态依赖。
默认 λ=0 时不计算频段统计，不改变分割/辅助损失或 optimizer 超参数。
新增参数名位于 `progressive_decoder.frequency_enhancer.stages.{0..3}.*`，
原训练脚本的 non-encoder 参数组将其完整纳入。冻结 Hiera 和 block adapter 语义未改。

## Checkpoint 审计

普通推理使用 `load_frequency_state_dict(model, state_dict)` 严格加载；
`frequency_mode=none` 与原 P5.1 的参数空间相同。
训练增加可选 `--init_checkpoint`，仅 P5.1 可用；在 naf2d 时允许**完整旧模型**迁移。
预检 missing/unexpected/shape，再执行加载；仅允许全部 20 个新 AFFE 参数键缺失，
缺一个旧键、多一个未知键、形状不同、部分损坏的 AFFE checkpoint 都报错。
所有 missing/unexpected keys 写入日志，不覆盖输入 checkpoint。
`--init_checkpoint` 是权重初始化，不是恢复 optimizer/scheduler 状态。
原训练脚本对 `--hiera_checkpoint` 的检查依然存在；完整初始化权重在 Hiera 预加载后载入。

配置（enabled_stages、边界、权重）与既有 decoder 开关一样不保存在 state_dict 中，
必须随训练日志保存并在推理时重建。特别是改变 enabled_stages 不会使键审计失败。
本轮原文件 checkpoint 的实际验证状态见 verification_report，不以合成 state_dict 冒充。

## 3090 服务器复核（仅合成）

先检查 nvidia-smi 和当前进程，选择现场空闲 GPU；按用户最新授权，无需另行协商。以下占位符必须替换；不默认 GPU0 空闲。
从仓库根目录执行，Python 使用已确认的项目环境：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> NO_ALBUMENTATIONS_UPDATE=1 \
AFFE_FULL_SMOKE=1 AFFE_DEVICE=cuda \
HIERA_CHECKPOINT=<Hiera权重绝对路径> P51_CHECKPOINT=<P5.1权重绝对路径> \
<python> -m pytest -q tests

CUDA_VISIBLE_DEVICES=<空闲GPU> <python> -m tools.verify_frequency \
  --device cuda --full --batch 2 --warmup 3 --repeats 10 \
  --hiera-checkpoint <Hiera权重绝对路径> \
  --p51-checkpoint <P5.1权重绝对路径> --output <新的3090证据JSON路径>
```

脚本仅使用随机输入，不读取 Kvasir/CVC 数据。报告文件要求不存在，避免覆盖既有证据。
`AFFE_FULL_SMOKE` 实例化真实 Hiera，检查三个合成优化步骤、全部冻结参数、全部可训练参数梯度、
真实训练入口的 optimizer 参数收集表达式。提供 HIERA_CHECKPOINT 时加载预训练；
不提供时明确属于随机权重测试。需要 pytest（开发依赖），其他依赖同项目环境。
