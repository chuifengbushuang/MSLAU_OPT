# 实验8：仅级联融合对照

用于与实验7比较，检验频率处理及其正则在级联融合之外是否带来额外收益。
直接相对冻结P5.1提交81b19e2构建；自己的changes.patch和added/networks/cascade_fusion.py可展开为独立完整源码，不在运行时导入实验7。实验0–7的既有文件不改。

## 保留与移除

```text
D4 = P4
F3 = Fusion3(P3, P4)    D3 = 原UpBlock3(D4, F3)
F2 = Fusion2(P2, F3)    D2 = 原UpBlock2(D3, F2)
F1 = Fusion1(P1, F2)    D1 = 原UpBlock1(D2, F1)
```

保留实验7的三个双输入模块、64通道投影、上下文上采样、GAP+centered RMS、两路逐通道softmax门控、输出投影与Pi直接残差。P4原样进入Decoder，无第4级模块，不detach上下文。
原RFB、SAM2UNetUp、D2/D3辅助头、分割头及训练/评估口径均不变。

每个融合模块的实际前向为：

```text
X = Conv1x1_local(Pi)
C = Conv1x1_context(Resize(context, Pi大小, bilinear, align_corners=True))
descriptor = Cat(GAP(X), centered_RMS(X), GAP(C), centered_RMS(C))
(a,b) = Softmax_branch(Conv1x1_16_to_128(GELU(Conv1x1_256_to_16(descriptor))))
H = a*X + b*C
Fi = Pi + Conv1x1_output(H)
```

centered_RMS=sqrt(mean((X-GAP(X))²)+1e-6)，所有共享运算顺序、FP32 autocast区域与实验7一致。
没有FFT/IFFT、频谱描述、选择掩码、频率预算、alpha或正则项；不生成虚假的NAF诊断。
训练字典不返回naf_regularization，损失中也没有对应加法。配置显式记录naf_reg_weight=0，CLI只接受0。
只记录每级context_weight、channel_gate_std、image_gate_std、residual_ratio，统计方式与实验7相同。
frequency_enhancer容器名保留以便逐参数对照，不表示仍执行频域运算。

## 共享层初始化完全一致

实验7的初始化顺序是：local投影、context投影、频率选择器、融合门控、输出投影。
直接跳过选择器初始化会改变后续门控初值，因此实验8在同样的CPU RNG隔离区中构造一个临时选择器，仅消耗相同初始化随机数，随后丢弃。
它不注册为模型子模块、不进入state_dict或优化器、不执行前向，也不依赖旧实验文件。
这使三个种子1234/42/2026的所有共享投影/门控权重逐元素匹配实验7，且不改变原Decoder及后续全局CPU随机状态。

融合门控末层和输出投影继续零初始化；初始两路权重0.5，Fi=Pi，原模型初始输出不变。
新增可训练参数55,728，每级18,576；比实验7少396个选择器/alpha参数。全部保留的参数均参与可学习融合，没有闲置的频率参数。
严格加载本实验checkpoint，不将实验7训练权重删参数后作为起点。

## 协议与检查

相对实验7的config仅改变id和naf_reg_weight：352/b12/nw8/e20、Structure、两组LR1e-3、AdamW及原调度、D2/D3各1、boundary0、官方Hiera初始化、原数据划分及增强均不变。
本对照同时移除频率处理及相应正则，不能单独分离正则作用。相同seed也不保证增强轨迹逐次一致。

测试验证共享初始化/RNG、非零融合权重下与“实验7滤波恒等化”的前向及反向一致、真实级联输入及梯度、FFT被禁止时正常运行、初始原解码器输出、64通道BF16 autocast、严格重载、实际训练无正则及三份权重保存。

服务器拉取后，先运行合成检查（含官方Hiera、batch12、Structure主/辅助损失反向，不读取训练集、不执行正式训练）：

```bash
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 EXP08_FULL_SMOKE=1 EXP08_DEVICE=cuda EXP08_SMOKE_BATCH=12 /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp08_fusion.py' -v
```

默认使用原官方Hiera路径，HIERA_CHECKPOINT环境变量可指定本地权重。完整检查打印EXP08_FULL_SMOKE、实际batch及峰值allocated显存。

通过后，三个种子依次正式训练/评估，任一次失败即停止：

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

三输出根目录、run_id及原三份权重命名不变：checkpoints、experiments、visualization；保留kvasir_best_model.pth、best_iou_*、best_loss_*。
默认debug=false，无预测图时不创建图像目录。GPU占用仅提示；不带--run只校验并预览。没有增加调度器或新训练入口。

本地PyTorch2.13 CPU检查：完整61项中58通过，旧NAF、实验7、实验8的完整Hiera检查各跳过1项；实验8新增10项中9通过、1项服务器检查未执行。10个展开Python文件语法、14份基线与原提交字节一致、所有旧实验文件未改检查通过。未加载服务器官方权重或执行3090 batch12检查，未启动正式训练。
