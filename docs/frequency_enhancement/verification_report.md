# AFFE 验证报告（2026-10-13，Asia/Shanghai）

## 验收结论与限制

本地代码、四级独立模块、P5.1 集成、默认模式兼容、梯度、CPU/CUDA AMP 和复杂度检查已执行。
**最终实验环境指定为 4×RTX 3090 共享服务器，尚未完成该服务器验收。**
SSH 连接 `le@10.243.96.120:22` 在 8 秒连接超时内未成功，未进入认证；没有登录或占用服务器 GPU。
等待用户提供只读查询输出以及与同学协调后的可用卡号。
本地没有找到本项目的 Hiera 预训练权重或原 P5.1 训练 checkpoint。
因此真实预训练冒烟、真实旧 checkpoint 迁移和 3090 性能数据均**未验证**。
没有启动正式 Kvasir/CVC 训练，没有读验证集调参，没有改变 split、原 head 或 fusion。

## 基线与环境

- 原分支 `lien1012`，HEAD `81b19e2c801efcf74826b8127f5ea712b0247ec5`，初始工作树干净。
  本地 origin/lien1012 指向相同提交（未 fetch）。功能分支 `feat/p5-naf-frequency-enhancement`。
- 完整读取 AGENTS.md；其提到的 handover.md 本地不存在，已报告。
  实际查看 networks/mslau_net.py、edge_guidance.py、msla.py、训练/测试、loss、loader、两份论文及 PPT。
- 本地 Python 3.11.17，torch 2.13.0+cu130，CUDA build 13.0，torchvision 0.28.0+cu130。
  `/tmp/affe-venv` 使用已有 torch/torchvision，隔离安装 timm 1.0.15、pytest 9.1.1、
  albumentations 1.4.24、torchmetrics 1.6.1、numpy 1.26.4 等项目依赖。
  未修改原 Conda 环境。完整包版本见 `environment_local.txt`。
- 项目 environment.yml 标称 Python3.10 / torch2.1.2 / CUDA11.8；**本地不等于该环境**。
  需在最终 3090 环境再次运行，不能宣称 torch2.1.2 已实测兼容。
- 本地 GPU：RTX 5070 Ti，16,303 MiB，驱动查询 NVIDIA-SMI 615.78.05 / KMD 617.42。
  开始检查时使用 2,406 MiB，利用率 3%，nvidia-smi 无可见运行进程；进程列表无 python 训练。
  WSL 显示/系统显存不等于 PyTorch 显存，进程可见性有限。
- 沙箱内 nvidia-smi 不在 PATH；使用 `/usr/lib/wsl/lib/nvidia-smi` 时 GPU 被 OS 沙箱阻止。
  经工具授权在沙箱外检查并执行本地 CUDA 合成验证；沙箱内 CUDA 测试自动 skip。

## 实际命令与结果

从仓库根目录执行。`NO_ALBUMENTATIONS_UPDATE=1` 仅关闭依赖库联网检查。

```bash
# Phase 1，先测独立模块：24 passed, 10 skipped（CUDA 在沙箱不可见）
PYTHONPATH=/tmp/affe-deps:. /home/lien/miniconda3/envs/test-py311/bin/python \
  -m pytest -q tests/test_frequency_enhancement.py

# 集成 CPU：27 passed, 12 skipped（CUDA、完整 Hiera opt-in、真实 checkpoint）
NO_ALBUMENTATIONS_UPDATE=1 /tmp/affe-venv/bin/python -m pytest -q tests

# 本地 CUDA，真实结构 + 随机权重，三步合成优化：38 passed, 1 skipped
NO_ALBUMENTATIONS_UPDATE=1 AFFE_FULL_SMOKE=1 AFFE_DEVICE=cuda \
  /tmp/affe-venv/bin/python -m pytest -q tests

# 后增加的 Decoder AMP 两项：2 passed
NO_ALBUMENTATIONS_UPDATE=1 /tmp/affe-venv/bin/python -m pytest -q \
  tests/test_p5_frequency_integration.py -k decoder_cuda_amp

# 最终完整回归（结果以保存的 XML 为准）
NO_ALBUMENTATIONS_UPDATE=1 AFFE_FULL_SMOKE=1 AFFE_DEVICE=cuda \
  /tmp/affe-venv/bin/python -m pytest -q tests \
  --junitxml=docs/frequency_enhancement/pytest_local_5070ti.xml

# 两个入口 --help、语法和 whitespace
NO_ALBUMENTATIONS_UPDATE=1 /tmp/affe-venv/bin/python kvasir_train.py --help
NO_ALBUMENTATIONS_UPDATE=1 /tmp/affe-venv/bin/python kvasir_test.py --help
/tmp/affe-venv/bin/python -m py_compile networks/frequency_enhancement.py \
  networks/frequency_config.py networks/mslau_net.py kvasir_train.py kvasir_test.py \
  tests/test_frequency_enhancement.py tests/test_p5_frequency_integration.py tools/verify_frequency.py
git diff --check
```

最终完整回归：**40 passed, 1 skipped, 1 warning，24.57 秒**，XML 已保存。
全部已执行测试无失败。依赖发出一次 torch.jit.interface 弃用警告，不是数值错误。
唯一实际 checkpoint 测试未提供 P51_CHECKPOINT，明确 skip。
开发过程中没有以 stub 替代 Hiera；完整梯度验证构造 timm 的真实 Hiera-L。

## T1–T10 覆盖

| 检查 | 实际证据与边界 |
|---|---|
| T1 | 单元测试奇/偶/非方形/1×1，FFT/IFFT 与权重恒等重建；四级 FP32 最大绝对误差 ≤9.54e-7 |
| T2 | [2,64,88,88]、44、22、11 四级输出形状/dtype 一致 |
| T3 | 参数 data_ptr 集合两两不交；只改 P1，P2–P4 逐元素相同；eval 确定；禁用返回原 tensor |
| T4 | 幅值乘 3/4 改变动态 M；门控有频点方差；人工 M=1.2、alpha=.1 时输出为1.02x |
| T5 | 初始以及3步合成更新：input、gate_in/out、alpha、RFB、48个 block prompt adapter 均有有限非零梯度；真实 Hiera 原参数全量快照比较不变；正则独立反向有效 |
| T6 | 0、1、1e4、1e-8、随机输入，奇数/非方形；FP32、CUDA FP16/BF16；无 NaN/Inf；输入 dtype 保留；Decoder AMP 也通过 |
| T7 | 合成四级 encoder features，B=2 三路352输出+原 Structure Loss 反向；真实 Hiera 随机权重 B=1 三步合成优化；**预训练未测** |
| T8 | 从 git show 81b19e2 导入独立原 Decoder，none 的全部键/初始化值/三个输出逐元素完全相同；旧 state_dict strict 成功；naf2d 只缺20个新增参数；损坏键/形状/部分AFFE拒绝；**真实训练checkpoint未测** |
| T9 | 实例化完整 P5.1 统计参数；CPU、GPU 两种 decoder 与 AFFE、各级耗时；本地 GPU 完整模型 B=2 前后向/峰值显存；**3090 未测** |
| T10 | 显式 detach 诊断、频段均值/计数、范数、alpha、R、谱变化；无模块级图缓存；半谱特殊列及重建频谱一致性通过 |

正则 R(M) 本身不依赖 alpha，故单独 R.backward() 不应要求 alpha 有梯度；
分割/重建损失提供 alpha 梯度。默认 λ=0 的初始完整模型反向也验证了所有 AFFE 参数有梯度。
optimizer 测试执行训练源码中的 encoder_params/decoder_params 表达式，确认无重复、无遗漏，
其参数集合恰好等于所有 requires_grad 参数。

## 参数量和初始数值

| 项目 | 参数量 |
|---|---:|
| 原完整 P5.1 | 216,419,043 |
| 原可训练 | 4,269,747 |
| 新完整 P5.1 + AFFE | 216,427,751 |
| 新可训练 | 4,278,455 |
| AFFE 合计 | 8,708 |
| 每级 AFFE | 2,177 |
| 原 Decoder / 新 Decoder | 2,555,331 / 2,564,039 |

CPU seed1234 的四级初始相对输出 L2 变化为
0.00098964 / 0.00109704 / 0.00094474 / 0.00068731（约0.07%–0.11%）。
对应 alpha_logit 梯度 L1 为 0.001024 / 0.000964 / 0.000999 / 0.000919；
gate_in.weight 梯度 L1 为 0.06438 / 0.08133 / 0.05179 / 0.07872。
输入幅值乘3后门控最大变化约0.04243 / 0.04103 / 0.03016 / 0.01556。
这验证初始化非空操作、近恒等和可导，不是训练效果或医学边缘选择结论。

## 性能（仅本地环境）

使用 3 次预热、CPU 5 次/GPU 10 次测量，报告中位数，GPU 每步计时前后 synchronize。
输入 B=2，FP32，CPU threads=2；前向 eval+no_grad，前后向 train，合成均方输出 loss，
不含 optimizer.step、数据加载或 Structure Loss 耗时。完整模型两种模式均计算三路输出。
GPU 峰值为 PyTorch max_memory_allocated，包含活跃模型/输入，非 nvidia-smi 全卡占用；
不含 AdamW 状态，所以不能用来估计 batch12 正式训练所需显存。

| 本地 RTX5070Ti | 前向中位 ms | 前后向中位 ms | 前向峰值 MiB | 前后向峰值 MiB |
|---|---:|---:|---:|---:|
| 原完整 P5.1 | 46.243 | 121.534 | 976.63 | 3138.91 |
| 完整 P5.1 + AFFE | 50.019 | 137.336 | 1007.88 | 3161.15 |
| 四级 AFFE 单独 | 2.748 | 8.898 | 26.96 | 49.63 |

各级 GPU 前向中位约0.743 / 0.740 / 0.711 / 0.703 ms；前后向约
4.807 / 1.735 / 1.689 / 1.685 ms。该平台小 kernel 启动/调度波动较明显，
单级之和与四级一起计时不会严格相等；原始 JSON 含 min/max，不把短测结果当精确硬件规律。
88×88 存储3960个频点，11×11 仅66个；FFT 复杂度 O(BCHW log(HW))，
MLP 主要为 O(BHWf(C+3)h + BHWfhC)，h=16，P1 是主要张量成本。
11×11 三段全谱等效计数21/68/32（与存储半谱数不同），有限分辨率需谨慎解释。

CPU 四级 AFFE 前向9.962 ms、前后向30.880 ms；原/新 Decoder 前向75.068/72.789 ms，
前后向230.368/261.452 ms。短 CPU 计时出现新模式前向略快属于波动，**不能据此声称加速**。

```bash
/tmp/affe-venv/bin/python -m tools.verify_frequency --device cpu --repeats 5 \
  --output docs/frequency_enhancement/evidence_cpu_with_input_grad.json
/tmp/affe-venv/bin/python -m tools.verify_frequency --device cuda --full --batch 2 --repeats 10 \
  --output docs/frequency_enhancement/evidence_local_5070ti.json
```

输出路径已存在，复跑需换新路径。早期 `evidence_cpu.json` 未打开合成输入梯度，
保留为开发记录；上表采用修正后的 `evidence_cpu_with_input_grad.json`，
包含向 encoder/RFB 输入传播梯度的成本，GPU 报告也是修正后脚本生成。

## 后续必须补齐

1. 获取服务器 Python/torch/CUDA/timm 版本、项目分支/HEAD、现场 GPU 占用、经协调的卡号。
2. 在独立分支/目录同步可审查改动，先比较服务器工作树，不覆盖同学或用户修改。
3. 提供 Hiera/P5.1 权重路径，按 interface_contract 中命令完成 pretrained/full/checkpoint 测试，逐项保存加载审计。
4. 用实际 3090 复测 AMP、延迟、显存；如安装环境需要变动，优先隔离环境而非升级共享环境。
5. 以上通过后，再按 future_ablation_plan 另行启动预注册单变量实验。本轮不执行。
