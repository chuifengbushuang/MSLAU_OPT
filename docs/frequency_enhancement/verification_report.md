# AFFE 验证报告（2026-10-13，Asia/Shanghai）

## 验收结论与限制

本地实现与测试已完成；用户随后在 RTX3090 上执行了提交 `604b1fd` 的合成、
真实权重加载与性能验证，并回传终端输出。目标环境的本轮工程检查通过，尚无正式分割效果结论。

证据来源需区分：本地测试由本助手执行；3090 结果来自用户回传，不是本助手直接登录执行。
完整服务器 JSON 位于 `/home/le/WorkSpace/20261012_PolypSegmentation/affe_3090_verification_01.json`，
完整JSON与两份日志现已同步、实际读取并原样归档到本报告同目录；
SHA256见 `evidence_3090.sha256`。已检查数值有限性、参数计数、全部加载审计键和各级梯度。
未启动正式 Kvasir/CVC 训练，未使用验证集调参，未改变 split、原 head 或 fusion。

## 3090 验收更新：用户回传（2026-10-13）

- 环境：Python3.10.22，torch2.1.2+cu118、torchvision0.16.2+cu118、timm1.0.15、
  numpy1.26.4、pytest8.4.2；驱动535.154.05；`pip check` 无损坏依赖。
- 用户确认数据继续使用 `/data/dataset/zsy/mslau-net/Kvasir-SEG` 和
  `/data/dataset/zsy/mslau-net/CVC-ClinicDB`。GPU 使用规则更新为：启动前检查，空闲卡即可使用，
  不再要求另行与同学协商；不占用已有任务的卡。本轮 CUDA_VISIBLE_DEVICES=0。
- 随机权重完整测试：40 passed、1 skipped（当时未指定真实checkpoint），16.02秒。
  CUDA FFT重建最大误差4.76837158203125e-7，梯度有限。
- 真实权重定向测试：2 passed、5 deselected，23.57秒；补齐之前跳过的checkpoint测试，
  并在Hiera预训练下重跑冻结参数、Adapter和AFFE梯度测试。不是单次运行42项不同测试。
- TypedStorage 弃用警告来自加载旧checkpoint，测试与性能脚本均完成，无需因此修改checkpoint。
- none：严格载入1180个键，missing=[]、unexpected=[]。
- naf2d：载入同样1180个旧键，unexpected=[]；missing恰为每级alpha_logit、
  gate_in.weight/bias、gate_out.weight/bias，共20项，legacy_migration=true。
  这是预期的新参数初始化，不是旧权重加载失败。

条件：B=2、352输入、FP32、预热3次、测量10次、完整三路输出；
前后向使用合成均方输出损失，不含optimizer.step、数据加载或正式训练的优化器状态。

| RTX3090 | 原P5.1 | P5.1+AFFE | 增量 |
|---|---:|---:|---:|
| 参数量 | 216419043 | 216427751 | 8708 |
| 可训练参数 | 4269747 | 4278455 | 8708 |
| 前向中位ms | 57.7094 | 60.5151 | +2.8057（4.86%） |
| 前后向中位ms | 145.3039 | 160.2846 | +14.9807（10.31%） |
| 前向峰值MiB | 997.48 | 1006.04 | +8.56 |
| 前后向峰值MiB | 3212.47 | 3239.67 | +27.20 |

远程日志：项目父目录中的 `affe_3090_tests_01.log`、
`affe_3090_pretrained_tests_01.log` 和 `affe_3090_verification_01.json`。
命令为接口文档中的pytest与verify_frequency，用户采用历史记录中的权重环境变量；
回传摘要确认pretrained_hiera=true和真实checkpoint加载成功，尚未回传权重文件哈希。


### 完整JSON补充核对

随机合成特征、未训练AFFE的各级诊断如下，不代表已学到去噪规律。

| Stage | 前向ms | 前后向ms | FFT最大误差 | 初始输出相对变化 |
|---|---:|---:|---:|---:|
| P1 | 0.876 | 3.542 | 1.431e-06 | 0.1081% |
| P2 | 0.893 | 3.477 | 1.431e-06 | 0.0795% |
| P3 | 0.956 | 3.733 | 7.153e-07 | 0.0919% |
| P4 | 0.996 | 3.622 | 7.153e-07 | 0.0908% |

四级AFFE整体前向4.060ms、前后向12.033ms；原/新Decoder前向6.884/10.451ms、
前后向32.152/35.830ms。全部alpha初值约0.1，全部门控与alpha参数初始梯度有限且非零。
三频段平均权重均接近1；输入乘3后的最大门控变化约0.029–0.038，支持输入自适应性。
完整四级FFT最大误差1.4305e-6，符合2e-6容差；此前4.7684e-7是单次小张量检查，
不能误写为所有尺寸的误差上限。
单级显存是测量时进程的绝对峰值，可能包含库工作区及其他活跃张量，
不能相加或解释为每级净新增显存；完整模型配对测量是本次开销比较的主要依据。

下面保留首次本地验收的历史结果；其中“未测”是当时状态，3090/真实权重项已由本节补齐。

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
  首次本地验收不覆盖torch2.1.2；后续用户回传的3090测试已覆盖该版本，见上节。
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

## 剩余归档与下一阶段

1. 完整JSON与两份测试日志已归档并核对，保留SHA256；该项完成。
2. 如进入正式实验，先固定运行配置、数据路径和split、权重哈希、seed与模型选择规则。
3. 按future_ablation_plan开展原P5.1 vs AFFE（λ=0）单变量对照；
   两组必须使用相同初始化协议，不能只让AFFE从完整P5.1 checkpoint继续训练而让基线从头训练。
4. 本轮不启动正式训练，不把工程验收解释为Dice/mIoU提升。
