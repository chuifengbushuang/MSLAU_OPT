# 实验9：级联离散频率选择与评分分段正则

直接相对原分支81b19e2冻结基线构建，独立运行，不导入实验7或其他实验目录。
只新增本实验，实验0–8与原始14文件不变。本版本是待检验候选，不宣称已有指标收益。

## 架构及二维适配

沿用实验7：F3=NAF3(P3,P4)，F2=NAF2(P2,F3)，F1=NAF1(P1,F2)。
三个Fi同时进入原U形Decoder，P4原样进入最深路径，无独立NAF4，不detach上下文。
RFB仍分别输出64通道；两个输入先双线性对齐到接收层，align_corners=True。
三个NAF接收层大小为22、44、88；各有local/context两个64→64的1×1投影。
联合描述仍为两路归一化幅值、幅值差、两个频率坐标，5→16→2选择器。
两路分别预测V=sigmoid(selector)，P=softmax(V)，选择softmax温度为1。
FFT/IFFT和整个新模块在FP32运行（数值检查可用float64），不混合两路复数相位。

每对共轭频率为一个选择单元；rFFT边界内的一对共享平均评分和二值掩码。
DC及其他自共轭位置固定保留；其余G对选择K=ceil(0.9G)，保留复数系数，未选中置零。
这不是保留90%能量。当前偶数网格有4个固定位置：

| 接收网格 | G | K | 完整频谱保留比例 |
|---|---:|---:|---:|
| 22×22 | 240 | 216 | 436/484 ≈ 90.083% |
| 44×44 | 966 | 870 | 1744/1936 ≈ 90.083% |
| 88×88 | 3870 | 3483 | 6970/7744 ≈ 90.005% |

训练：对log(P)+Gumbel取Top-K，实现加权无放回抽样。
随机数在CPU局部Generator产生，SHA256键仅含协议naf_discrete_v1、seed、epoch、样本ID、层级和分支。
不含实验名、repeat、worker、batch序号；不消费全局CPU/CUDA或数据增强RNG。
每次训练前向必须通过set_selection_context传入当前seed/epoch/样本ID，遗漏即报错。
评估：按V排序，稳定Top-K，平分按固定索引顺序，不抽样。
训练抽样与评估Top-K存在分布差异，这是明确的工程选择，不声称原论文规定了该推理方式。

前向始终为真正0/1掩码。反向使用有偏straight-through连续近似：
q=sigmoid((s-t)/tau)，tau=0.1，阈值t满足sum(q)=K。
自定义反向包含阈值的隐式导数diag(d)-d*d.T/sum(d)，d=q(1-q)/tau。
实现为hard+(q-stop_gradient(q))；连续近似只影响反向，不恢复被删频率。
这不是离散Top-K的真实导数或无偏估计；测试对连续近似做有限差分检查。

去掉实验7的内部alpha混合，直接IFFT(mask*Q)。保留GAP+centered RMS两路逐通道softmax融合，
再经零初始化输出投影，Fi=Pi+Wout(fused)。外部Pi残差仍是未滤波通路，
因此只能声称新分支执行完整滤波，不能声称整个网络删除了某些频率。
共享参数初值、构造后的CPU RNG与实验7一致；新增56,118参数，比实验7少6个alpha参数。
零输出投影使初始Decoder输出与基线完全一致；输出投影更新后选择器可获得任务梯度。

## 分段正则

约束选择前的评分V，不约束二值mask，也不约束softmax概率。
使用平方L2，rho=sqrt((2fy)^2+(2fx)^2)/sqrt(2)，以1/3和2/3分低/中/高频。
local的n=i，context的n=i+1，混合上下文标签是层级先验，不是原文物理时间尺度。
三段系数依次为n^-1、n^-2、n^-3；n=1三段同权，没有频段偏向。

R_i,b=mean(batch,可选择共轭对)[V^2 * n^(-band_index-1)]。
固定位置不参与；每个可选择对均含两个完整频谱系数，按对均值等于按这部分完整频谱均值。
R=sum(三个模块×两个分支的R_i,b)/6，lambda=0.001，仅训练加lambda*R，验证损失保持原公式。
lambda是初版工程设定，不因与实验7数值相同就认定强度相同。
记录V均值/频率标准差、归一化概率熵、分频段保留率、完整频谱保留率、R、
投影后分支的能量保留率/相对变化，以及融合权重/外部残差比。能量比不能代替分割指标。

## 训练协议和记录

采用已有paired_aug_v1：原增强操作/概率不变，按样本、epoch固定增强，独立采样器RNG。
原始实验0和旧NAF不补丁化回写。paired_inputs.py的输入算法与已验证基线逐项相同，
只在initial_state.json额外记录剔除新模块后的baseline_state_sha256，
可与exp00_baseline_paired_aug_v1的state_sha256核对；完整模型state_sha256本来就应不同。
同协议不代表CUDA训练逐位确定，剩余计算波动仍需纳入结果解释。

352/b12/nw8/e20、Structure、两组LR=1e-3、AdamW wd=5e-4、原cosine、D2/D3权重均1、boundary=0。
官方Hiera初始化、冻结骨干和BlockAdapter不变。对照使用exp00_baseline_paired_aug_v1；
旧实验7/8增强未配对，只作参考，不能据此隔离新版频率选择收益。
后续仅融合、均匀抽样、去正则对照尚未在本实验中混入。

输出遵循/data/models/le/mslau-net/{checkpoints,experiments,visualization}/<run_id>，
保留固定IoU、带指标IoU、带指标Loss三份原始命名权重；训练后自动独立评估最佳IoU权重。
记录源码/补丁/配置/官方权重哈希、环境、运行时、初值、40项输入指纹与高精度epoch指标。
epoch指标额外保存naf_regularization和selection_mode；验证不把R加入loss。
P51_REPEAT只标记重复，不改变初始化、输入或频率抽样。

## 服务器命令（本代理不启动正式训练）

在仓库根目录，用已克隆的mslau-p51环境，不重新安装依赖。

合成检查含官方Hiera、正式batch12、一次合成优化更新和第二次纯任务反向，检查选择器梯度打开：

```bash
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 EXP09_FULL_SMOKE=1 EXP09_DEVICE=cuda EXP09_SMOKE_BATCH=12 /home/le/miniconda3/envs/mslau-p51/bin/python -m unittest discover -s tests -p 'test_exp09_discrete.py' -v
```

这是随机图像/掩码的合成检查，不使用原数据训练或保存模型权重。通过后正式三seed：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    for P51_SEED in 1234 42 2026; do
        /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp09_naf_discrete --seed "$P51_SEED" --gpu 0 --run
    done
)
```

只有--run会正式训练；移除它可预览。GPU占用只提示，不阻止运行；失败即停止循环。
基线paired协议尚缺seed42/2026时，只补对应种子；seed1234已有两次应全部保留报告，不挑最高。

## 借鉴边界

NAF-MSNN（IEEE TASE 2025, DOI 10.1109/TASE.2025.3579694）III-B式6–9：
评分→概率→选择90%分量、对V施加分段约束、逆变换。本文Linear在初版取恒等映射。
SPLF-SAM：相邻层级级联融合的连接启发；未复刻其UMFEB/MAFA/PromptBank。
当前二维实频谱分组、幅值描述、Gumbel抽样/梯度近似、确定性推理与GAP+RMS融合都是明确的适配。
Gumbel无放回抽样依据：Kool et al., ICML2019, https://proceedings.mlr.press/v97/kool19a.html 。
不宣称这是两篇论文的逐项复现或已验证创新效果。
