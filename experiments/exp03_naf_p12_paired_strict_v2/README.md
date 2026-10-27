# 实验3：在可重复的v2基线上直接检验原P1/P2 NAF

当前只回答“原先表现较好的NAF能否在新基线上有增益”。采用原实验3设计，不新增级联融合，不调参，不接着已训练的baseline权重微调。

旧实验3三seed均值0.873464，略高于实验7的0.872907；旧输入及数值随机性未完全控制，作为候选筛选依据，不能当作本轮增益结论。

与已完成exp00_baseline_paired_strict_v2配对：seed1234/42/2026各一次，基线最佳IoU分别0.8682272643/0.8691121638/0.8768093199，均值0.8713829160。不重跑基线，比较同seed差值及三seed等权平均差，同时报告各seed方向、Dice和最后5轮表现，不挑最高重复。

本实验直接相对冻结基线81b19e2构建，独立补丁/新增文件，旧实验及v2基线不变。NAF结构及公式采用历史shared_naf_v1，仅为严格CUDA替换累加实现：

- RFB输出64通道后，仅P1(88×88)、P2(44×44)接原BudgetNAF2D；P3/P4旁路，原Decoder不变。
- 原软频率预算0.9、temperature0.1、alpha初值0.1/上限0.25不变；不是实验9的离散90%选择。
- 原正则R=(R1+R2)/4，lambda=0.001不变，仅训练加入；新增164个参数。
- paired_aug_v2输入与strict_cuda_v1训练/评估匹配已验证基线；新增原层初始状态baseline_state_sha256，核对共享初值。
- 从同一官方SAM2 Hiera预训练文件初始化适配器/解码器，冻结主干，352/b12/nw8/e20、Structure三头监督、AdamW两组LR1e-3、Cosine保持不变。

2026-10-27服务器合成检查发现PyTorch2.1.2的CUDA浮点cumsum在严格模式下报错，正式训练尚未启动。当前仅在本实验的NAF文件中，将前缀/后缀累加换为固定树形顺序的切片＋加法＋拼接；前向与自动求导均留在原设备，没有CPU回传，也不关闭严格模式。累加数学公式不变，但浮点加法顺序不同，不承诺逐位复现旧非确定性实现。原模型参数、初始化、预算、alpha、正则及训练配置均不变，旧实验和已完成v2基线文件不改。

新增检查对比原累加的前向/梯度与预算，覆盖88/44真实尺寸，并在禁止调用原生cumsum时检查NAF前向/反向。服务器合成检查额外核对CUDA累加及NAF重复性；运行时间尚未在3090上测量。

仓库中实验8/9同协议入口仍保留供后续机制分析，当前不用执行那六次训练。仅运行本实验三次即可。

服务器完整命令（不需激活环境或切换目录）：

```bash
(
    set -e
    set -o pipefail
    git pull --ff-only origin exp/p51-independent
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1

    CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 P12_STRICT_FULL_SMOKE=1 \
        /home/le/miniconda3/envs/mslau-p51/bin/python -u \
        -m unittest discover -s tests -p 'test_p12_strict_v2.py' -v \
        2>&1 | tee "$P51_OUTPUT_ROOT/experiments/p12_strict_v2_smoke_$(date +%Y%m%d_%H%M%S).log"

    for P51_SEED in 1234 42 2026; do
        P51_REPEAT=1 /home/le/miniconda3/envs/mslau-p51/bin/python -u \
            tools/run_experiment.py exp03_naf_p12_paired_strict_v2 \
            --seed "$P51_SEED" --gpu 3 --run
    done
)
```

合成检查先验证原模型/配置、三seed共享初值、真实小型训练重复性和三权重，再在严格CUDA下对官方Hiera做batch12两步随机合成更新及评估；不使用原数据、不保存正式模型。通过后才依次启动三次20轮正式训练。输出继续按run_id进入原三个根目录，保留三份历史命名权重。代理不代启动训练。
