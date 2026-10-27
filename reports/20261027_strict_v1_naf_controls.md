# 实验3.1 / 9.1：匹配严格v1基线的六次NAF训练

本批明确保留v1增强联动和严格CUDA计算，仅加入NAF；已有基线和所有旧实验不改、不重跑。两组各使用42、1234、2026三个seed，从同一个官方SAM2 Hiera文件初始化，不用已训练的P5.1权重继续训练。

| 实验 | 入口 | 模型与正则 |
|---|---|---|
| 3.1 | exp03_1_naf_p12_paired_strict_v1 | P1/P2独立软预算NAF，R=(R1+R2)/4，lambda=0.001 |
| 9.1 | exp09_1_naf_discrete_paired_strict_v1 | 原实验9三级级联、90%离散选择、V分段正则六路均值，lambda=0.001 |

两个模型文件分别与已完成的严格v2实验3/9逐字节相同，仅回到v1输入算法。实验3.1保留严格CUDA所需的固定顺序累加，不退回无法通过torch2.1.2严格检查的旧浮点cumsum。实验9.1保留原频率抽样键naf_discrete_v1，无独立NAF4、无内部alpha、保留外部残差；不是实验7的软频率融合版本。

训练为352/b12/nw8/e20、原Structure三头监督、AdamW两组LR=1e-3、weight_decay=5e-4、原Cosine、D2/D3各1、boundary=0。原Decoder、数据划分880/120、数据路径、官方预训练SHA、TF32/注意力开关及最佳权重选择保持原样。实验名和重复标签不参与输入或频率抽样种子。

paired_inputs.py与严格v1基线只在record_initial_state额外增加baseline_state_sha256。其余输入逻辑和strict_runtime.py与该基线完全相同；运行后应核对40条输入记录和共享原层初始哈希。完整含NAF模型的state_sha256本来就不同。

已有基线最佳验证IoU：42=0.8745769829、1234=0.8537621558、2026=0.8820540309，均值0.8701310565。1234两个严格重复完全相同，只计一次。比较各seed差值、三seed平均差、方向及Dice/末5轮，不只看seed2026峰值。v1是含增强联动的特定协议，不能把它与v2混成一组基线。

在服务器仓库目录执行以下拉取及检查，不需要重新激活环境。默认物理GPU0，可把后续P51_GPU改成其他空闲3090编号。GPU占用只提示，不阻止运行。

```bash
git pull --ff-only origin exp/p51-independent
```

合成检查不会使用真实训练数据，也不保存正式权重。CPU部分检查三个seed两轮train/valid增强输入及0/8-worker一致性、共享原层初值和RNG；GPU部分对两个官方Hiera候选分别独立初始化两次，各做两步合成更新和一次评估，检查重复结果和梯度。原3090/PyTorch2.1.2检查须在服务器执行，本地CPU通过不替代此项。

```bash
(
    set -e
    set -o pipefail
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    P51_GPU=0
    CUDA_VISIBLE_DEVICES="$P51_GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 NAF_V1_FULL_SMOKE=1 \
        /home/le/miniconda3/envs/mslau-p51/bin/python -u \
        -m unittest discover -s tests -p 'test_naf_strict_v1.py' -v \
        2>&1 | tee "$P51_OUTPUT_ROOT/experiments/naf_strict_v1_smoke_$(date +%Y%m%d_%H%M%S).log"
)
```

检查通过后，一次执行下列批量命令，按42、1234、2026各运行实验3.1和9.1，共六次正式训练及独立评估。任一次失败即停止；再次执行整个块会创建新重复记录，不会自动跳过已完成实验。

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    export NO_ALBUMENTATIONS_UPDATE=1
    export P51_REPEAT=1
    P51_GPU=0
    for P51_SEED in 42 1234 2026; do
        for P51_EXPERIMENT in exp03_1_naf_p12_paired_strict_v1 exp09_1_naf_discrete_paired_strict_v1; do
            /home/le/miniconda3/envs/mslau-p51/bin/python -u \
                tools/run_experiment.py "$P51_EXPERIMENT" \
                --seed "$P51_SEED" --gpu "$P51_GPU" --run
        done
    done
)
```

每次使用独立run_id，输出布局仍为：

- `/data/models/le/mslau-net/checkpoints/<run_id>/`：`kvasir_best_model.pth`、`best_iou_…pth`、`best_loss_…pth`三份权重。
- `/data/models/le/mslau-net/experiments/<run_id>/`：配置、命令、环境、源码哈希、console/evaluation日志、初值和输入哈希、逐轮指标、数值状态及权重内容哈希。
- `/data/models/le/mslau-net/visualization/<run_id>/`：实际启用可视化时写入；沿用debug=false，不承诺生成图片。

代理仅检查和提供命令，不代为启动这六次正式训练。
