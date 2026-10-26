# 实验0：配对输入与严格CUDA执行

直接相对原始冻结提交81b19e2构建。原基线、paired_aug_v1及所有旧NAF实验保持不变。
新增的numerical_protocol为strict_cuda_v1，input_protocol仍为paired_aug_v1；paired_inputs.py与旧配对基线逐字节相同，不改变增强序列、样本顺序或初始化算法。

原网络、352/batch12/workers8/20epoch、官方Hiera初始化、AdamW weight_decay=5e-4、两组LR1e-3、Cosine、Structure主损失及D2/D3各权重1不变。
不增加warmup、梯度裁剪、AMP、EMA或新初始化，不改变TF32及SDPA后端允许状态。

训练与独立评估都启用torch.use_deterministic_algorithms(True, warn_only=False)、CuDNN deterministic=True/benchmark=False；启动器在子进程启动前固定CUBLAS_WORKSPACE_CONFIG=:4096:8和PYTHONHASHSEED。
这是计算执行协议的变化。PyTorch2.1.2的严格双线性上采样采用替代实现，不能保证与普通模式的数值逐位一致，也不承诺得到历史0.8786。严格实现可能增加耗时，本轮未进行正式性能测量。

已有服务器证据：同一GPU3、相同初值和输入，普通模式首次前向一致，首次反向已有394/396组梯度张量哈希不同；严格模式前三个更新全部一致。完整20轮的重复性尚待验证。

## 执行

在仓库根目录先预览，不训练：

```bash
/home/le/miniconda3/envs/mslau-p51/bin/python tools/run_experiment.py exp00_baseline_paired_strict_v1 --seed 1234 --gpu 3
```

同一seed独立训练两次，仅repeat标签不同，顺序执行，失败即停止：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    for P51_REPEAT in 1 2; do
        P51_REPEAT="$P51_REPEAT" /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp00_baseline_paired_strict_v1 --seed 1234 --gpu 3 --run
    done
)
```

沿用本次数值对照使用的物理GPU3；可换其他可用卡，但两次应保持同一张卡。GPU占用只提示，不阻止。

## 输出与判据

沿用三个输出根目录、独立run_id和历史三份权重命名。日志在experiments/<run_id>/logs，除既有initial_state.json、input_fingerprints.jsonl、epoch_metrics.jsonl外，新增：

- numerical_runtime.json：实际确定性开关、精度/SDPA允许状态、初始CPU/CUDA随机状态。
- numerical_states.jsonl：40条训练/验证阶段末记录，含可训练参数、全部缓冲区、AdamW状态哈希、调度器状态、学习率和CPU/CUDA随机状态。
- checkpoint_tensors.json：最佳IoU及最佳Loss的完整state_dict内容哈希；比较张量内容，不能用序列化文件字节是否相同代替。

阶段末指纹只作观察，不消费随机数或更新参数。batch循环不增加GPU张量指纹；阶段末读取CPU会同步CUDA，因此新增记录的耗时不能作为公平速度基准。
启动器检查40条状态完整、严格设置始终开启且非warn_only后才标记completed。旧结果不能与新协议混作同一批次。
下一步只验证两次完整基线，不新增NAF实验；若两次一致，再把同一输入与数值协议用于后续模块对照。
