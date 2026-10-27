# 实验0 v2：修复逐样本重设种子引入的增强联动

v1通过了重复性检查，但并未保持原增强的联合分布。Albumentations1.4.24的Compose.set_random_seed(s)给各子变换分别创建相同起点的随机流；每个样本都重设时，p=0.25的水平翻转、垂直翻转、ShiftScaleRotate会一起触发或跳过，且触发时p=0.5的CoarseDropout也触发。
这是旧输入协议的实现遗漏，受影响的是paired_aug_v1配对基线、严格v1和实验9；不修改或删除旧定义/记录。原实验0–8未使用这一逐样本重设方法。

本地真实Albumentations1.4.24、4096次合成增强：水平与垂直同时触发比例，原始未固定方式6.714%、v1为24.805%、v2为6.104%，独立概率理论值6.25%。三种几何增强同时触发理论值1.5625%、v1为24.805%、v2为1.489%。四种随机变换的开关组合，v1仅3种，原始与v2均16种。

v2仍按(seed,phase,epoch,sample_id)确定每个样本的增强，改为公开API：

```python
transform.set_random_state(np.random.default_rng(seed), random.Random(seed))
```

Compose与子变换共享同一条依次推进的私有Python/NumPy随机流，各变换取后续随机数，不再分别从同一个首数开始。图像和mask仍在原来的同一次Compose调用中同步处理。
该API直接设置随机状态，Compose.seed属性可能仍为None；是否可重复应核对实际输入哈希，不能仅检查这个属性。
input_protocol标记为paired_aug_v2，底层SEED_NAMESPACE保留paired_aug_v1，确保采样顺序、worker种子和样本根种子与v1完全相同。只改变增强内部随机流的组织方式；验证图像、模型初始化、原352/b12/nw8/e20训练配方及strict_cuda_v1不变。changes.patch和strict_runtime.py与严格v1逐字节相同。

这修复了已确认的分布问题，不保证IoU必然提高。此前普通模式的同一v1输入也得到过0.863–0.875，故不能把严格模式0.853762的全部下降归因于这个缺陷。历史那次未记录的增强序列无法从全局seed恢复。

## 服务器检查和最小对照

不改Conda环境。先做真实数据的CPU输入审计，检查独立进程、0/8worker、epoch/seed变化及4096次合成增强的联合触发概率；通过后只跑seed1234一次20轮，与已有严格v1比较：

```bash
(
    set -e
    export P51_OUTPUT_ROOT=/data/models/le/mslau-net
    /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/check_paired_inputs.py --experiment exp00_baseline_paired_strict_v2 --seed 1234 --workers 8 --epochs 2
    P51_REPEAT=1 /home/le/miniconda3/envs/mslau-p51/bin/python -u tools/run_experiment.py exp00_baseline_paired_strict_v2 --seed 1234 --gpu 3 --run
)
```

CPU报告为experiments/paired_aug_v2_input_check_*.json。训练继续按三个根目录和原三权重命名保存；记录目录为experiments/exp00_baseline_paired_strict_v2_seed1234_*_r1。代理不代启动训练。
先观察修复后的同种子结果，再补v2的42/2026，不能挑选最高的一种增强随机实现充当基线。旧v1尚未启动的补训暂缓；已完成结果保留为旧协议参考。
