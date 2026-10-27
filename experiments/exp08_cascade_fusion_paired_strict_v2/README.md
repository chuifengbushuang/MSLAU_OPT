# 实验8：仅级联融合，同协议v2对照

对应历史实验8的同协议重测，不是新架构，也不覆盖原实验8。
直接从冻结基线81b19e2构建；模型文件与旧实验8逐字节相同。

- F3=Fusion3(P3,P4)，F2=Fusion2(P2,F3)，F1=Fusion1(P1,F2)；原Decoder不变。
- 保留投影、GAP+centered RMS两路通道门控和外部残差；无FFT、选择器、内部alpha或正则。
- 新增55,728参数。与实验9共享的投影/门控/残差层初始化完全相同，不消费全局初始化RNG。
- 输入采用paired_aug_v2；采样、增强、worker及验证处理与v2基线一致。
- strict_cuda_v1用于训练和独立评估，严格检查失败就停止，不降级为警告。
- 352/b12/nw8/e20、原Structure三头监督、AdamW/两组LR1e-3和Cosine均不变。
- 新增baseline_state_sha256记录，剔除新模块后应等于同seed基线完整初始状态哈希。
- 完整输入/逐阶段数值状态和三份历史命名权重继续保留，输出根目录不变。

本轮只需将现有v2三seed基线与本实验、实验9比较，基线不重跑。
基线→本实验衡量级联融合收益；本实验→实验9衡量频率选择与分段正则的组合收益，不能据此单独归因于频率选择。

服务器先拉取代码，然后执行两组新对照的严格GPU合成检查：

```bash
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1 STRICT_NAF_FULL_SMOKE=1 /home/le/miniconda3/envs/mslau-p51/bin/python -u -m unittest discover -s tests -p 'test_strict_naf_controls.py' -v
```

该检查使用官方Hiera与随机合成输入，默认batch12，两个模型分别独立初始化两次、每次两步更新并评估，核对重复性及有限梯度；只在临时目录保存小型CPU测试权重，不用原数据、不保存正式模型、不启动真实训练。

正式六次运行见[实验9同协议说明](../exp09_naf_discrete_paired_strict_v2/README.md)。
