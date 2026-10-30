# Polyper：作者公开配置的完整 80k 对照

用户于 2026-10-30 要求先完整复现作者协议，再继续统一协议比较。本轮单次 seed=1234，不按测试成绩重试或挑种子。使用官方仓库 `haoshao-nku/medical_seg`，固定提交 `9a18e129f6f7db3018b509677ef426c67a616e0b`，直接执行该仓库 `mmsegmentation/tools/train.py` 和 `tools/test.py`。模型、损失、优化器、增强、采样器、训练循环、后处理、评测器均使用原源码。

- 数据取作者 README 引用的 PraNet 原始 ZIP，不重新随机划分。混合训练 Kvasir 900 + ClinicDB 550，Kvasir 测试 100；另保留 ClinicDB 62、ColonDB 380、CVC-300 60、ETIS 196 官方测试集。每张图片保持原始文件字节，记录档案/样本 SHA256 和集合无交集检查。
- PraNet 掩码是 1-bit PNG；OpenCV 会解码为 0/255，而 MMSeg 中 255 表示忽略。因此只把精确的二值支持集保存为 8-bit 类别 ID 0/1，不改变前景像素或几何。原始 ZIP 保留，转换明确写入数据清单。这是公开配置缺少说明的数据准备步骤。
- 继承 README 指定 `main/polyper_polypseg_224*224_80k.py`：训练 resize 到 512 范围内随机缩放和 224 裁剪，batch 6，FP32，完整 80,000 更新，无早停。Swin-T 使用相同 ImageNet 权重；验证每 8,000 步，主结果固定最终 `iter_80000.pth`，不选最佳测试轮。
- 本次按**发布主配置** AdamW lr=6e-5、wd=0.01，而论文文字是 lr=2e-4、wd=1e-4。1500 步预热、PolyLR 的 end=160000 也保留作者主配置；不擅自把调度终点改成 80k。该实验称为“公开代码/主配置复现”，不能声称消除了论文与代码的全部歧义。
- 官方测试先保持长宽比缩放到 512，再恢复原始图像尺寸，原始阈值 0.3；不开 TTA。官方 `IoUMetric` 汇总全数据像素、对背景/前景两类求平均。添加 `mIoU` 与已有 `mDice` 同时输出，不改变公式；README 的旧 `--eval mIoU` 已不被发布的工具 CLI 接受，使用现代配置参数指定同一官方评测器。另从保存的原始尺寸类别图计算前景逐图 IoU/Dice，解释与此前统一协议结果的差异。
- 新环境只读复用本机 PyTorch，其他依赖单独安装。硬件是 RTX 5070 Ti，区别于论文的 RTX 3090；环境、必要兼容修复和检查结果写入实际运行记录。新加 Hook 只记录进度、数值与初始化核验，不更新模型、调度或随机状态。

准备：`work/polyper_official_20261030/env/bin/python MSLAU_OPT/studies/polyper_official/prepare.py`。数据、环境、临时预检和正式权重在仓库外 `work/polyper_official_20261030`；审计报告在 `analysis/20261030_polyper_official`。运行前必须通过官方入口真实数据训练/评测/断点续训预检，并提交本目录。此前统一协议结果完整保留，不能用本次混合训练的数字替换同划分结果。

原始来源：[Polyper 官方源码](https://github.com/haoshao-nku/medical_seg)、[PraNet 官方数据说明](https://github.com/DengPingFan/PraNet)、[AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/view/28274)。目标值为论文表格 Kvasir mIoU 90.36%、mDice 94.82%；是否接近必须等待完整训练和最终官方评测，不能从预检或适配实验推断。
