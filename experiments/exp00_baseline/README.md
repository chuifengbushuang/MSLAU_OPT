# 实验0：未经修改的原始P5.1

来源：原分支`kvasir-p5-sam2unet-official-adapter`，提交`81b19e2`。
`source_manifest.json`记录14份原文件SHA256；新增文件仅run.py、config.json、本说明与manifest。
run.py每次启动先验证原文件未改变。代码保留原分支内其他模式定义，不在原文件中裁剪，避免改变实验0。

固定克隆环境mslau-p51；原环境与副本运行时及pip freeze一致已由用户核对。
新基线在三个seed各运行一次，不复用旧环境结果或挑选历史最高重复。
除路径与seed外，训练CLI沿用原实验者启动命令。特别不额外加入NAF、增强seed控制或新损失。
原脚本未显式设置Albumentations内部seed，运行仍可能波动，后续用匹配多seed比较。

从仓库根目录使用tools/run_experiment.py exp00_baseline，默认临时展开和预览，加--run才训练。
后续实验保存相对本基线的patch与added文件，构建和导出不修改本目录。
展开目录含完整源码，可脱离Git和其他实验独立运行。路径和命令见根README。

每次保留原脚本的固定名称最佳IoU、带指标名称最佳IoU及最佳Loss三份权重，不做训练后删减。
