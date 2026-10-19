"""Read-only inventory. Writes reports; never deletes or moves scanned files."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

WEIGHTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx'}
EXCLUDE = {'.git', '__pycache__', '.conda', 'miniconda3', 'papers', 'work',
           'MSLAU_OPT', 'MSLAU_OPT_original_p51', 'Kvasir-SEG', 'CVC-ClinicDB'}


def recommendation(path):
    name = path.name
    if path.suffix in WEIGHTS:
        if 'pretrained' in path.parts:
            return 'KEEP', '预训练输入，保留原路径；不参与清理'
        if 'smoke' in str(path).lower():
            return 'REVIEW_CLEANUP', '冒烟权重；确认验收记录完整且无后续引用后可清理'
        if name.startswith('best_iou_'):
            return 'VERIFY_DUPLICATE', '可能与kvasir_best_model.pth同模型；须逐tensor核对，未验证不删除'
        if name.startswith('best_loss_'):
            return 'REVIEW_KEEP', '仅在与最佳IoU不同且需要该评估/对照时保留；先核对引用'
        return 'KEEP', '最佳/其他权重默认保留；关键基线及各seed不得只留最高分一次'
    if name.endswith('.pid'):
        return 'REVIEW_CLEANUP', '历史PID记录；确认任务已结束后可清理，禁止据此直接kill进程'
    if path.suffix in {'.npy', '.npz', '.pkl'}:
        return 'REVIEW_KEEP', '较大特征缓存；确认能重建且诊断结果已归档后再评估清理'
    if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}:
        return 'REVIEW_KEEP', '保留论文/汇报图和固定样本；全量重复预测需人工筛选'
    return 'KEEP', '轻量实验/代码/诊断证据默认保留，包括负向结果'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    roots = [p.resolve() for p in args.root]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries, errors = [], {}, []
    for root in roots:
        if not root.is_dir():
            errors.append({'path': str(root), 'error': 'root does not exist or is not a directory'})
            continue
        def onerror(error):
            errors.append({'path': str(error.filename), 'error': str(error)})
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE
                             and not d.startswith('.')
                             and (Path(directory) / d).resolve() != output)
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError as error:
                    errors.append({'path': str(path), 'error': str(error)})
                    continue
                relative = path.relative_to(root)
                group = relative.parts[0] if len(relative.parts) > 1 else '(root files)'
                decision, reason = recommendation(path)
                if path.suffix in WEIGHTS and size == 0:
                    decision, reason = 'CHECK_INCOMPLETE', '零字节权重占位/同步不完整，不能当有效模型；须检查服务器原文件'
                row = dict(root=str(root), group=group, path=str(path), bytes=size,
                           is_weight=path.suffix in WEIGHTS, recommendation=decision, reason=reason)
                rows.append(row)
                summary = summaries.setdefault((str(root), group), dict(root=str(root), group=group,
                    files=0, bytes=0, weight_files=0, weight_bytes=0))
                summary['files'] += 1
                summary['bytes'] += size
                if row['is_weight']:
                    summary['weight_files'] += 1
                    summary['weight_bytes'] += size
    for name, data, fields in [('files.csv', rows,
            ['root','group','path','bytes','is_weight','recommendation','reason']),
            ('directories.csv', list(summaries.values()),
             ['root','group','files','bytes','weight_files','weight_bytes'])]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    result = dict(created_at=dt.datetime.now().astimezone().isoformat(),
        roots=list(map(str, roots)), excluded_directory_names=sorted(EXCLUDE),
        files=len(rows), observed_bytes=sum(r['bytes'] for r in rows),
        observed_weight_files=sum(r['is_weight'] for r in rows), errors=errors,
        note='No deletion/move performed; byte counts are observed logical sizes, not reclaimable disk space. '
             'Unsynced server checkpoints cannot be counted locally. Candidate labels are not deletion approval.')
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
