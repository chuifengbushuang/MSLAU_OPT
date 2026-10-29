"""Audit native-recipe records and render the requested comparison table."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
OUT=WORKSPACE/'analysis/20261030_paper_recipes'
RUNS=WORKSPACE/'work/paper_recipes_20261030'
MODELS=('polyper','eat','condseg','asgnet')
SEEDS=(1234,42,2026)


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def write(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def audit_stage(records,checkpoints,result):
    history=[json.loads(line) for line in (records/'epoch_metrics.jsonl').read_text().splitlines()]
    stop=read(records/'early_stopping.json')
    best=-float('inf');bad=0;decisions=[]
    for epoch in history:
        value=epoch['valid']['iou']
        if value>best:
            best,bad=value,0
        else:
            bad+=1
        decisions.append(bad>=10)
    if not history or len(history)>100 or any(decisions[:-1]) or (len(history)<100 and not decisions[-1]):
        raise ValueError('Maximum100/patience10 trajectory mismatch')
    if len(history)!=stop['completed_epochs'] or len(history)!=result['completed_epochs']:
        raise ValueError('Epoch count mismatch')
    if abs(best-result['evaluation']['iou'])>1e-8 or not result['strict_complete_reload']:
        raise ValueError('Best model/reload mismatch')
    if len(result['checkpoints'])!=3 or len(list(checkpoints.glob('*.pth')))!=3:
        raise ValueError('Three checkpoint names must be retained')
    for name,digest in result['checkpoints'].items():
        if file_hash(checkpoints/name)!=digest:
            raise ValueError(f'Checkpoint changed: {checkpoints/name}')
    if len((records/'input_fingerprints.jsonl').read_text().splitlines())!=2*len(history):
        raise ValueError('Missing input trajectory')
    states=[json.loads(line) for line in (records/'numerical_states.jsonl').read_text().splitlines()]
    if len(states)!=2*len(history):
        raise ValueError('Missing numerical trajectory')
    for train,valid in zip(states[::2],states[1::2]):
        if train['model_sha256']!=valid['model_sha256']:
            raise ValueError('Validation changed model state')
    per_image=read(records/'per_image.json')
    if len(per_image)!=120 or abs(statistics.mean(row['iou'] for row in per_image)-best)>1e-8:
        raise ValueError('Per-image evaluation mismatch')
    return dict(stage=result['stage'],epochs=len(history),best_epoch=result['best_epoch'],iou=best,
        checkpoints_verified=3,validation_pure=True)


def collect():
    scope=read(OUT/'scope.json');protocols=read(OUT/'protocol_differences.json')
    rows=[dict(model='baseline',seed=r['seed'],evaluation=r['evaluation'],records=r['path'],
        completed_epochs=r['completed_epochs'],best_epoch=r['best_epoch']) for r in scope['baseline_reference']]
    completed=[];audits=[];pending=[]
    gate=read(OUT/'gate.json') if (OUT/'gate.json').exists() else None
    for model in MODELS:
        for seed in SEEDS:
            candidates=list((RUNS/'experiments').glob(f'native_train_{model}_seed{seed}_*'))
            if len(candidates)>1:
                raise ValueError('Duplicate formal run; cannot select by score')
            if not candidates or not (candidates[0]/'result.json').exists():
                pending.append(dict(model=model,seed=seed));continue
            path=candidates[0];config=read(path/'config.json');result=read(path/'result.json')
            if gate is None or config['source_signature']!=gate['signature'] or config['precision']!='fp32':
                raise ValueError('Formal source or precision differs from native preflight')
            stage_audits=[]
            for stage in result['stages']:
                index=stage['stage']
                stage_audits.append(audit_stage(path/f'stage{index}',RUNS/'checkpoints'/path.name/f'stage{index}',stage))
            if model=='condseg':
                transfer=read(path/'stage2/encoder_transfer.json')
                stage1=RUNS/'checkpoints'/path.name/'stage1/kvasir_best_model.pth'
                if transfer['source_sha256']!=file_hash(stage1) or transfer['tensors']<1:
                    raise ValueError('Stage1 best encoder was not transferred')
            entry=dict(model=model,seed=seed,evaluation=result['evaluation'],records=str(path),
                completed_epochs=result['stages'][-1]['completed_epochs'],best_epoch=result['stages'][-1]['best_epoch'])
            if model=='condseg':
                entry['stage1_epochs']=result['stages'][0]['completed_epochs']
            rows.append(entry);completed.append(entry);audits.append(dict(model=model,seed=seed,stages=stage_audits))
    for baseline in scope['baseline_reference']:
        directory=WORKSPACE/'work/paper_models_20261030/runs/checkpoints'/Path(baseline['path']).name
        for name,digest in baseline['checkpoint_sha256'].items():
            if file_hash(directory/name)!=digest:
                raise ValueError('Reused baseline checkpoint changed')
    summary=[]
    for model in ('baseline',*MODELS):
        selected=[row for row in rows if row['model']==model]
        entry=dict(model=model,seeds=len(selected),author_iou_percent=protocols.get(model,{}).get('reported_iou_percent'),
            author_dice_percent=protocols.get(model,{}).get('reported_dice_percent'))
        for metric in ('iou','dice'):
            values=[row['evaluation'][metric]*100 for row in selected]
            entry[metric+'_mean_percent']=statistics.mean(values) if values else None
            entry[metric+'_sd_percent']=statistics.stdev(values) if len(values)>1 else None
        summary.append(entry)
    status='completed' if not pending else 'running'
    report=dict(status=status,runs=rows,summary=summary,protocols=protocols,pending=pending)
    write(OUT/'comparison.json',report)
    with (OUT/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write(OUT/'audit.json',dict(status='passed' if not pending else 'partial',completed_paper_runs=len(completed),
        reused_baseline_runs=3,audits=audits,pending=pending))
    return report


def render(report):
    esc=lambda value:html.escape(str(value))
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+str(cell)+'</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table>'
    def metric(row,name):
        mean,sd=row[name+'_mean_percent'],row[name+'_sd_percent']
        return '待完成' if mean is None else f'{mean:.2f}'+(f' ± {sd:.2f}' if sd is not None else '')
    def number(value):
        return '—' if value is None else f'{value:.2f}'
    rows=[[esc(row['model']),str(row['seeds'])+'/3',metric(row,'iou'),metric(row,'dice'),
        number(row['author_iou_percent'])+(' †' if row['model']=='eat' else ''),number(row['author_dice_percent'])]
        for row in report['summary']]
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>论文配方重跑对比</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182635;background:#f7f9fc}table{width:100%;border-collapse:collapse;background:white;margin:18px 0}td,th{text-align:left;border:1px solid #dbe3ec;padding:10px;vertical-align:top}th{background:#e8eff7}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.note{background:#fff4dc;padding:15px;border-left:4px solid #db9f25}a{color:#1267b5}</style>
<h1>论文原生训练配方 · Kvasir 对比</h1>'''
    document+='<p>状态：'+esc(report['status'])+'。固定 seeds：1234 / 42 / 2026。指标单位：%，± 为种子间样本标准差。</p>'
    document+='<p class="note">统一 Kvasir 880 训练 / 120 验证与报告、352×352 输入；每阶段最多 100 轮，各自早停 10 轮。验证与测试为同一清单，结果不是独立测试。四种方法恢复各自论文／官方代码训练配方；基线复用已完成的三个相同早停规则运行。</p>'
    document+=table(['方法','已完成 seeds','本机 IoU','本机 Dice','作者 IoU','作者 Dice'],rows)
    document+='<p>† EAT 数值来自作者公开表，IoU 大于 Dice 与同口径二值分割关系不符，尚未澄清，不能当作核实的 SOTA。EAT 论文全文未获得；用户已同意采用官方代码超参数并明确标注。</p>'
    document+='<h2>配方与作者协议的差异</h2>'
    differences=[]
    for model in MODELS:
        p=report['protocols'][model]
        differences.append([esc(model),esc(p.get('author_training','')),esc(p.get('our_changes','')),esc(p.get('paper_vs_code',p.get('hyperparameters_verified','')))])
    document+=table(['方法','作者实验','本机约束','来源及实现差异'],differences)
    document+='<p>精度：作者方法使用发布脚本中的 FP32；原基线保持原 BF16。ASGNet 保留全局 batch40，以单卡 4×10 累计，并保留作者 DataParallel 的首副本 BN 状态；ConDSeg 阶段一采用经数值核对的等价低显存计算顺序。数学兼容性改动、EAT 原生无效浅层分支等均记录于配方和运行源码。</p>'
    document+='<h2>逐种子结果</h2>'
    document+=table(['方法','seed','IoU','Dice','最佳轮','分割阶段轮数','预训练阶段轮数'],[
        [esc(r['model']),str(r['seed']),f"{r['evaluation']['iou']*100:.3f}",f"{r['evaluation']['dice']*100:.3f}",
         str(r['best_epoch']),str(r['completed_epochs']),str(r.get('stage1_epochs','—'))] for r in report['runs']])
    document+='<p>L2S：按用户指示等待官方代码，本批不重实现。论文表与本机表的训练数据、训练预算和评估处理不同，不作为同协议 SOTA 排名。</p>'
    document+='<p><a href="comparison.csv">CSV</a> · <a href="comparison.json">完整结果与协议</a> · <a href="audit.json">运行审计</a> · <a href="protocol_differences.json">协议来源</a></p></html>'
    temporary=OUT/'comparison.html.tmp';temporary.write_text(document);temporary.replace(OUT/'comparison.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        report=collect();render(report)
        print(json.dumps(dict(status=report['status'],completed=len(report['runs'])-3)),flush=True)
        if not args.watch or report['status']=='completed':
            return
        time.sleep(45)


if __name__=='__main__':
    main()
