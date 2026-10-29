"""Render measured Kvasir scores beside author reports and protocol differences."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics

import paper_model_study as driver


def analyze():
    driver.OUT.mkdir(parents=True,exist_ok=True)
    protocol = driver.read(driver.DEFINITION/'protocol.json')
    rows,runs = [],[]
    for model in driver.VARIANTS:
        chosen = []
        for seed in driver.SEEDS:
            matches = list((driver.OUTPUT/'experiments').glob(f'paper_train_{model}_seed{seed}_*'))
            if len(matches)>1:
                raise RuntimeError(f'Never select among duplicate runs: {model}/{seed}')
            if not matches or not (matches[0]/'result.json').is_file():
                continue
            path = matches[0]
            result,stop = driver.read(path/'result.json'),driver.read(path/'early_stopping.json')
            if result['status']!='completed' or (path/'failure.json').exists():
                continue
            run = dict(model=model,seed=seed,path=str(path),best_epoch=result['best_epoch']+1,
                epochs=stop['completed_epochs'],iou=100*result['evaluation']['iou'],
                dice=100*result['evaluation']['dice'],precision=100*result['evaluation']['precision'],
                recall=100*result['evaluation']['recall'],elapsed_minutes=result['elapsed_minutes'],
                **driver.read(path/'initial_state.json'))
            chosen.append(run);runs.append(run)
        published = protocol['reported'].get(model,{})
        rows.append(dict(model=model,status='completed' if len(chosen)==3 else f'{len(chosen)}/3 completed',
            n_seeds=len(chosen),iou_mean=statistics.mean(x['iou'] for x in chosen) if chosen else None,
            iou_sample_sd=statistics.stdev(x['iou'] for x in chosen) if len(chosen)>1 else None,
            dice_mean=statistics.mean(x['dice'] for x in chosen) if chosen else None,
            dice_sample_sd=statistics.stdev(x['dice'] for x in chosen) if len(chosen)>1 else None,
            author_iou=published.get('iou'),author_dice=published.get('dice'),
            epochs='/'.join(str(x['epochs']) for x in chosen),
            best_epochs='/'.join(str(x['best_epoch']) for x in chosen),
            source=published.get('source','Original frozen MSLAU baseline'),
            author_protocol=published.get('protocol','Original 880/120 protocol, originally 20 epochs'),
            adaptation=protocol['head_mapping'].get(model,'Original complete frozen baseline network'),
            our_protocol='Kvasir 880/120 shared val/test; 352; paired_aug_v2; batch12 BF16; AdamW .001; cosine100; patience10; selected-set foreground image-mean IoU'))
    old = [driver.read(driver.baseline_records(s)/'evaluation.json') for s in driver.SEEDS]
    rows.append(dict(model='baseline_history_20epochs',status='historical reference',n_seeds=3,
        iou_mean=100*statistics.mean(x['iou'] for x in old),iou_sample_sd=100*statistics.stdev(x['iou'] for x in old),
        dice_mean=100*statistics.mean(x['dice'] for x in old),dice_sample_sd=100*statistics.stdev(x['dice'] for x in old),
        author_iou=None,author_dice=None,epochs='20/20/20',best_epochs='',source='Cached matched local baseline runs',
        author_protocol='Historical 20-epoch, cosine20 baseline; budget differs from current study',
        adaptation='No changes',our_protocol='Same data/input/metric, historical 20-epoch schedule'))
    rows.append(dict(model='l2s',status='deferred by user: official code unavailable',n_seeds=0,
        iou_mean=None,iou_sample_sd=None,dice_mean=None,dice_sample_sd=None,
        author_iou=87.02,author_dice=91.87,epochs='',best_epochs='',
        source=protocol['reported']['l2s']['source'],author_protocol=protocol['reported']['l2s']['protocol'],
        adaptation='Not substituted with a different architecture',our_protocol='Not run'))
    # Cross-model input equality for every observed overlapping epoch, including
    # all post-20 epochs. Original history independently anchors the first 20.
    observed={};matched=0
    for run in runs:
        for text in (Path(run['path'])/'input_fingerprints.jsonl').read_text().splitlines():
            value=json.loads(text);key=(run['seed'],value['epoch'],value['phase'])
            if key in observed:
                if value!=observed[key]:raise RuntimeError(f'Cross-model inputs differ: {key}')
                matched+=1
            else:observed[key]=value
    complete = all(r['n_seeds']==3 for r in rows if r['model'] in driver.VARIANTS)
    driver.common.write_json(driver.OUT/'comparison.json',dict(status='completed' if complete else 'running',
        protocol=protocol,rows=rows,runs=runs,matched_cross_model_phase_inputs=matched,
        metric_scope='The 120 selected-on images are NOT an independent test set. Author results are different protocols, not a unified leaderboard.'))
    with (driver.OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def number(x):return '—' if x is None else f'{x:.2f}'
    def pair(row,key):
        value=number(row[key+'_mean']);sd=row[key+'_sample_sd']
        return value if sd is None else f'{value} ± {sd:.2f}'
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (r['model'],r['status'],pair(r,'iou'),pair(r,'dice'),number(r['author_iou']),number(r['author_dice']),
         r['epochs'],r['best_epochs'],r['author_protocol'],r['adaptation']))+'</tr>' for r in rows)
    output='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Kvasir 模型对照</title>
<style>body{font:15px system-ui;margin:32px;color:#18212b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dc;padding:10px;text-align:left}th{background:#edf3f8}td:nth-child(n+3):nth-child(-n+6){white-space:nowrap}p{max-width:1000px;line-height:1.7}</style>
<h1>Kvasir 880/120 统一协议模型对照</h1><p>最多100轮，连续10轮前景逐图IoU无严格提升早停。120张同时用于验证、选权与报告，属于选权集结果，不是独立测试。输入352×352，batch12，BF16，原始增强、Structure损失、AdamW和阈值。均值±样本标准差；三个种子固定1234/42/2026。</p>
<p>作者报告值来自不同训练划分和指标。EAT作者公开表IoU97.85高于Dice97.09，需澄清口径；本表不将其视为已验证目标。L2S按用户指示等待官方实现。</p><table><thead><tr>'''
    headings=['模型','状态','本协议 IoU %','本协议 Dice %','作者 IoU %','作者 Dice %','实际轮数','最佳轮数','作者协议差异','本次输出适配']
    output+=''.join('<th>'+html.escape(x)+'</th>' for x in headings)+'</tr></thead><tbody>'+table+'</tbody></table>'
    output+='<p>模型使用各自公开预训练骨干；baseline为SAM2 Hiera-L，其余为ImageNet Swin-T/PVT/ResNet。结果衡量统一配方下的适配网络，不代表原论文完整训练流程的复现。</p></html>'
    (driver.OUT/'comparison.html').write_text(output)
    return dict(status='completed' if complete else 'running',completed_runs=len(runs),required_runs=15)


if __name__=='__main__':
    print(json.dumps(analyze()))
