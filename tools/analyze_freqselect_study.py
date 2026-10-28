"""Audit all fixed seeds; infer best checkpoints without updates; write comparisons."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/freqselect_mpl')

import freqselect_study as driver
import local_decoder_study as common
from run_freqselect_study import OUT, OUTPUT, SEEDS, read, lines, locate, validate_seed

FIG = common.WORKSPACE / 'visualization/freqselect_study_20261028'
LABELS = dict(baseline='B', fredft_frequency='F', fredft_spatial='S',
              freqselect_h='H', freqselect_g='G', freqselect_fs='FS')
FIXED = ('cju17r8il13910799dr2wme2e.jpg', 'cju76o55nymqd0871h31sph9w.jpg')


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def records_for(seed, variant):
    if variant in driver.VARIANTS:
        return locate('train', variant, seed)
    matches = [p for p in (common.WORKSPACE / 'experiments').glob(f'local_train_{variant}_seed{seed}_*')
               if (p / 'result.json').is_file()]
    if len(matches) != 1:
        raise RuntimeError((seed, variant, matches))
    return matches[0]


def summarize():
    audits = [validate_seed(seed) for seed in SEEDS]
    rows, histories = [], {}
    for seed in SEEDS:
        for variant in LABELS:
            records = records_for(seed, variant)
            result, initial = read(records / 'result.json'), read(records / 'initial_state.json')
            history = lines(records / 'epoch_metrics.jsonl')
            row = dict(seed=seed, variant=variant, label=LABELS[variant],
                **{k:result['evaluation'][k] for k in ('iou','dice','precision','recall')},
                best_epoch=result['best_epoch'] + 1,
                last5_iou=statistics.mean(x['valid']['iou'] for x in history[-5:]),
                last10_iou=statistics.mean(x['valid']['iou'] for x in history[-10:]),
                trainable_parameters=initial['trainable_parameters'], elapsed_minutes=result['elapsed_minutes'],
                records=str(records), checkpoint=str(records.parent.parent / 'checkpoints' / records.name / 'kvasir_best_model.pth'))
            rows.append(row);histories[(seed, variant)] = history
    aggregates = {}
    for variant in LABELS:
        selected = [r for r in rows if r['variant'] == variant]
        aggregates[variant] = dict(iou_mean=statistics.mean(r['iou'] for r in selected),
            iou_sample_sd=statistics.stdev(r['iou'] for r in selected),
            dice_mean=statistics.mean(r['dice'] for r in selected),
            last5_mean=statistics.mean(r['last5_iou'] for r in selected))
    paired = []
    for seed in SEEDS:
        values = {r['variant']:r for r in rows if r['seed'] == seed}
        for variant in driver.VARIANTS:
            for ref in ('baseline','fredft_frequency','fredft_spatial','freqselect_h','freqselect_g'):
                if variant == ref:
                    continue
                paired.append(dict(seed=seed, variant=variant, reference=ref,
                    best_iou_delta_pp=100*(values[variant]['iou']-values[ref]['iou']),
                    last5_delta_pp=100*(values[variant]['last5_iou']-values[ref]['last5_iou'])))
    summary = dict(status='completed', seeds=list(SEEDS), independent_test_set=False,
        selection='Best validation per-image IoU under fixed20epochs, no seed selection',
        runs=rows, aggregate=aggregates, paired=paired, audits=audits,
        smoke={v:read(locate('smoke',v,1234)/'smoke.json') for v in driver.VARIANTS})
    common.write_json(OUT / 'summary.json', summary)
    csv_write(OUT / 'runs.csv', rows);csv_write(OUT / 'paired_differences.csv', paired)
    plot_curves(histories)
    return summary


def plot_curves(histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    colors = dict(baseline='#475569', fredft_frequency='#dc2626', fredft_spatial='#a855f7',
                  freqselect_h='#94a3b8', freqselect_g='#2563eb', freqselect_fs='#059669')
    for row, seed in enumerate(SEEDS):
        for variant in LABELS:
            history = histories[(seed, variant)];epochs = [x['epoch']+1 for x in history]
            for col, (phase, metric) in enumerate((('valid','iou'),('valid','main_loss'),('train','main_loss'))):
                axes[row,col].plot(epochs,[x[phase][metric] for x in history],color=colors[variant],label=LABELS[variant])
                axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Epoch')
        axes[row,0].set_ylabel(f'seed{seed}: validation IoU')
        axes[row,1].set_ylabel('Validation Structure loss');axes[row,2].set_ylabel('Training Structure loss')
    axes[0,0].legend(ncol=3)
    fig.suptitle('FreqSelect controlled study | BF16 | paired_aug_v2 | 20epochs | six matched models')
    fig.savefig(FIG / 'learning_curves.png',dpi=160);plt.close(fig)


def evaluate(summary):
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from scipy.ndimage import distance_transform_edt
    import cv2
    import fredft_study
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    cv2.setNumThreads(0)
    source = driver.baseline_records(1234) / 'source'
    sys.path.insert(0,str(source))
    import paired_inputs as paired
    factories = common.definitions(source/'kvasir_train.py',{'load_ids','get_valid_transform'},dict(A=A,ToTensorV2=ToTensorV2,os=os))
    factories.update(common.definitions(source/'kvasir_test.py',{'compute_batch_metrics'},dict(torch=torch)))
    ids = factories['load_ids'](str(source/'configs/splits/kvasir/val.txt'))
    all_rows, diagnostics, predicted = [], {}, {}
    for seed in SEEDS:
        dataset = paired.PairedDataset(str(common.WORKSPACE/'datasets/Kvasir-SEG'), ids,
                                      factories['get_valid_transform'](352),seed,'valid')
        loader = paired.make_loader(dataset,6,2,pin_memory=True)
        for variant in LABELS:
            records = records_for(seed,variant)
            args = argparse.Namespace(**read(records/'arguments.json'));args.hiera_checkpoint=Path(args.hiera_checkpoint)
            factory = driver.make_model if variant in driver.VARIANTS else (fredft_study.make_model if variant.startswith('fredft_') else common.make_model)
            model = factory(records/'source',args)
            checkpoint = records.parent.parent/'checkpoints'/records.name/'kvasir_best_model.pth'
            before = common.file_hash(checkpoint)
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
            model.requires_grad_(False).cuda().eval()
            rows, predictions, targets = [], [], []
            digest = paired.InputDigest()
            with torch.inference_mode(),common.autocast(args):
                for images,masks,image_ids in loader:
                    digest.update(images,masks,image_ids)
                    logits = model(images.cuda(non_blocking=True))
                    pred = logits.float().sigmoid()>.5
                    metrics = factories['compute_batch_metrics'](pred.float(),masks.cuda(non_blocking=True).unsqueeze(1))
                    pn,gn = pred[:,0].cpu().numpy(),masks.numpy()>0
                    for j,image_id in enumerate(image_ids):
                        p,g = pn[j],gn[j]
                        band = np.where(g,distance_transform_edt(g),distance_transform_edt(~g))<=4
                        rows.append(dict(seed=seed,variant=variant,id=image_id,
                            **{k:v[j] for k,v in metrics.items()},gt_area_fraction=float(g.mean()),
                            false_positive=int((p&~g).sum()),false_negative=int((~p&g).sum()),
                            false_positive_far=int((p&~g&~band).sum()),false_negative_far=int((~p&g&~band).sum()),
                            errors_4px_band=int(((p!=g)&band).sum()),
                            band_iou=(np.count_nonzero(p&g&band)+1)/(np.count_nonzero((p|g)&band)+1)))
                    predictions.extend(pn);targets.extend(gn)
            expected = next(x for x in lines(records/'input_fingerprints.jsonl') if x['epoch']==0 and x['phase']=='valid')
            assert digest.record(seed,0,'valid') == expected
            values = {k:statistics.mean(r[k] for r in rows) for k in ('iou','dice','precision','recall')}
            original = read(records/'evaluation.json')
            assert all(abs(values[k]-original[k])<1e-6 for k in values), (seed,variant,values,original)
            assert common.file_hash(checkpoint)==before
            selected = [r for r in rows if r['gt_area_fraction']<.05]
            diagnostics[f'{seed}/{variant}'] = dict(**values,small_n=len(selected),
                small_iou=statistics.mean(r['iou'] for r in selected),
                band_iou=statistics.mean(r['band_iou'] for r in rows),
                far_fp=sum(r['false_positive_far'] for r in rows),far_fn=sum(r['false_negative_far'] for r in rows),
                weights_unchanged=True,original_metrics_reproduced=True,inputs_reproduced=True)
            all_rows.extend(rows)
            np.savez_compressed(OUT/f'predictions_{variant}_seed{seed}.npz',ids=np.asarray(ids),
                masks=np.asarray(predictions,dtype=np.uint8),gt=np.asarray(targets,dtype=np.uint8))
            if seed==42:
                predicted[variant]=np.asarray(predictions,dtype=bool)
            common.write_json(OUT/'evaluation_diagnostics.json',diagnostics)
            print('EVALUATED',seed,variant,json.dumps(diagnostics[f'{seed}/{variant}']),flush=True)
            del model,logits,pred
            gc.collect();torch.cuda.empty_cache()
    reference = {(r['seed'],r['variant'],r['id']):r for r in all_rows}
    for row in all_rows:
        for variant in ('baseline','freqselect_h','freqselect_g'):
            row['delta_'+variant+'_iou'] = row['iou']-reference[(row['seed'],variant,row['id'])]['iou']
    csv_write(OUT/'per_image.csv',all_rows)
    csv_write(OUT/'fixed_failure_images.csv',[r for r in all_rows if r['seed']==42 and r['id'] in FIXED])
    # Same two previously selected failures, never reselect flattering examples.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,8,figsize=(20,6),constrained_layout=True)
    for row,image_id in enumerate(FIXED):
        index=ids.index(image_id)
        image=cv2.cvtColor(cv2.resize(cv2.imread(str(common.WORKSPACE/'datasets/Kvasir-SEG/images'/image_id)),(352,352)),cv2.COLOR_BGR2RGB)
        gt=np.load(OUT/'predictions_baseline_seed42.npz')['gt'][index].astype(bool)
        axes[row,0].imshow(image);axes[row,1].imshow(gt,cmap='gray')
        for col,variant in enumerate(LABELS,2):
            p=predicted[variant][index];overlay=np.zeros((352,352,3),dtype=np.uint8)
            overlay[p&gt]=(230,230,230);overlay[p&~gt]=(245,90,70);overlay[~p&gt]=(60,140,245)
            axes[row,col].imshow(overlay)
            axes[row,col].set_title(f'{LABELS[variant]} IoU={reference[(42,variant,image_id)]["iou"]:.3f}')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(image_id,fontsize=7)
    axes[0,0].set_title('Input');axes[0,1].set_title('GT')
    fig.suptitle('Fixed seed42 failures | white TP / red FP / blue FN')
    fig.savefig(FIG/'fixed_failures.png',dpi=150);plt.close(fig)
    return diagnostics


def report(summary, diagnostics=None):
    text = ['# FreqSelect H/G/FS: three-seed controlled study','',
        '当前5070Ti，固定seed1234/42/2026，各20轮；所有方案保留原Decoder。',
        '输入/增强/精度/损失/优化器/调度器与本机基线匹配；各次从官方Hiera重新初始化。',
        'H为仅外围融合，G为同参数空间门控，FS为分频+空间门控；F/S为已完成FreDFT参考。',
        '主指标最佳验证逐图IoU，完整权重严格重载复评；120张验证图同时用于选权重，并非独立测试。','',
        '|seed|方案|IoU|Dice|最佳轮|末5轮IoU|末10轮IoU|可训练参数|分钟|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['runs']:
        text.append(f'|{r["seed"]}|{r["label"]}|{r["iou"]:.6f}|{r["dice"]:.6f}|{r["best_epoch"]}|{r["last5_iou"]:.6f}|{r["last10_iou"]:.6f}|{r["trainable_parameters"]}|{r["elapsed_minutes"]:.2f}|')
    text += ['', '|方案|三seed IoU均值±样本SD|平均Dice|', '|---|---:|---:|']
    for v,r in summary['aggregate'].items():
        text.append(f'|{LABELS[v]}|{r["iou_mean"]:.6f}±{r["iou_sample_sd"]:.6f}|{r["dice_mean"]:.6f}|')
    text += ['', '|FS对照|平均配对差，百分点|差值样本SD，百分点|获胜seed数|','|---|---:|---:|---:|']
    for ref in ('baseline','freqselect_h','freqselect_g','fredft_frequency','fredft_spatial'):
        ds=[r['best_iou_delta_pp'] for r in summary['paired'] if r['variant']=='freqselect_fs' and r['reference']==ref]
        text.append(f'|{LABELS[ref]}|{statistics.mean(ds):+.4f}|{statistics.stdev(ds):.4f}|{sum(d>0 for d in ds)}/3|')
    if diagnostics:
        text += ['', '|seed|方案|面积<5% IoU|4px带IoU|带外FP|带外FN|','|---|---|---:|---:|---:|---:|']
        for seed in SEEDS:
            for v in LABELS:
                r=diagnostics[f'{seed}/{v}']
                text.append(f'|{seed}|{LABELS[v]}|{r["small_iou"]:.6f}|{r["band_iou"]:.6f}|{r["far_fp"]}|{r["far_fn"]}|')
    text += ['', '所有配对结果完整保留；三seed为描述性探索，未证明统计显著性或独立数据泛化。',
        'FS−G检验分频的额外作用；FS−H包含增加门控的作用。原NAF比较另含移除选择器正则等变化。',
        '各组40条输入与对应旧基线一致；G/FS完整初值相同；H/G/FS共享层和初始全RNG/runtime匹配。',
        '每次保留三份完整最佳权重及40条阶段状态；本批未追加完整同seed重复训练，不将确定性设置冒充完整复现证明。',
        '耗时包含阶段哈希观测；显存与速度见smoke记录，不据历史训练用时推断硬件因果。',
        '未运行AdaKern/AdaDR、F_eq、40轮、额外seed或阈值调参。', '',
        f'![学习曲线]({FIG / "learning_curves.png"})']
    if diagnostics:text += ['',f'![固定失败图]({FIG / "fixed_failures.png"})']
    text += ['', f'数据与审计：{OUT}', '']
    (common.ROOT/'studies/freqselect/RESULTS.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    summary=summarize();diagnostics=evaluate(summary) if args.evaluate else None
    report(summary,diagnostics)
    print('ANALYSIS_COMPLETED',json.dumps(summary['aggregate']),flush=True)


if __name__=='__main__':main()
