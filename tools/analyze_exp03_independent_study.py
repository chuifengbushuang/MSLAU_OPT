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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/exp03_independent_mpl')

import exp03_independent_study as driver
import local_decoder_study as common
from run_exp03_independent_study import OUT, OUTPUT, SEEDS, read, lines, locate, validate_seed, reference_records

FIG = common.WORKSPACE / 'visualization/exp03_independent_study_20261029'
LABELS = dict(baseline='B', freqselect_h='H', sgg_deep='DeepGate', exp03_independent='NAF3')
FIXED = ('cju17r8il13910799dr2wme2e.jpg', 'cju76o55nymqd0871h31sph9w.jpg')


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def records_for(seed, variant):
    if variant in driver.VARIANTS:
        return locate('train',variant,seed)
    return reference_records(seed,variant)


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
            for ref in ('baseline','freqselect_h','sgg_deep'):
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
    colors = dict(baseline='#475569',freqselect_h='#94a3b8',sgg_deep='#059669',exp03_independent='#a855f7')
    for row, seed in enumerate(SEEDS):
        for variant in LABELS:
            history = histories[(seed, variant)];epochs = [x['epoch']+1 for x in history]
            for col, (phase, metric) in enumerate((('valid','iou'),('valid','main_loss'),('train','main_loss'))):
                axes[row,col].plot(epochs,[x[phase][metric] for x in history],color=colors[variant],label=LABELS[variant])
                axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Epoch')
        axes[row,0].set_ylabel(f'seed{seed}: validation IoU')
        axes[row,1].set_ylabel('Validation Structure loss');axes[row,2].set_ylabel('Training Structure loss')
    axes[0,0].legend(ncol=3)
    fig.suptitle('Unmodified NAF3 | BF16 | paired_aug_v2 | 20epochs | matched B/H/Deep references')
    fig.savefig(FIG / 'learning_curves.png',dpi=160);plt.close(fig)


def evaluate(summary):
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from scipy.ndimage import distance_transform_edt
    import cv2
    import freqselect_study
    import sgg_residual_study
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
            factory = (driver.make_model if variant in driver.VARIANTS else
                       freqselect_study.make_model if variant=='freqselect_h' else
                       sgg_residual_study.make_model if variant=='sgg_deep' else common.make_model)
            model = factory(records/'source',args)
            checkpoint = records.parent.parent/'checkpoints'/records.name/'kvasir_best_model.pth'
            before = common.file_hash(checkpoint)
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
            model.requires_grad_(False).cuda().eval()
            rows, predictions, targets = [], [], []
            mask_outputs, handles = {}, []
            if variant == 'sgg_deep':
                for stage_name, stage in model.progressive_decoder.frequency_enhancer.stages.items():
                    def capture(module, inputs, output, name=stage_name):
                        mask_outputs[name] = output.detach().float().cpu()
                    handles.append(stage.residual_gate.register_forward_hook(capture))
            digest = paired.InputDigest()
            with torch.inference_mode(),common.autocast(args):
                for images,masks,image_ids in loader:
                    digest.update(images,masks,image_ids)
                    logits = model(images.cuda(non_blocking=True))
                    pred = logits.float().sigmoid()>.5
                    metrics = factories['compute_batch_metrics'](pred.float(),masks.cuda(non_blocking=True).unsqueeze(1))
                    pn,gn = pred[:,0].cpu().numpy(),masks.numpy()>0
                    gate_values = {}
                    for stage_name, values in mask_outputs.items():
                        fg = torch.nn.functional.interpolate(masks[:,None].float(), size=values.shape[-2:],mode='nearest')>0
                        for j in range(len(image_ids)):
                            gate_values.setdefault(j,{})
                            for region, selected in (('fg',values[j][fg[j]]),('bg',values[j][~fg[j]])):
                                gate_values[j][stage_name+'_'+region] = float(selected.mean()) if selected.numel() else None
                            gate_values[j][stage_name+'_std'] = float(values[j].std(unbiased=False))
                            gate_values[j][stage_name+'_below_one'] = float((values[j]<1).float().mean())
                    for j,image_id in enumerate(image_ids):
                        p,g = pn[j],gn[j]
                        band = np.where(g,distance_transform_edt(g),distance_transform_edt(~g))<=4
                        rows.append(dict(seed=seed,variant=variant,id=image_id,
                            **{k:v[j] for k,v in metrics.items()},**{f'{stage}_{stat}':gate_values.get(j,{}).get(f'{stage}_{stat}')
                            for stage in ('p1','p2','p3') for stat in ('fg','bg','std','below_one')},gt_area_fraction=float(g.mean()),
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
            if variant=='exp03_independent':
                diagnostics[f'{seed}/{variant}']['alpha'] = {name:float((stage.max_alpha*stage.alpha_logit.sigmoid()).item())
                    for name,stage in model.progressive_decoder.frequency_enhancer.stages.items()}
            all_rows.extend(rows)
            np.savez_compressed(OUT/f'predictions_{variant}_seed{seed}.npz',ids=np.asarray(ids),
                masks=np.asarray(predictions,dtype=np.uint8),gt=np.asarray(targets,dtype=np.uint8))
            if seed==42:
                predicted[variant]=np.asarray(predictions,dtype=bool)
            common.write_json(OUT/'evaluation_diagnostics.json',diagnostics)
            print('EVALUATED',seed,variant,json.dumps(diagnostics[f'{seed}/{variant}']),flush=True)
            for handle in handles:handle.remove()
            del model,logits,pred
            gc.collect();torch.cuda.empty_cache()
    reference = {(r['seed'],r['variant'],r['id']):r for r in all_rows}
    for row in all_rows:
        for variant in ('baseline','freqselect_h','sgg_deep'):
            row['delta_'+variant+'_iou'] = row['iou']-reference[(row['seed'],variant,row['id'])]['iou']
    csv_write(OUT/'per_image.csv',all_rows)
    csv_write(OUT/'fixed_failure_images.csv',[r for r in all_rows if r['seed']==42 and r['id'] in FIXED])
    # Same two previously selected failures, never reselect flattering examples.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(LABELS)+2,figsize=(19,6),constrained_layout=True)
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
    text=['# Current-server unmodified experiment3 independent NAF', '',
        'RTX5070Ti；Kvasir-SEG 880train/120valid；352、BF16/NAF-FP32、b12/valid6、workers2/CPU4、paired_aug_v2/严格CUDA。',
        '固定seed1234/42/2026，各20轮；官方Hiera逐次重新初始化；AdamW两组lr1e-3/weight_decay5e-4、Cosine20。',
        'Structure主/D2/D3各1；实验三原训练正则0.001，实际软mask平方按原尺度加权后(R1+R2)/4，验证不加正则。',
        '直接展开exp03_naf_p12_paired_strict_v2原始完整模型和NAF源码；仅P1/P2独立，保留90%软预算、温度0.1、alpha初值0.1/上限0.25。',
        '没有增加SGG/FreqSelect/MFDA/FDFFL/跨尺度连接，没有新模块或参数搜索。B/H/Deep复用当前服务器同seed结果，不重跑。',
        '最佳验证IoU选权重，严格重载后逐图复评；阈值FP32 sigmoid>0.5。120张验证图同用于选权重，没有独立测试集。','',
        '|seed|方案|IoU|Dice|最佳轮|末5轮IoU|末10轮IoU|可训练参数|分钟|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['runs']:
        text.append(f'|{r["seed"]}|{r["label"]}|{r["iou"]:.6f}|{r["dice"]:.6f}|{r["best_epoch"]}|{r["last5_iou"]:.6f}|{r["last10_iou"]:.6f}|{r["trainable_parameters"]}|{r["elapsed_minutes"]:.2f}|')
    text += ['', '|方案|IoU均值±样本SD|平均Dice|末5轮均值|', '|---|---:|---:|---:|']
    for v,r in summary['aggregate'].items():
        text.append(f'|{LABELS[v]}|{r["iou_mean"]:.6f}±{r["iou_sample_sd"]:.6f}|{r["dice_mean"]:.6f}|{r["last5_mean"]:.6f}|')
    text += ['', '|NAF3对照|平均配对差，百分点|差值样本SD，百分点|获胜seed数|', '|---|---:|---:|---:|']
    for ref in ('baseline','freqselect_h','sgg_deep'):
        ds=[r['best_iou_delta_pp'] for r in summary['paired'] if r['reference']==ref]
        text.append(f'|{LABELS[ref]}|{statistics.mean(ds):+.4f}|{statistics.stdev(ds):.4f}|{sum(d>0 for d in ds)}/3|')
    if diagnostics:
        text += ['', '|seed|方案|面积<5% IoU|4px带IoU|带外FP|带外FN|', '|---|---|---:|---:|---:|---:|']
        for seed in SEEDS:
            for v in LABELS:
                r=diagnostics[f'{seed}/{v}']
                text.append(f'|{seed}|{LABELS[v]}|{r["small_iou"]:.6f}|{r["band_iou"]:.6f}|{r["far_fp"]}|{r["far_fn"]}|')
    text += ['', '新增每次40条输入与旧基线一致，保留40阶段模型/缓冲/AdamW/调度器/全RNG哈希、三份完整最佳权重。',
        '新增每seed全部原层初值/CPU与完整初始RNG/runtime与已有参照核对；NAF仅新增164参数。',
        '同seed输入及环境匹配，使本机比较可解释；各方案结构/正则不同，不能将差值单独归因某个频率机制。',
        '三seed作为描述性探索；本批没有完整同seed重跑，不能以严格配置代替完整重复性证明。',
        '历史3090实验三结果另批保留，不直接与本批5070Ti结果混合平均或排名。','',
        f'![学习曲线]({FIG/"learning_curves.png"})']
    if diagnostics:text += ['',f'![固定失败图]({FIG/"fixed_failures.png"})']
    text += ['',f'数据与审计：{OUT}','']
    (common.ROOT/'studies/exp03_independent/RESULTS.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    summary=summarize();diagnostics=evaluate(summary) if args.evaluate else None
    report(summary,diagnostics)
    print('ANALYSIS_COMPLETED',json.dumps(summary['aggregate']),flush=True)


if __name__=='__main__':main()
