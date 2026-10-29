"""Run one paper-native recipe under the user-fixed Kvasir/input/stop constraints."""
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torchvision import transforms as T

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
DEFINITION=ROOT/'studies/paper_recipes'
sys.path.insert(0,str(DEFINITION))
import recipes as native
import data as inputs
from early_stopping import EarlyStopping
import paper_model_study as legacy

OUT=WORKSPACE/'analysis/20261030_paper_recipes'
OUTPUT=WORKSPACE/'work/paper_recipes_20261030'
MODELS=tuple(native.POLICY)
SEEDS=(1234,42,2026)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def signature():
    files=[*DEFINITION.glob('*.py'),ROOT/'tools/paper_recipe_study.py',
        ROOT/'tools/run_paper_recipe_study.py',ROOT/'tests/test_paper_recipes.py',
        *list((ROOT/'studies/paper_models').glob('*.py'))]
    return {str(p.relative_to(ROOT)):legacy.common.file_hash(p) for p in sorted(files)}


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            value=value.detach().contiguous().cpu()
            digest.update(str((value.dtype,tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def runtime(seed):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(0)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')


def photo_augmentation():
    return T.Compose([T.ColorJitter(brightness=.3,contrast=.3,saturation=.3,hue=.1),
        T.RandomGrayscale(p=.5),T.GaussianBlur(kernel_size=(1,7),sigma=(.1,3))])


def move(batch):
    return {key:value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key,value in batch.items()}


def scalar_loss(model,batch,criterion,name,stage,fraction,training):
    outputs=native.forward(model,batch['image'],name,stage)
    augmented=None
    if training and name=='condseg' and stage==1:
        augmented=native.forward(model,photo_augmentation()(batch['image']),name,stage)
    loss=criterion(outputs,batch['mask'],background=batch['background'],edge=batch['edge'],
        augmented=augmented,fraction=fraction)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite native loss')
    return outputs,loss


def metric_rows(probability,mask,ids):
    # Preserve original FP32 threshold, counts, smoothing1 and image averaging.
    p=(probability>.5).float().flatten(1);g=(mask>.5).float().flatten(1)
    tp=(p*g).sum(1);ps=p.sum(1);gs=g.sum(1);total=p.shape[1]
    iou=(tp+1)/(ps+gs-tp+1);dice=(2*tp+1)/(ps+gs+1)
    precision=torch.where(ps>0,tp/ps.clamp_min(1e-7),torch.zeros_like(tp))
    recall=torch.where(gs>0,tp/gs.clamp_min(1e-7),torch.zeros_like(tp))
    f1=torch.where(precision+recall>0,2*precision*recall/(precision+recall).clamp_min(1e-7),torch.zeros_like(tp))
    return [dict(id=sample,iou=i.item(),dice=d.item(),foreground_pixels=int(s.item()),
        accuracy=((2*t+total-a-b)/total).item(),precision=pr.item(),recall=re.item(),f1=f.item())
        for sample,i,d,s,t,a,b,pr,re,f in zip(ids,iou,dice,gs,tp,ps,gs,precision,recall,f1)]


def gradient_gate(model,name,stage):
    report=legacy.gradient_report(model)
    if report['nonfinite'] or any(v is None or v<=0 for v in report['groups'].values()):
        raise RuntimeError(f'Broken gradient path: {report}')
    missing=report['missing']
    if name=='polyper':
        expected={f'decoder.refine{i}.project_out.{p}' for i in range(1,5) for p in ('weight','bias')}
    elif name=='asgnet':
        expected={f'network.SNP_{i}.module1.project_out.weight' for i in range(2,6)}
    else:
        expected=set()
    unexpected=set(missing)-expected
    if unexpected:
        raise RuntimeError(f'Unexpected disconnected parameters: {sorted(unexpected)}')
    if name=='eat':
        legacy.check_gradients(report,name)
    report['stage']=stage
    return report


def update(model,batch,opt,schedule,criterion,name,stage,fraction):
    opt.zero_grad(set_to_none=True)
    schedule.before_step()
    if name=='asgnet':
        value=native.backward_asgnet(model,batch['image'],batch['mask'],batch['edge'],criterion,fraction)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.clamp_(-.5,.5)
    else:
        _,loss=scalar_loss(model,batch,criterion,name,stage,fraction,True)
        loss.backward();value=loss.detach().item()
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError('Nonfinite gradient')
    opt.step();schedule.after_step()
    return value


@torch.no_grad()
def evaluate(model,loader,criterion,name,stage):
    model.eval();rows=[];loss_sum=0.;batches=0;digest=hashlib.sha256()
    for batch in loader:
        digest.update(state_hash({k:v for k,v in batch.items() if torch.is_tensor(v)}).encode())
        batch=move(batch)
        # ASGNet edges are training-only. Select its official final head and use
        # the final segmentation loss for deterministic validation-loss logging.
        if name=='asgnet':
            outputs=native.forward(model,batch['image'],name,stage)
            loss=criterion.official['structure_loss'](outputs[4],batch['mask'])
        else:
            outputs,loss=scalar_loss(model,batch,criterion,name,stage,1.,False)
        rows.extend(metric_rows(native.probabilities(outputs,name,stage),batch['mask'],batch['id']))
        loss_sum+=loss.item();batches+=1
    result={key:float(np.mean([r[key] for r in rows])) for key in ('iou','dice','accuracy','precision','recall','f1')}
    result.update(loss=loss_sum/batches,samples=len(rows),input_sha256=digest.hexdigest())
    return result,rows


def smoke(name,seed,records):
    stages=(1,2) if name=='condseg' else (2,)
    results=[];previous=None
    for stage in stages:
        model=native.make_model(name,stage).cuda().train()
        transfer=native.transfer_encoder(previous,model) if previous is not None else None
        if previous is not None:
            del previous;previous=None
        opt=native.optimizer(model,name,stage);schedule=native.Schedule(opt,name)
        smoke_warmup_offset=0
        if name=='eat':
            # The released schedule uses lr0 for its first TWO full epochs.
            # Disposable connectivity checks exercise the first nonzero-LR
            # phase; formal training below retains the entire native warmup.
            schedule.after_epoch(0,0.);schedule.after_epoch(1,0.)
            smoke_warmup_offset=2
        criterion=native.NativeLoss(name,stage)
        train=inputs.loader(name,True,seed,stage)
        valid=inputs.loader(name,False,seed,stage)
        iterator=iter(train);timings=[];values=[]
        torch.cuda.reset_peak_memory_stats()
        print('SMOKE_STAGE',name,stage,'batch',native.POLICY[name]['batch'],flush=True)
        for step in range(6):
            batch=move(next(iterator));torch.cuda.synchronize();started=time.perf_counter()
            value=update(model,batch,opt,schedule,criterion,name,stage,(step+1)/len(train))
            torch.cuda.synchronize();timings.append(time.perf_counter()-started);values.append(value)
            print('SMOKE_STEP',stage,step+1,value,timings[-1],flush=True)
        gradient=gradient_gate(model,name,stage)
        model.eval();batch=move(next(iter(valid)))
        with torch.no_grad():
            p1=native.probabilities(native.forward(model,batch['image'],name,stage),name,stage)
            p2=native.probabilities(native.forward(model,batch['image'],name,stage),name,stage)
            if not torch.equal(p1,p2) or not torch.isfinite(p1).all() or p1.shape!=batch['mask'].shape:
                raise RuntimeError('Smoke repeated inference/shape failed')
            path=records/f'smoke_stage{stage}.pth'
            torch.save(model.state_dict(),path)
            restored=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(restored,strict=True)
            if not torch.equal(p1,native.probabilities(native.forward(model,batch['image'],name,stage),name,stage)):
                raise RuntimeError('Full state round trip failed')
            path.unlink()
        results.append(dict(stage=stage,batch=native.POLICY[name]['batch'],steps=6,losses=values,
            median_seconds=float(np.median(timings[2:])),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            gradients=gradient,repeated_eval_equal=True,strict_complete_reload=True,transfer=transfer))
        results[-1]['disposable_smoke_warmup_offset_epochs']=smoke_warmup_offset
        write(records/'smoke_stages.json',results)
        del opt,schedule,iterator,train,valid,criterion,batch,restored
        if name=='condseg' and stage==1:
            previous=model.cpu()
        else:
            del model
        torch.cuda.empty_cache()
    result=dict(status='completed',model=name,seed=seed,precision='fp32',stages=results,signature=signature())
    write(records/'smoke.json',result);return result


def train_stage(name,seed,stage,records,checkpoints,previous=None):
    records.mkdir(parents=True,exist_ok=False);checkpoints.mkdir(parents=True,exist_ok=False)
    # The author uses separate stage scripts with a fresh seed. Decoder
    # initialization must not depend on how many epochs stage1 needed to stop.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=native.make_model(name,stage).cuda()
    if previous is not None:
        state=torch.load(previous,map_location='cpu',weights_only=True)
        selected={k:v for k,v in state.items() if k.startswith(tuple(f'network.layer{i}.' for i in range(4)))}
        own=model.state_dict()
        if set(selected)!={k for k in own if k.startswith(tuple(f'network.layer{i}.' for i in range(4)))}:
            raise RuntimeError('Stage transfer incomplete')
        model.load_state_dict(selected,strict=False)
        actual={k:model.state_dict()[k] for k in selected}
        if state_hash(selected)!=state_hash(actual):
            raise RuntimeError('Stage transfer hash mismatch')
        write(records/'encoder_transfer.json',dict(source=str(previous),source_sha256=legacy.common.file_hash(previous),
            tensors=len(selected),encoder_sha256=state_hash(actual)))
        del state,selected,actual
    write(records/'initialization.json',dict(full_state_sha256=state_hash(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters())))
    criterion=native.NativeLoss(name,stage);opt=native.optimizer(model,name,stage);schedule=native.Schedule(opt,name)
    train=inputs.loader(name,True,seed,stage);valid=inputs.loader(name,False,seed,stage)
    stop=EarlyStopping(10);best_iou=-math.inf;best_loss=math.inf;history=[]
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(100):
        model.train();train.dataset.epoch=epoch;schedule.before_epoch(epoch)
        lr_start=[g['lr'] for g in opt.param_groups];loss_sum=0.;digest=hashlib.sha256();order=hashlib.sha256()
        for step,batch in enumerate(train):
            digest.update(state_hash({k:v for k,v in batch.items() if torch.is_tensor(v)}).encode())
            order.update('\n'.join(batch['id']).encode())
            batch=move(batch)
            loss_sum+=update(model,batch,opt,schedule,criterion,name,stage,(step+1)/len(train))
            if epoch==(2 if name=='eat' else 0) and step==5:
                write(records/'training_gradients.json',gradient_gate(model,name,stage))
            if step==0 or (step+1)%25==0:
                print('TRAIN',name,seed,'stage',stage,'epoch',epoch+1,'step',step+1,'/',len(train),'loss',loss_sum/(step+1),flush=True)
        append(records/'input_fingerprints.jsonl',dict(epoch=epoch,phase='train',sha256=digest.hexdigest(),order_sha256=order.hexdigest()))
        observation=dict(epoch=epoch,phase='train',model_sha256=state_hash(model.state_dict()),
            optimizer_sha256=state_hash({f'{i}.{k}':v for i,item in enumerate(opt.state.values()) for k,v in item.items()}),schedule=schedule.state_dict())
        append(records/'numerical_states.jsonl',observation)
        metrics,rows=evaluate(model,valid,criterion,name,stage)
        if observation['model_sha256']!=state_hash(model.state_dict()):
            raise RuntimeError('Validation modified model parameters or buffers')
        append(records/'input_fingerprints.jsonl',dict(epoch=epoch,phase='valid',sha256=metrics['input_sha256']))
        schedule.after_epoch(epoch,metrics['loss'])
        append(records/'numerical_states.jsonl',dict(epoch=epoch,phase='valid',model_sha256=state_hash(model.state_dict()),schedule=schedule.state_dict()))
        if metrics['loss']<=best_loss:
            best_loss,best_loss_epoch,best_loss_iou=metrics['loss'],epoch,metrics['iou']
            torch.save(model.state_dict(),checkpoints/'running_best_loss.pth')
        if metrics['iou']>=best_iou:
            best_iou,best_epoch,best_iou_loss=metrics['iou'],epoch,metrics['loss']
            torch.save(model.state_dict(),checkpoints/'kvasir_best_model.pth')
        should_stop=stop.update(metrics['iou'])
        row=dict(epoch=epoch,stage=stage,train_loss=loss_sum/len(train),valid=metrics,
            lr_start=lr_start,lr_end=[g['lr'] for g in opt.param_groups],bad_epochs=stop.bad_epochs,
            elapsed_minutes=(time.perf_counter()-started)/60)
        history.append(row);append(records/'epoch_metrics.jsonl',row)
        write(records/'progress.json',dict(status='training',completed_epochs=epoch+1,best_iou=best_iou,best_epoch=best_epoch+1,bad_epochs=stop.bad_epochs))
        print('EPOCH',json.dumps(row),flush=True)
        if should_stop:
            break
    shutil.copy2(checkpoints/'kvasir_best_model.pth',checkpoints/f'best_iou_{best_iou:.6f}_epoch_{best_epoch}_{best_iou_loss:.6f}.pth')
    (checkpoints/'running_best_loss.pth').rename(checkpoints/f'best_loss_{best_loss:.6f}_epoch_{best_loss_epoch}_{best_loss_iou:.6f}.pth')
    write(records/'early_stopping.json',dict(maximum_epochs=100,patience=10,completed_epochs=len(history),
        bad_epochs=stop.bad_epochs,best_iou=stop.best,stopped_early=should_stop))
    state=torch.load(checkpoints/'kvasir_best_model.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);metrics,rows=evaluate(model,valid,criterion,name,stage)
    if abs(metrics['iou']-best_iou)>1e-8:
        raise RuntimeError('Best checkpoint reload changed IoU')
    write(records/'per_image.json',rows)
    result=dict(model=name,seed=seed,stage=stage,completed_epochs=len(history),best_epoch=best_epoch+1,
        evaluation=metrics,strict_complete_reload=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_minutes=(time.perf_counter()-started)/60,
        checkpoints={p.name:legacy.common.file_hash(p) for p in checkpoints.glob('*.pth')})
    write(records/'result.json',result);write(records/'progress.json',dict(status='completed',**result))
    del model,opt,schedule,train,valid,state;torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=1234)
    args=parser.parse_args(argv)
    if args.mode=='train':
        gate=read(OUT/'gate.json')
        if gate['signature']!=signature() or gate['status']!='passed' or set(gate['smokes'])!=set(MODELS):
            raise RuntimeError('All new native recipes must pass preflight before formal training')
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id=f'native_{args.mode}_{args.model}_seed{args.seed}_{tag}_{os.getpid()}'
    records=OUTPUT/'experiments'/run_id;records.mkdir(parents=True,exist_ok=False)
    print('RECORDS',str(records),flush=True)
    try:
        runtime(args.seed)
        assets=legacy.check_assets(args.model)
        original=ROOT/'experiments/exp00_baseline'
        frozen=read(original/'source_manifest.json')['files']
        if any(legacy.common.file_hash(original/key)!=digest for key,digest in frozen.items()):
            raise RuntimeError('An original frozen file changed')
        write(records/'config.json',dict(model=args.model,seed=args.seed,recipe=native.POLICY[args.model],
            fixed=read(OUT/'scope.json')['fixed'],precision='fp32',strict_cuda=True,source_signature=signature(),author_assets=assets,
            split_sha256={name:legacy.common.file_hash(inputs.SPLITS/f'{name}.txt') for name in ('train','val','test')},
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
        write(records/'frozen_originals.json',frozen)
        shutil.copytree(DEFINITION,records/'source/paper_recipes',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'studies/paper_models',records/'source/paper_models',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(__file__,records/'driver.py')
        if args.mode=='smoke':
            smoke(args.model,args.seed,records)
        else:
            stages=(1,2) if args.model=='condseg' else (2,)
            previous=None;results=[]
            for stage in stages:
                directory=OUTPUT/'checkpoints'/run_id/f'stage{stage}'
                results.append(train_stage(args.model,args.seed,stage,records/f'stage{stage}',directory,previous))
                previous=directory/'kvasir_best_model.pth'
            write(records/'result.json',dict(model=args.model,seed=args.seed,stages=results,
                evaluation=results[-1]['evaluation'],status='completed'))
    except BaseException as exc:
        write(records/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
