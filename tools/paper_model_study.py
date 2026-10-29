"""Controlled Kvasir 880/120 study: fixed baseline inputs and author networks."""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')

import local_decoder_study as common
import numpy as np
import torch
from torch import nn

ROOT,WORKSPACE = common.ROOT,common.WORKSPACE
DEFINITION = ROOT/'studies/paper_models'
ASSETS = WORKSPACE/'work/paper_models_20261030'
OUTPUT = ASSETS/'runs'
OUT = WORKSPACE/'analysis/20261030_paper_models'
VARIANTS = ('baseline','polyper','eat','condseg','asgnet')
SEEDS = (1234,42,2026)
BASELINES = {
 1234:'local_train_baseline_seed1234_20261028_004839_75754',
 42:'local_train_baseline_seed42_20261028_111330_218272',
 2026:'local_train_baseline_seed2026_20261028_114725_227920'}
ORIGINAL_LOSSES = common.losses
ORIGINAL_APPEND = common.append_json
AUTHOR_COMMITS = dict(polyper='9a18e129f6f7db3018b509677ef426c67a616e0b',
 eat='c699f7e5e5f457d7fd5789f1e309e380048142f8',
 condseg='b4c22c399e72ec858026abc0e87143f3c53fe12d',
 asgnet='97c4e746e94bb6a184a14775b771226eed88c3e2')


def read(path):
    return json.loads(Path(path).read_text())


def baseline_records(seed):
    return WORKSPACE/'experiments'/BASELINES[seed]


def encoder_parameters(model):
    encoder = model.encoder
    modules = encoder if isinstance(encoder,tuple) else (encoder,)
    return [p for module in modules for p in module.parameters() if p.requires_grad]


def make_optimizer(model):
    encoder = encoder_parameters(model)
    ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    if len(ids)+len(decoder) != len(list(p for p in model.parameters() if p.requires_grad)):
        raise RuntimeError('Optimizer omits or duplicates a trainable parameter')
    return torch.optim.AdamW([dict(params=encoder,lr=.001),dict(params=decoder,lr=.001)],weight_decay=.0005)


def gradient_report(model):
    ids = {id(p) for p in encoder_parameters(model)}
    groups = dict(encoder=[],decoder=[])
    missing,nonfinite = [],[]
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            nonfinite.append(name)
        groups['encoder' if id(p) in ids else 'decoder'].append(p.grad.detach().abs().max())
    # Report unused author parameters explicitly; do not fabricate auxiliary heads
    # or silently delete dead branches from released architecture definitions.
    deformable={}
    for name,module in model.named_modules():
        if hasattr(module,'offset_mask') and hasattr(module,'group_channels'):
            gradient=module.offset_mask.weight.grad
            if gradient is None:
                deformable[name]=dict(coordinates=None,mask=None)
            else:
                k=module.kernel_size**2-module.remove_center
                active=gradient[:module.group*k*3].reshape(module.group,3*k,-1)
                deformable[name]=dict(coordinates=active[:,:2*k].abs().max().item(),mask=active[:,2*k:].abs().max().item())
    return dict(groups={k:torch.stack(v).max().item() if v else None for k,v in groups.items()},
                missing=missing,nonfinite=nonfinite,deformable=deformable)


def check_gradients(report,variant):
    if report['nonfinite'] or any(v is None or v <= 0 for v in report['groups'].values()):
        raise RuntimeError(f'Broken segmentation gradient path: {report}')
    def intentional(name):
        if variant=='polyper':
            return name in {f'decoder.refine{i}.project_out.{p}' for i in range(1,5) for p in ('weight','bias')}
        if variant=='condseg':
            return name.startswith(('network.decouple_layer.cbr_uc.','network.aux_head.branch_uc.'))
        if variant=='asgnet':
            return name in ({f'network.SNP_{i}.module1.project_out.weight' for i in range(2,6)}|
                {f'network.DCI_{i}.out_edge.0.{p}' for i in range(1,5) for p in ('weight','bias')})
        return False
    unexpected=[name for name in report['missing'] if not intentional(name)]
    if unexpected:raise RuntimeError(f'Unexpected disconnected model parameters: {unexpected}')
    if variant=='eat' and (not report['deformable'] or any(v is None or v<=0
            for values in report['deformable'].values() for v in values.values())):
        raise RuntimeError(f'DCNv4 coordinates or masks cannot learn: {report["deformable"]}')


def fixed_losses(outputs,labels,criterion,variant,training):
    return ORIGINAL_LOSSES(outputs,labels,criterion,'baseline',training)


def check_assets(variant):
    if variant == 'baseline':
        return {}
    directory = ASSETS/'vendor'/variant
    pin = read(DEFINITION/'upstreams.json')[variant]
    if common.file_hash(ASSETS/pin['weights']) != pin['weight_sha256']:
        raise RuntimeError(f'Pretrained weights changed: {variant}')
    manifest = read(directory/'upstream_manifest.json')
    if manifest['commit'] != AUTHOR_COMMITS[variant]:
        raise RuntimeError('Unexpected author revision')
    selected = {}
    for name,digest in manifest['files'].items():
        if name.endswith('.py') or name.endswith('.pth') or Path(name).name.startswith('LICENSE'):
            if common.file_hash(directory/name) != digest:
                raise RuntimeError(f'Author snapshot changed: {variant}/{name}')
            selected[name] = digest
    return dict(repo=manifest['repo'],commit=manifest['commit'],files=selected,
                pretrained_weights=pin['weights'],pretrained_sha256=pin['weight_sha256'])


def snapshot_assets(variant,records):
    evidence = check_assets(variant)
    target = records/'source/paper_models'
    shutil.copytree(DEFINITION,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if variant == 'baseline':
        return target,evidence
    assets = records/'source/paper_assets'
    (assets/'vendor'/variant).mkdir(parents=True)
    for name in evidence['files']:
        origin = ASSETS/'vendor'/variant/name
        destination = assets/'vendor'/variant/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.pth'):
            destination.symlink_to(origin)
        else:
            shutil.copy2(origin,destination)
    (assets/'weights').symlink_to(ASSETS/'weights',target_is_directory=True)
    (assets/'deps').symlink_to(ASSETS/'deps',target_is_directory=True)
    return target,evidence


def install_hooks(records,reference):
    from check_baseline_numerics import rng_record
    expected = {(v['epoch'],v['phase']):v for v in
                (json.loads(x) for x in (reference/'input_fingerprints.jsonl').read_text().splitlines())}
    def append(path,value):
        if Path(path).name == 'input_fingerprints.jsonl':
            key = (value['epoch'],value['phase'])
            if key in expected and value != expected[key]:
                raise RuntimeError(f'Inputs differ from original baseline at {key}')
            frame = sys._getframe(1).f_locals
            model,optimizer,scheduler = (frame[k] for k in ('model','optimizer','scheduler'))
            digest = frame['paired'].state_digest
            before = rng_record(torch)
            moments = {name+'.'+key:item for name,p in model.named_parameters()
                       for key,item in optimizer.state.get(p,{}).items() if torch.is_tensor(item)}
            state = dict(epoch=value['epoch'],phase=value['phase'],
                model_sha256=digest(model.state_dict()),optimizer_sha256=digest(moments),
                scheduler=scheduler.state_dict(),rng=before)
            if before != rng_record(torch):
                raise RuntimeError('Read-only observer changed RNG')
            ORIGINAL_APPEND(records/'numerical_states.jsonl',state)
        ORIGINAL_APPEND(path,value)
    common.append_json = append
    common.losses = fixed_losses
    common.make_optimizer = make_optimizer
    common.gradient_report,common.check_gradients = gradient_report,check_gradients
    common.set_context = lambda *args:None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=VARIANTS,required=True)
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=1234)
    parser.add_argument('--steps',type=int,default=6)
    args = parser.parse_args(argv)
    args.variant = args.model
    args.epochs,args.batch,args.workers,args.threads = 100,12,2,4
    args.precision,args.strict = 'bf16',True
    if args.mode == 'train':
        from run_paper_model_study import signature
        gate = read(OUT/'gate.json')
        if gate['status']!='passed' or gate['source_signature']!=signature() or set(gate['smokes'])!=set(VARIANTS):
            raise RuntimeError('All-model preflight must pass for the current code before training')
    tag = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run_id = f'paper_{args.mode}_{args.model}_seed{args.seed}_{tag}_{os.getpid()}'
    records,checkpoints = OUTPUT/'experiments'/run_id,OUTPUT/'checkpoints'/run_id
    records.mkdir(parents=True,exist_ok=False)
    if args.mode == 'train':
        checkpoints.mkdir(parents=True,exist_ok=False)
    print('RECORDS',records,flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        import cv2
        cv2.setNumThreads(0)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed);np.random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        reference = baseline_records(args.seed)
        source = records/'source'
        built = common.build('exp00_baseline_paired_strict_v2',source)
        for split,count in [('train',880),('val',120),('test',120)]:
            path = source/f'configs/splits/kvasir/{split}.txt'
            if len(path.read_text().splitlines()) != count or path.read_bytes() != (reference/f'source/configs/splits/kvasir/{split}.txt').read_bytes():
                raise RuntimeError(f'Original split changed: {split}')
        if (source/'configs/splits/kvasir/val.txt').read_bytes() != (source/'configs/splits/kvasir/test.txt').read_bytes():
            raise RuntimeError('This study explicitly requires the original shared validation/test list')
        definition,evidence = snapshot_assets(args.model,records)
        shutil.copy2(__file__,records/'driver.py')
        shutil.copy2(common.__file__,records/'common_driver.py')
        sys.path.insert(0,str(source))
        sys.path.insert(0,str(definition))
        import paired_inputs as paired
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        factories = common.definitions(source/'kvasir_train.py',
            {'load_ids','get_train_transform','get_valid_transform'},dict(A=A,ToTensorV2=ToTensorV2,os=os))
        factories.update(common.definitions(source/'loss.py',{'StructureLoss_binary','IoU_binary'},
            dict(torch=torch,nn=nn,F=torch.nn.functional)))
        factories.update(common.definitions(source/'kvasir_test.py',{'compute_batch_metrics'},dict(torch=torch)))
        loaders = {}
        for phase,split in [('train','train'),('valid','val')]:
            ids = factories['load_ids'](str(source/f'configs/splits/kvasir/{split}.txt'))
            dataset = paired.PairedDataset(str(WORKSPACE/'datasets/Kvasir-SEG'),ids,
                factories['get_train_transform' if phase=='train' else 'get_valid_transform'](352),args.seed,phase)
            loaders[phase] = paired.make_loader(dataset,12 if phase=='train' else 6,2,pin_memory=True)
        if args.model == 'baseline':
            args.hiera_checkpoint = WORKSPACE/'checkpoints/sam2_hiera_large.fb_r1024.safetensors'
            if common.file_hash(args.hiera_checkpoint)!='7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c':
                raise RuntimeError('Original Hiera initialization changed')
            model = common.make_model(source,args)
            initial = paired.state_digest({k:v for k,v in model.state_dict().items()
                if '.frequency_enhancer.' not in k and not any(f'.up{i}.' in k for i in (1,2,3))})
            old = read(reference/'initial_state.json')
            if initial != old['shared_state_sha256']:
                raise RuntimeError('Original complete baseline initialization changed')
        else:
            import adapters
            adapters.ASSETS = records/'source/paper_assets'
            model = adapters.make_model(args.model)
        install_hooks(records,reference)
        config = read(DEFINITION/'protocol.json')
        config.update(model=args.model,seed=args.seed,split_hashes={name:common.file_hash(source/f'configs/splits/kvasir/{name}.txt') for name in ['train','val','test']})
        common.write_json(records/'effective_config.json',config)
        common.write_json(records/'initial_state.json',dict(complete_state_sha256=paired.state_digest(model.state_dict()),
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            deterministic_global_max_pool_count=getattr(model,'deterministic_global_max_pool_count',0),
            checkpointed_modules=getattr(model,'checkpointed_modules',[])))
        common.write_json(records/'manifest.json',dict(source=built,author=evidence,
            definition_sha256={p.name:common.file_hash(p) for p in DEFINITION.iterdir() if p.is_file()},
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                numpy=np.__version__,albumentations=A.__version__,precision='bf16',strict=True),
            original_training_function_sha256=hashlib.sha256(inspect.getsource(common.train).encode()).hexdigest(),
            command=[sys.executable,*sys.argv]))
        (records/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        model.cuda()
        if args.mode == 'smoke':
            result = common.smoke(model,loaders,factories,args,records)
            result['estimated_100_epoch_cap_compute_minutes']=5*result.pop('estimated_20_epoch_compute_minutes')
            before=paired.state_digest(model.state_dict())
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(state,strict=True)
            if paired.state_digest(model.state_dict())!=before:raise RuntimeError('Complete state reload changed content')
            common.write_json(records/'smoke.json',result)
            common.write_json(records/'full_reload.json',dict(strict_complete_state_reload=True,complete_state_sha256=before))
        else:
            from early_stopping import install_train
            train,expanded = install_train(common,10)
            (records/'effective_training_loop.py').write_text(expanded)
            result = train(model,loaders,paired,factories,args,records,checkpoints)
            result.update(early_stopping=read(records/'early_stopping.json'))
            common.write_json(records/'result.json',result)
        check_assets(args.model)
    except BaseException as error:
        common.write_json(records/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
