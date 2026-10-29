"""Authorized experiment-9 original/local/deep residual-gate comparison, preserving the audited training loop."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import local_decoder_study as common
import numpy as np
import torch
from torch import nn

ROOT, WORKSPACE = common.ROOT, common.WORKSPACE
CORE = ROOT / 'studies/sgg_residual/sgg_residual_core.py'
VARIANTS = ('sgg_original', 'sgg_local', 'sgg_deep')
OLD_RUNS = {
    'baseline': WORKSPACE / 'experiments/local_train_baseline_seed1234_20261028_004839_75754',
    'naf_decoder': WORKSPACE / 'experiments/local_train_naf_decoder_seed1234_20261028_010212_79638',
}


def make_model(source, args):
    from networks.mslau_net import MSLAU_net
    model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet', encoder_name='sam2_hiera_large',
                      hiera_checkpoint=str(args.hiera_checkpoint))
    legacy = common.load_module('sgg_legacy_shell', source / 'networks/naf_discrete.py')
    module = common.load_module('sgg_core', source / 'networks/sgg_residual_core.py')
    shell = legacy.CascadeDiscreteNAF2D(64)
    enhancer = module.install_residual_gates(shell, mode=args.variant.removeprefix('sgg_'))
    model.progressive_decoder = common.StudyDecoder(model.progressive_decoder, enhancer, False)
    return model


_original_gradient_report = common.gradient_report
_original_check_gradients = common.check_gradients


def gradient_report(model):
    report = _original_gradient_report(model)
    gates, stages = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or '.frequency_enhancer.stages.' not in name:
            continue
        stage = name.split('.stages.')[1].split('.')[0]
        value = parameter.grad.abs().max().item() if parameter.grad is not None else None
        if '.residual_gate.' in name:
            gates[name] = value
        for group in ('local_projection', 'context_projection', 'selector', 'fusion_gate', 'output_projection'):
            if f'.{group}.' in name:
                stages.setdefault(stage, {}).setdefault(group, []).append(value)
    report['residual_gate_parameters'] = gates
    report['stages'] = {s:{g:max(v) if all(x is not None for x in v) else None
                              for g,v in groups.items()} for s,groups in stages.items()}
    return report


def check_gradients(report, variant):
    _original_check_gradients(report, variant)
    if len(report['stages']) != 3 or any(v is None or v <= 0
            for stage in report['stages'].values() for v in stage.values()):
        raise RuntimeError(f'Closed original NAF gradient path: {report}')
    if variant != 'sgg_original':
        values = report['residual_gate_parameters']
        if len(values) != 30 or any(v is None or v <= 0 for v in values.values()):
            raise RuntimeError(f'Closed residual gate gradient: {report}')


def install_loop_hooks():
    # Keep common.losses and common.set_context unchanged: selector regularizer
    # .001 and sample-keyed frequency selection are required in all three arms.
    common.gradient_report = gradient_report
    common.check_gradients = check_gradients


def zero_projection_parity(model, loaders, args, records):
    model.eval()
    images, _, _ = next(iter(loaders['valid']))
    with torch.no_grad(), common.autocast(args):
        images = images.cuda(non_blocking=True)
        features = model.encoder(images)
        decoder = model.progressive_decoder
        projected = [rfb(x) for rfb, x in zip(decoder.rfbs, features)]
        enhanced = decoder.frequency_enhancer(projected)[0]
        if not all(torch.equal(a, b) for a, b in zip(projected, enhanced)):
            raise RuntimeError('Zero output projection changed RFB features')
        d3 = decoder.up1(projected[3], projected[2])
        d2 = decoder.up2(d3, projected[1])
        d1 = decoder.up3(d2, projected[0])
        expected = decoder.resize_logits(decoder.head(d1), 352)
        if not torch.equal(expected, model(images)):
            raise RuntimeError('Initial logits differ from original decoder')
    common.write_json(records / 'zero_projection_parity.json', dict(features_equal=True, logits_equal=True))



def baseline_records(seed):
    matches = [p for p in (WORKSPACE / 'experiments').glob(f'local_train_baseline_seed{seed}_*')
               if (p / 'result.json').is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one cached baseline: {seed}: {matches}')
    return matches[0]


def install_audit_hooks(baseline, records):
    from check_baseline_numerics import rng_record
    original_append, original_train = common.append_json, common.train
    inputs = [json.loads(line) for line in (baseline / 'input_fingerprints.jsonl').read_text().splitlines()]
    expected = {(v['epoch'], v['phase']): v for v in inputs}
    def append(path, value):
        if Path(path).name == 'input_fingerprints.jsonl':
            if value != expected[(value['epoch'], value['phase'])]:
                raise RuntimeError(f'Input trajectory differs from cached baseline: {value}')
            context = sys._getframe(1).f_locals
            model, optimizer, scheduler = (context[k] for k in ('model', 'optimizer', 'scheduler'))
            digest = context['paired'].state_digest
            before = rng_record(torch)
            moments = {name + '.' + key: value for name, parameter in model.named_parameters()
                       for key, value in optimizer.state.get(parameter, {}).items() if torch.is_tensor(value)}
            state = dict(full_model_sha256=digest(model.state_dict()),
                buffers_sha256=digest(dict(model.named_buffers())), optimizer_tensors_sha256=digest(moments),
                optimizer_param_groups=optimizer.state_dict()['param_groups'], scheduler=scheduler.state_dict(),
                model_training=model.training, rng=before)
            if before != rng_record(torch):
                raise RuntimeError('Read-only state hashing changed RNG')
            original_append(records / 'numerical_states.jsonl', dict(epoch=value['epoch'], phase=value['phase'],
                position='after_phase_metrics_and_input_digest_before_scheduler_and_checkpoint', **state))
        original_append(path, value)
    def train(model, loaders, paired, factories, args, record_path, checkpoints):
        strict = common.load_module('sgg_runtime_observer', records / 'source/strict_runtime.py')
        if (records / 'environment.txt').read_bytes() != (baseline / 'environment.txt').read_bytes():
            raise RuntimeError('Environment differs from cached baseline')
        common.write_json(records / 'initial_runtime.json', dict(runtime=strict.runtime_record(), rng=rng_record(torch),
            full_model_sha256=paired.state_digest(model.state_dict()), baseline=str(baseline),
            train_function_sha256=hashlib.sha256(original_train.__code__.co_code).hexdigest()))
        return original_train(model, loaders, paired, factories, args, record_path, checkpoints)
    common.append_json, common.train = append, train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--dataset', type=Path, default=WORKSPACE / 'datasets/Kvasir-SEG')
    parser.add_argument('--hiera-checkpoint', type=Path, default=WORKSPACE / 'checkpoints/sam2_hiera_large.fb_r1024.safetensors')
    parser.add_argument('--output-root', type=Path, default=WORKSPACE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.steps < 4 or min(args.batch, args.threads, args.epochs) < 1 or args.workers < 0:
        parser.error('Invalid resource arguments')
    tag = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run_id = f'local_{args.mode}_{args.variant}_seed{args.seed}_{tag}_{os.getpid()}'
    records = args.output_root / 'experiments' / run_id
    records.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output_root / 'checkpoints' / run_id
    if args.mode == 'train':
        checkpoints.mkdir(parents=True, exist_ok=False)
    common.write_json(records / 'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    print('RECORDS', records, flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        import cv2
        cv2.setNumThreads(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(args.strict, warn_only=False)
        source = records / 'source'
        manifest = common.build('exp00_baseline_paired_strict_v2', source)
        shutil.copy2(common.NAF_FILE, source / 'networks/naf_discrete.py')
        shutil.copy2(CORE, source / 'networks/sgg_residual_core.py')
        shutil.copy2(__file__, records / 'driver.py')
        shutil.copy2(common.__file__, records / 'common_driver.py')
        shutil.copy2(ROOT / 'tools/check_baseline_numerics.py', records / 'rng_audit_source.py')
        sys.path.insert(0, str(source))
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import paired_inputs as paired
        factories = common.definitions(source / 'kvasir_train.py',
            {'load_ids', 'get_train_transform', 'get_valid_transform'}, dict(A=A, ToTensorV2=ToTensorV2, os=os))
        factories.update(common.definitions(source / 'loss.py', {'StructureLoss_binary', 'IoU_binary'},
                                            dict(torch=torch, nn=nn, F=torch.nn.functional)))
        factories.update(common.definitions(source / 'kvasir_test.py', {'compute_batch_metrics'}, dict(torch=torch)))
        loaders = {}
        for phase, split in (('train', 'train'), ('valid', 'val')):
            ids = factories['load_ids'](str(source / f'configs/splits/kvasir/{split}.txt'))
            dataset = paired.PairedDataset(str(args.dataset), ids,
                factories['get_train_transform' if phase == 'train' else 'get_valid_transform'](352), args.seed, phase)
            loaders[phase] = paired.make_loader(dataset, args.batch if phase == 'train' else max(1, args.batch // 2),
                                                args.workers, pin_memory=True)
        checkpoint_hash = common.file_hash(args.hiera_checkpoint)
        if checkpoint_hash != '7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c':
            raise RuntimeError('Unexpected Hiera checkpoint')
        model = make_model(source, args)
        shared = {n: t for n, t in model.state_dict().items() if '.frequency_enhancer.' not in n
                  and not any(f'.up{i}.' in n for i in (1, 2, 3))}
        initial = dict(shared_state_sha256=paired.state_digest(shared),
            baseline_state_sha256=paired.state_digest({n: t for n, t in model.state_dict().items() if '.frequency_enhancer.' not in n}),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            complete_initial_sha256=paired.state_digest(model.state_dict()),
            cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        if args.variant != 'baseline':
            shell = {n: t for n, t in model.state_dict().items() if '.frequency_enhancer.' in n
                     and any(f'.{key}.' in n for key in ('local_projection', 'context_projection', 'fusion_gate', 'output_projection'))}
            initial['shell_initial_sha256'] = paired.state_digest(shell)
        if args.seed == 1234:
            for label, directory in OLD_RUNS.items():
                previous = json.loads((directory / 'initial_state.json').read_text())
                for key in ('shared_state_sha256', 'cpu_rng_sha256'):
                    if initial[key] != previous[key]:
                        raise RuntimeError(f'Initial {key} differs from cached {label}')
        baseline = baseline_records(args.seed)
        previous = json.loads((baseline / 'initial_state.json').read_text())
        for key in ('shared_state_sha256', 'cpu_rng_sha256'):
            if initial[key] != previous[key]:
                raise RuntimeError(f'Baseline initialization differs: {args.seed}/{key}')
        if 'baseline_state_sha256' in previous and initial['baseline_state_sha256'] != previous['baseline_state_sha256']:
            raise RuntimeError('Complete original-model initialization changed')
        initial['legacy_initial_sha256'] = paired.state_digest({n:t for n,t in model.state_dict().items() if '.residual_gate.' not in n})
        del shared
        common.write_json(records / 'initial_state.json', initial)
        common.write_json(records / 'manifest.json', dict(source=manifest, core_sha256=common.file_hash(CORE),
            naf_shell_source_sha256=common.file_hash(common.NAF_FILE), driver_sha256=common.file_hash(__file__),
            common_driver_sha256=common.file_hash(common.__file__), hiera_sha256=checkpoint_hash,
            input_protocol=paired.PROTOCOL, architecture=args.variant,
            paper_doi='10.1109/TGRS.2026.3694686', adaptation='mask_only_raw_local_residual_2sigmoid_width16_dilation2',
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                albumentations=A.__version__, gpu=torch.cuda.get_device_name(0), strict=args.strict,
                precision=args.precision, threads=args.threads), command=[sys.executable, *sys.argv]))
        config = json.loads((OLD_RUNS['baseline'] / 'effective_config.json').read_text())
        config.update(variant=args.variant, seed=args.seed, dataset=str(args.dataset.resolve()),
            hiera_checkpoint=str(args.hiera_checkpoint.resolve()), batch=args.batch,
            validation_batch=max(1, args.batch // 2), workers=args.workers, cpu_threads=args.threads,
            precision=args.precision, strict=args.strict, epochs=args.epochs, naf_reg_weight=.001)
        common.write_json(records / 'effective_config.json', config)
        (records / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        install_loop_hooks()
        install_audit_hooks(baseline, records)
        model.cuda()
        if args.mode == 'smoke':
            if args.variant != 'baseline':
                zero_projection_parity(model, loaders, args, records)
            common.smoke(model, loaders, factories, args, records)
            # Full state, including frozen encoder and running statistics.
            buffer = records / 'smoke_full_state.pth'
            torch.save(model.state_dict(), buffer)
            restored = torch.load(buffer, map_location='cpu', weights_only=True)
            model.load_state_dict(restored, strict=True)
            del restored
            buffer.unlink()
            common.write_json(records / 'full_reload.json', dict(strict_full_state_reload=True))
        else:
            common.train(model, loaders, paired, factories, args, records, checkpoints)
    except BaseException as error:
        common.write_json(records / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
