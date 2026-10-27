"""Local comparison: original decoder, NAF+decoder, and NAF as decoder.

Builds a persistent source snapshot without editing any frozen experiment.
All arms use the original augmentation operations with paired_aug_v2 streams,
the original three Structure losses, AdamW and cosine schedule. Smoke updates
are disposable and never become a formal training initialization.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from build_experiment import ROOT, build

WORKSPACE = ROOT.parent
NAF_FILE = ROOT / 'experiments/exp09_naf_discrete_paired_strict_v2/added/networks/naf_discrete.py'
VARIANTS = ('baseline', 'naf_decoder', 'naf_replacement')


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing source definitions: {names}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


class StudyDecoder(nn.Module):
    """Reuse original RFB/heads and optional UpBlocks, preserving their tensors."""
    def __init__(self, original, enhancer, replacement):
        super().__init__()
        self.rfbs = original.rfbs
        if not replacement:
            self.up1, self.up2, self.up3 = original.up1, original.up2, original.up3
        self.side1, self.side2, self.head = original.side1, original.side2, original.head
        self.frequency_enhancer = enhancer
        self.replacement = replacement

    @staticmethod
    def resize_logits(logits, output_size):
        return F.interpolate(logits, size=(output_size, output_size), mode='bilinear', align_corners=False)

    def forward(self, encoder_features, output_size, return_aux=False):
        features = [rfb(feature) for rfb, feature in zip(self.rfbs, encoder_features)]
        (e1, e2, e3, e4), regularizer, stats, fusion = self.frequency_enhancer(features)
        if self.replacement:
            d1, d2, d3 = e1, e2, e3
        else:
            d3 = self.up1(e4, e3)
            d2 = self.up2(d3, e2)
            d1 = self.up3(d2, e1)
        logits = self.resize_logits(self.head(d1), output_size)
        if not return_aux:
            return logits
        return dict(logits=logits, aux_d2=self.resize_logits(self.side2(d2), output_size),
                    aux_d3=self.resize_logits(self.side1(d3), output_size),
                    naf_regularization=regularizer, naf_stats=stats, naf_fusion_stats=fusion)


def make_model(source, args):
    from networks.mslau_net import MSLAU_net
    model = MSLAU_net(img_size=352, decoder_mode='p5_sam2unet',
                     encoder_name='sam2_hiera_large', hiera_checkpoint=str(args.hiera_checkpoint))
    if args.variant != 'baseline':
        naf = load_module('local_study_naf', source / 'networks/naf_discrete.py')
        enhancer = naf.CascadeDiscreteNAF2D(channels=64)
        model.progressive_decoder = StudyDecoder(model.progressive_decoder, enhancer,
                                                  args.variant == 'naf_replacement')
    return model


def gradient_report(model):
    groups = {key: [] for key in ('encoder_adapters', 'rfbs', 'head', 'selector', 'fusion', 'output_projection')}
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f'Nonfinite gradient: {name}')
        for group, token in (('encoder_adapters', 'prompt_learn'), ('rfbs', '.rfbs.'),
                             ('head', '.head.'), ('selector', '.selector.'),
                             ('fusion', '.fusion_gate.'), ('output_projection', '.output_projection.')):
            if token in name:
                groups[group].append(gradient.abs().max())
    report = {key: torch.stack(values).max().item() if values else None for key, values in groups.items()}
    report['missing'] = missing
    return report


def check_gradients(report, variant):
    required = ['encoder_adapters', 'rfbs', 'head']
    if variant != 'baseline':
        required += ['selector', 'fusion', 'output_projection']
    if report['missing'] or any(report[key] is None or report[key] <= 0 for key in required):
        raise RuntimeError(f'Gradient path failed: {report}')


def make_optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=1e-3),
        dict(params=[p for name, p in model.named_parameters() if not name.startswith('encoder.')], lr=1e-3)
    ], weight_decay=5e-4)


def set_context(model, args, epoch, ids):
    if args.variant != 'baseline':
        model.progressive_decoder.frequency_enhancer.set_selection_context(args.seed, epoch, ids)


def losses(outputs, labels, criterion, variant, training):
    main = criterion(outputs['logits'], labels)
    total = main + criterion(outputs['aux_d2'], labels) + criterion(outputs['aux_d3'], labels)
    if training and variant != 'baseline':
        total = total + .001 * outputs['naf_regularization']
    return main, total


def autocast(args):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16')


def smoke(model, loaders, factories, args, records):
    criterion, metric = factories['StructureLoss_binary'](), factories['IoU_binary']()
    optimizer = make_optimizer(model)
    model.train()
    iterator = iter(loaders['train'])
    timings, rows = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        images, masks, ids = next(iterator)
        images = images.cuda(non_blocking=True)
        labels = masks.cuda(non_blocking=True).float().unsqueeze(1)
        set_context(model, args, 0, ids)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args):
            outputs = model(images, return_aux=True)
            main, total = losses(outputs, labels, criterion, args.variant, True)
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite smoke loss')
        total.backward()
        if step == args.steps - 1:
            gradients = gradient_report(model)
            check_gradients(gradients, args.variant)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        row = dict(step=step, seconds=elapsed, loss=total.item(), main_loss=main.item())
        rows.append(row)
        print('SMOKE', json.dumps(row), flush=True)
    del iterator, outputs, total, main
    model.eval()
    images, masks, ids = next(iter(loaders['valid']))
    images = images.cuda(non_blocking=True)
    with torch.no_grad(), autocast(args):
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = model(images)
        second = model(images)
        torch.cuda.synchronize()
        inference_seconds = (time.perf_counter() - started) / 2
        if first.shape != (len(ids), 1, 352, 352) or not torch.isfinite(first).all():
            raise RuntimeError('Invalid inference output')
        if not torch.equal(first, second):
            raise RuntimeError('Repeated eval outputs differ')
        buffer = records / 'smoke_trainable_state.pth'
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        state = {name: t.detach().cpu() for name, t in model.state_dict().items() if name in trainable}
        torch.save(state, buffer)
        restored = torch.load(buffer, map_location='cpu', weights_only=True)
        model.load_state_dict(restored, strict=False)
        if not torch.equal(first, model(images)):
            raise RuntimeError('Checkpoint round trip changed logits')
        buffer.unlink()
    steady = statistics.median(timings[2:-1])  # Exclude startup and gradient-audit overhead.
    result = dict(status='completed', variant=args.variant, batch=args.batch, workers=args.workers,
                  threads=args.threads, precision=args.precision, strict=args.strict, steps=rows,
                  steady_step_seconds=steady, images_per_second=args.batch / steady,
                  inference_batch=len(ids), inference_batch_seconds=inference_seconds,
                  estimated_20_epoch_compute_minutes=20 * (len(loaders['train']) * steady +
                      len(loaders['valid']) * inference_seconds) / 60,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  gradient_checks=gradients, repeated_eval_equal=True, checkpoint_roundtrip_equal=True)
    write_json(records / 'smoke.json', result)
    print('SMOKE_RESULT', json.dumps(result), flush=True)
    return result


def train(model, loaders, paired, factories, args, records, checkpoints):
    criterion, metric = factories['StructureLoss_binary'](), factories['IoU_binary']()
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    history, best_iou, best_loss = [], -math.inf, math.inf
    started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        result = dict(epoch=epoch)
        for phase in ('train', 'valid'):
            training = phase == 'train'
            model.train(training)
            paired.set_loader_epoch(loaders[phase], epoch)
            digest = paired.InputDigest()
            count, main_sum, loss_sum, iou_sum = 0, 0., 0., 0.
            phase_start = time.perf_counter()
            for step, (images, masks, ids) in enumerate(loaders[phase]):
                digest.update(images, masks, ids)
                images = images.cuda(non_blocking=True)
                labels = masks.cuda(non_blocking=True).float().unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                set_context(model, args, epoch, ids)
                with torch.set_grad_enabled(training), autocast(args):
                    outputs = model(images, return_aux=True)
                    main, total = losses(outputs, labels, criterion, args.variant, training)
                    if not torch.isfinite(total):
                        raise RuntimeError(f'Nonfinite loss: {epoch}/{phase}/{step}')
                    # Keep probabilities/thresholding FP32, as in the original
                    # FP32 recipe; BF16 sigmoid would round a band around .5.
                    score = metric(outputs['logits'].float(), labels)
                    if training:
                        total.backward()
                        if epoch == 0 and step == 2:
                            gradients = gradient_report(model)
                            check_gradients(gradients, args.variant)
                            write_json(records / 'training_gradients.json', gradients)
                        optimizer.step()
                n = len(ids)
                count += n
                main_sum += main.item() * n
                loss_sum += total.item() * n
                iou_sum += score.item() * n
                if training and (step + 1) % 20 == 0:
                    print(f'Epoch {epoch + 1}/{args.epochs} train {step + 1}/{len(loaders[phase])} '
                          f'loss={loss_sum / count:.6f}', flush=True)
            values = dict(main_loss=main_sum / count, loss=loss_sum / count, iou=iou_sum / count,
                          samples=count, seconds=time.perf_counter() - phase_start)
            result[phase] = values
            append_json(records / 'input_fingerprints.jsonl', digest.record(args.seed, epoch, phase))
            if training:
                scheduler.step()  # Same position as the original loop: before validation.
            else:
                if values['main_loss'] <= best_loss:
                    best_loss, best_loss_epoch, best_loss_iou = values['main_loss'], epoch, values['iou']
                    torch.save(model.state_dict(), checkpoints / 'running_best_loss.pth')
                if values['iou'] >= best_iou:
                    best_iou, best_epoch, best_iou_loss = values['iou'], epoch, values['main_loss']
                    torch.save(model.state_dict(), checkpoints / 'kvasir_best_model.pth')
        result.update(seconds=time.perf_counter() - epoch_start, best_iou=best_iou, best_epoch=best_epoch,
                      lr=[group['lr'] for group in optimizer.param_groups])
        append_json(records / 'epoch_metrics.jsonl', result)
        history.append(result)
        write_json(records / 'progress.json', dict(status='training', completed_epochs=epoch + 1,
                   total_epochs=args.epochs, best_iou=best_iou, best_epoch=best_epoch,
                   elapsed_minutes=(time.perf_counter() - started) / 60,
                   estimated_remaining_minutes=(time.perf_counter() - started) / (epoch + 1) *
                       (args.epochs - epoch - 1) / 60))
        print('EPOCH', json.dumps(result), flush=True)
    shutil.copy2(checkpoints / 'kvasir_best_model.pth',
                 checkpoints / f'best_iou_{best_iou:.6f}_epoch_{best_epoch}_{best_iou_loss:.6f}.pth')
    (checkpoints / 'running_best_loss.pth').rename(
        checkpoints / f'best_loss_{best_loss:.6f}_epoch_{best_loss_epoch}_{best_loss_iou:.6f}.pth')
    model.load_state_dict(torch.load(checkpoints / 'kvasir_best_model.pth', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    paired.set_loader_epoch(loaders['valid'], 0)
    metrics = {}
    with torch.no_grad(), autocast(args):
        for images, masks, ids in loaders['valid']:
            logits = model(images.cuda(non_blocking=True))
            # Match the original training/checkpoint-selection IoU comparator.
            values = factories['compute_batch_metrics']((logits.float().sigmoid() > .5).float(),
                                                       masks.cuda(non_blocking=True).unsqueeze(1))
            for key, value in values.items():
                metrics.setdefault(key, []).extend(value)
    evaluation = {key: statistics.mean(value) for key, value in metrics.items()}
    if abs(evaluation['iou'] - best_iou) > 1e-6:
        raise RuntimeError(f'Independent evaluation mismatch: {evaluation["iou"]} vs {best_iou}')
    write_json(records / 'evaluation.json', evaluation)
    result = dict(status='completed', variant=args.variant, seed=args.seed, best_iou=best_iou,
                  best_epoch=best_epoch, evaluation=evaluation,
                  baseline_target=.8786, tolerance=.005,
                  within_reference_tolerance=abs(best_iou - .8786) <= .005,
                  elapsed_minutes=(time.perf_counter() - started) / 60,
                  checkpoints={p.name: file_hash(p) for p in checkpoints.glob('*.pth')})
    write_json(records / 'result.json', result)
    print('TRAIN_RESULT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--dataset', type=Path, default=WORKSPACE / 'datasets/Kvasir-SEG')
    parser.add_argument('--hiera-checkpoint', type=Path, default=WORKSPACE / 'checkpoints/sam2_hiera_large.fb_r1024.safetensors')
    parser.add_argument('--output-root', type=Path, default=WORKSPACE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.steps < 4 or min(args.batch, args.threads, args.epochs) < 1 or args.workers < 0:
        parser.error('Invalid batch/threads/epochs/workers; smoke requires at least four steps')
    tag = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run_id = f'local_{args.mode}_{args.variant}_seed{args.seed}_{tag}_{os.getpid()}'
    records = args.output_root / 'experiments' / run_id
    records.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output_root / 'checkpoints' / run_id
    if args.mode == 'train':
        checkpoints.mkdir(parents=True, exist_ok=False)
    write_json(records / 'arguments.json', {key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()})
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
        manifest = build('exp00_baseline_paired_strict_v2', source)
        shutil.copy2(NAF_FILE, source / 'networks/naf_discrete.py')
        shutil.copy2(__file__, records / 'driver.py')
        sys.path.insert(0, str(source))
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import paired_inputs as paired
        factories = definitions(source / 'kvasir_train.py',
            {'load_ids', 'get_train_transform', 'get_valid_transform'}, dict(A=A, ToTensorV2=ToTensorV2, os=os))
        factories.update(definitions(source / 'loss.py', {'StructureLoss_binary', 'IoU_binary'},
                                     dict(torch=torch, nn=nn, F=F)))
        factories.update(definitions(source / 'kvasir_test.py', {'compute_batch_metrics'}, dict(torch=torch)))
        loaders = {}
        for phase, split in (('train', 'train'), ('valid', 'val')):
            ids = factories['load_ids'](str(source / f'configs/splits/kvasir/{split}.txt'))
            missing = [str(args.dataset / sub / sample) for sample in ids for sub in ('images', 'masks')
                       if not (args.dataset / sub / sample).is_file()]
            if missing:
                raise FileNotFoundError(missing[:5])
            dataset = paired.PairedDataset(str(args.dataset), ids,
                factories['get_train_transform' if phase == 'train' else 'get_valid_transform'](352), args.seed, phase)
            loaders[phase] = paired.make_loader(dataset, args.batch if phase == 'train' else max(1, args.batch // 2),
                                                args.workers, pin_memory=True)
        checkpoint_hash = file_hash(args.hiera_checkpoint)
        if checkpoint_hash != '7694579e76566289f25f63799024a2fe83144005c75ece6c09458254aa3ee13c':
            raise RuntimeError('Unexpected Hiera checkpoint hash')
        model = make_model(source, args)
        shared = {name: value for name, value in model.state_dict().items()
                  if '.frequency_enhancer.' not in name and not any(f'.up{i}.' in name for i in (1, 2, 3))}
        write_json(records / 'initial_state.json', dict(shared_state_sha256=paired.state_digest(shared),
                   parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        del shared
        write_json(records / 'manifest.json', dict(source=manifest, naf_sha256=file_hash(NAF_FILE),
            driver_sha256=file_hash(__file__), hiera_sha256=checkpoint_hash, input_protocol=paired.PROTOCOL,
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                         albumentations=A.__version__, gpu=torch.cuda.get_device_name(0),
                         strict=args.strict, precision=args.precision, threads=args.threads),
            architecture=args.variant, reference_iou=.8786, tolerance=.005,
            command=[sys.executable, *sys.argv]))
        write_json(records / 'effective_config.json', dict(variant=args.variant, seed=args.seed,
            dataset=str(args.dataset.resolve()), hiera_checkpoint=str(args.hiera_checkpoint.resolve()),
            img_size=352, batch=args.batch, validation_batch=max(1, args.batch // 2),
            workers=args.workers, cpu_threads=args.threads, precision=args.precision,
            strict=args.strict, input_protocol=paired.PROTOCOL, epochs=args.epochs,
            optimizer='AdamW', encoder_lr=.001, decoder_lr=.001, weight_decay=.0005,
            scheduler='CosineAnnealingLR', eta_min=0, loss='StructureLoss_binary',
            main_weight=1, aux_d2_weight=1, aux_d3_weight=1, boundary_weight=0,
            naf_reg_weight=0 if args.variant == 'baseline' else .001,
            evaluation_probability_dtype='float32', evaluation_threshold=.5,
            evaluation_comparator='>', training_drop_last=True))
        (records / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        model.cuda()
        if args.mode == 'smoke':
            smoke(model, loaders, factories, args, records)
        else:
            train(model, loaders, paired, factories, args, records, checkpoints)
    except BaseException as error:
        write_json(records / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
