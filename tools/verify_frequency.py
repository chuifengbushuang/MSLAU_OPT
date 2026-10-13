"""Synthetic AFFE evidence and benchmark; never loads segmentation datasets.

Run from repository root: python -m tools.verify_frequency --device cpu
Use --full for real Hiera-L; --hiera-checkpoint enables pretrained smoke.
Output is created exclusively so earlier evidence cannot be overwritten.
"""
import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import timm
from networks.frequency_enhancement import FourStageFrequencyEnhancer, load_frequency_state_dict
from networks.mslau_net import MSLAU_net, SAM2UNetDecoder


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(module, forward, device, warmup, repeats, inputs=()):
    result = {}
    def clear_gradients():
        module.zero_grad(set_to_none=True)
        for tensor in inputs:
            tensor.grad = None

    for backward in (False, True):
        module.train(backward)
        def step():
            clear_gradients()
            with torch.set_grad_enabled(backward):
                output = forward()
                tensors = list(output.values()) if isinstance(output, dict) else output
                if torch.is_tensor(tensors):
                    tensors = [tensors]
                if backward:
                    sum(t.float().square().mean() for t in tensors).backward()
        for _ in range(warmup):
            step()
        clear_gradients()
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            step()
            synchronize(device)
            times.append((time.perf_counter()-start)*1000)
        result['forward_backward' if backward else 'forward'] = {
            'median_ms': statistics.median(times), 'min_ms': min(times), 'max_ms': max(times),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        }
        clear_gradients()
    return result


def stage_evidence(stage, x):
    """Return Python scalars only; release diagnostic graphs before benchmarking."""
    y, _, diag = stage.forward_with_details(x, return_diagnostics=True)
    y.square().mean().backward()
    with torch.no_grad():
        entry = {k: v.cpu().tolist() for k, v in diag.items() if k != 'weights'}
        entry.update({
            'shape': list(x.shape),
            'weight_std': diag['weights'].std().item(),
            'fft_identity_max_error': (torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho'), s=x.shape[-2:], norm='ortho')-x).abs().max().item(),
            'initial_relative_change': ((y-x).norm()/x.norm()).item(),
            'input_sensitivity_max': (stage.frequency_weights(3*x)-diag['weights']).abs().max().item(),
            'initial_gradient_l1': {k: p.grad.abs().sum().item() for k, p in stage.named_parameters()},
        })
    stage.zero_grad(set_to_none=True)
    x.grad = None
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--hiera-checkpoint')
    parser.add_argument('--p51-checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.repeats, args.threads) < 1:
        parser.error('batch, warmup, repeats and threads must be positive')
    if Path(args.output).exists():
        parser.error('output already exists; choose a new evidence path')
    torch.manual_seed(1234)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    report = {'environment': {
        'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__,
        'cuda_build': torch.version.cuda, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'threads': args.threads, 'batch': args.batch, 'warmup': args.warmup, 'repeats': args.repeats,
        'dtype': 'float32', 'timing_loss': 'sum of mean squared logits/features; no dataset or optimizer step',
        'measurement_version': 2,
        'memory_scope': 'absolute PyTorch allocated peak; diagnostics released and input/parameter gradients cleared between measurements',
    }}
    features = [torch.randn(args.batch, 64, s, s, device=device, requires_grad=True) for s in (88, 44, 22, 11)]
    affe = FourStageFrequencyEnhancer().to(device)
    report['affe_parameters'] = sum(p.numel() for p in affe.parameters())
    report['stage_parameters'] = [sum(p.numel() for p in m.parameters()) for m in affe.stages]
    report['stages'] = []
    for stage, x in zip(affe.stages, features):
        entry = stage_evidence(stage, x)
        entry['timing'] = measure(stage, lambda: stage(x), device, args.warmup, args.repeats, inputs=(x,))
        report['stages'].append(entry)
    report['affe_timing'] = measure(affe, lambda: affe(features), device, args.warmup, args.repeats, inputs=features)
    del affe, features, x, stage
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    encoder_features = [torch.randn(args.batch, c, s, s, device=device, requires_grad=True)
                        for c, s in zip((144, 288, 576, 1152), (88, 44, 22, 11))]
    report['decoders'] = {}
    for mode in ('none', 'naf2d'):
        decoder = SAM2UNetDecoder(frequency_mode=mode).to(device)
        report['decoders'][mode] = {
            'parameters': sum(p.numel() for p in decoder.parameters()),
            'timing': measure(decoder, lambda: decoder(encoder_features, 352, return_aux=True), device, args.warmup, args.repeats, inputs=encoder_features),
        }
        del decoder
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    del encoder_features
    report['full_models'] = {}
    # Always count instantiated real model parameters, even without expensive full forwards.
    for mode in ('none', 'naf2d'):
        model = MSLAU_net(img_size=352, encoder_name='sam2_hiera_large', decoder_mode='p5_sam2unet',
                          frequency_mode=mode, hiera_checkpoint=args.hiera_checkpoint)
        entry = {'parameters': sum(p.numel() for p in model.parameters()),
                 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'pretrained_hiera': bool(args.hiera_checkpoint), 'timing': None}
        if args.p51_checkpoint:
            entry['checkpoint_audit'] = load_frequency_state_dict(model, torch.load(
                args.p51_checkpoint, map_location='cpu', weights_only=True), allow_legacy=mode == 'naf2d')
        if args.full:
            model.to(device)
            image = torch.randn(args.batch, 3, 352, 352, device=device)
            entry['timing'] = measure(model, lambda: model(image, return_aux=True), device, args.warmup, args.repeats)
            del image
        report['full_models'][mode] = entry
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    with open(args.output, 'x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({'output': args.output, 'affe_parameters': report['affe_parameters'],
                      'full_models': report['full_models']}, indent=2))


if __name__ == '__main__':
    main()
