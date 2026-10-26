"""Replay three real baseline updates in fresh processes; never save weights."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from zoneinfo import ZoneInfo

from build_experiment import build


def definitions(path, names, namespace):
    """Use original definitions without importing train/loss optional packages."""
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f'Missing source definitions: {names}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def first_difference(left, right, path=''):
    if type(left) is not type(right):
        return path or '<root>'
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + '.keys'
        for key in left:
            difference = first_difference(left[key], right[key], f'{path}.{key}'.lstrip('.'))
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + '.length'
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f'{path}[{index}]')
            if difference:
                return difference
    elif left != right:
        return path or '<root>'
    return None


def tensor_hashes(tensors, digest):
    return {name: digest({name: value}) for name, value in sorted(tensors.items())}


def rng_record(torch):
    import pickle
    import random
    import numpy as np
    sha = lambda value: hashlib.sha256(value).hexdigest()
    return dict(cpu=sha(torch.get_rng_state().numpy().tobytes()),
                cuda=[sha(value.cpu().numpy().tobytes()) for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
                python=sha(pickle.dumps(random.getstate(), protocol=4)),
                numpy=sha(pickle.dumps(np.random.get_state(), protocol=4)))


def replay_steps(model, optimizer, criterion, batches, digest, device, weights, completed=None):
    """Original FP32, train mode, main+D2+D3 loss and AdamW update order."""
    import torch
    model.train(True)
    records = []
    for index, (images, masks, ids) in enumerate(batches):
        record = dict(step=index, inputs=dict(ids=list(ids),
                      tensors=tensor_hashes({'images': images, 'masks': masks}, digest)),
                      rng_before=rng_record(torch))
        images, labels = images.to(device), masks.to(device).float()
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images, return_aux=True)
        terms = {key: criterion(outputs[key], labels)
                 for key in ('logits', 'aux_d2', 'aux_d3')}
        loss = terms['logits'] + weights[0] * terms['aux_d2'] + weights[1] * terms['aux_d3']
        if not torch.isfinite(loss).item():
            raise RuntimeError(f'Nonfinite loss at step {index}')
        record['forward'] = dict(outputs=tensor_hashes(
            {key: outputs[key] for key in terms}, digest),
            losses={**{key: value.item() for key, value in terms.items()}, 'total': loss.item()})
        loss.backward()
        gradients = {name: parameter.grad.detach().cpu().clone()
                     for name, parameter in model.named_parameters() if parameter.grad is not None}
        if not all(torch.isfinite(value).all().item() for value in gradients.values()):
            raise RuntimeError(f'Nonfinite gradient at step {index}')
        record['backward'] = dict(gradients=tensor_hashes(gradients, digest),
            norm=sum(value.double().square().sum().item() for value in gradients.values()) ** 0.5,
            missing=[name for name, p in model.named_parameters() if p.requires_grad and p.grad is None])
        optimizer.step()
        record['update'] = dict(parameters=tensor_hashes(
            {name: p for name, p in model.named_parameters() if p.requires_grad}, digest),
            buffers=tensor_hashes(dict(model.named_buffers()), digest))
        record['rng_after'] = rng_record(torch)
        records.append(record)
        if completed:
            completed(records)
        print(f'Completed diagnostic update {index + 1}/{len(batches)}; loss={loss.item():.10f}', flush=True)
    return records


def worker(args):
    # Called only in a fresh subprocess. Import no kvasir_train/loss module:
    # their unused torchmetrics dependency caused the previous /tmp probe failure.
    report = dict(status='running', initial=None, steps=[])
    output = args.worker_output

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    try:
        if not Path(args.dataset).is_dir():
            raise FileNotFoundError(f'Dataset directory not found: {args.dataset}')
        if not Path(args.hiera_checkpoint).is_file():
            raise FileNotFoundError(f'Hiera checkpoint not found: {args.hiera_checkpoint}')
        import random
        import numpy as np
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import timm

        source = args.worker_source
        sys.path.insert(0, str(source))
        from networks.mslau_net import MSLAU_net
        from paired_inputs import PairedDataset, make_loader, set_loader_epoch, state_digest

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the real Hiera diagnostic')
        config = json.loads((source / 'config.json').read_text())['train_options']
        factories = definitions(source / 'kvasir_train.py', {'load_ids', 'get_train_transform'},
                                dict(os=os, A=A, ToTensorV2=ToTensorV2))
        criterion = definitions(source / 'loss.py', {'StructureLoss_binary'},
                                dict(torch=torch, nn=torch.nn, F=torch.nn.functional))['StructureLoss_binary']()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(args.mode == 'strict', warn_only=False)
        ids = factories['load_ids'](str(source / 'configs/splits/kvasir/train.txt'))
        dataset = PairedDataset(args.dataset, ids,
            factories['get_train_transform'](config['img_size']), args.seed, 'train')
        loader = make_loader(dataset, config['batch'], config['num_workers'], pin_memory=True)
        model = MSLAU_net(img_size=config['img_size'], decoder_mode=config['decoder_mode'],
                         encoder_name=config['encoder_name'], hiera_checkpoint=args.hiera_checkpoint)
        report['initial'] = dict(state_sha256=state_digest(model.state_dict()), rng=rng_record(torch))
        model = model.cuda()
        model.train(True)
        report['runtime'] = dict(python=sys.version, executable=sys.executable,
            torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            timm=timm.__version__, albumentations=A.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0), mode=args.mode,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                              memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                              math=torch.backends.cuda.math_sdp_enabled()),
            fused_attention_modules=[name for name, module in model.named_modules()
                                     if getattr(module, 'fused_attn', False)],
            active_stochastic_modules=[name for name, module in model.named_modules()
                if module.training and 'Drop' in type(module).__name__
                and float(getattr(module, 'p', getattr(module, 'drop_prob', 0))) > 0],
            train_options=config, dataset=args.dataset, hiera_checkpoint=args.hiera_checkpoint,
            cpu_threads=torch.get_num_threads(),
            cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        save()
        print('Initial state:', report['initial']['state_sha256'], flush=True)
        print('Strict algorithms:', report['runtime']['deterministic'],
              '| Active stochastic modules:', report['runtime']['active_stochastic_modules'], flush=True)
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in model.encoder.parameters() if p.requires_grad],
                 lr=float(config['encoder_lr'])),
            dict(params=[p for name, p in model.named_parameters() if not name.startswith('encoder.')],
                 lr=float(config['decoder_lr']))], weight_decay=5e-4)
        # First three batches of epoch 0; no scheduler step occurs inside an epoch.
        set_loader_epoch(loader, 0)
        iterator = iter(loader)
        try:
            batches = [next(iterator) for _ in range(args.steps)]
        finally:
            del iterator

        def completed(records):
            report['steps'] = records
            save()

        replay_steps(model, optimizer, criterion, batches, state_digest, torch.device('cuda'),
                     (config['aux_d2_weight'], config['aux_d3_weight']), completed)
        report['status'] = 'completed'
        save()
        return 0
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mode', choices=('strict', 'default'), default='strict')
    parser.add_argument('--steps', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--dataset', default='/data/dataset/zsy/mslau-net/Kvasir-SEG')
    parser.add_argument('--hiera-checkpoint', default='/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-source', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source:
        return worker(args)

    tag = f'p51_numeric_replay_{args.mode}_seed{args.seed}_' + datetime.datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    output = args.output or Path(os.environ.get('P51_OUTPUT_ROOT', '/data/models/le/mslau-net')) / 'experiments' / tag
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'Diagnostic records: {output}\nTwo fresh processes, {args.steps} updates each; no weight files.', flush=True)
    summary = dict(status='running', mode=args.mode, steps=args.steps, seed=args.seed,
                   physical_gpu=args.gpu, dataset=args.dataset, hiera_checkpoint=args.hiera_checkpoint,
                   command=sys.argv, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   limitations=['Short diagnostic updates, not formal training or validation.',
                                'Equality covers the observed steps, not an entire training run.',
                                'CPU fingerprints synchronize CUDA; this is not a performance benchmark.',
                                'SDPA flags describe enabled backends, not a measured selected kernel.'])
    with tempfile.TemporaryDirectory(prefix='p51_numeric_replay_') as temporary:
        source = Path(temporary) / 'source'
        summary['source'] = build('exp00_baseline_paired_aug_v1', source)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
            CUBLAS_WORKSPACE_CONFIG=':4096:8', NO_ALBUMENTATIONS_UPDATE='1',
            PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE='1')
        for repeat in (1, 2):
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-source', str(source),
                '--worker-output', str(output / f'replay_{repeat}.json'), '--seed', str(args.seed),
                '--mode', args.mode, '--steps', str(args.steps), '--dataset', args.dataset,
                '--hiera-checkpoint', args.hiera_checkpoint]
            print(f'Replay {repeat}/2 ({args.mode})', flush=True)
            with (output / f'replay_{repeat}.log').open('x') as log:
                with subprocess.Popen(command, env=environment, cwd=source, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    result = process.wait()
            if result:
                summary.update(status='failed', failed_replay=repeat, child_exit_code=result)
                break
        else:
            runs = [json.loads((output / f'replay_{i}.json').read_text()) for i in (1, 2)]
            difference = first_difference(
                {key: runs[0][key] for key in ('initial', 'runtime', 'steps')},
                {key: runs[1][key] for key in ('initial', 'runtime', 'steps')})
            summary.update(status='matched' if difference is None else 'different', first_difference=difference)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary.get(key) for key in ('status', 'first_difference', 'failed_replay')}, indent=2))
    print(f'Summary: {output / "summary.json"}')
    return 0 if summary['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
