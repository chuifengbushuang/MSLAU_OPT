"""Fixed B/NAF3/H study on the joint holdout; reuse existing training code."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import local_decoder_study as common
import exp03_independent_study as p12
import freqselect_study as shell
from check_baseline_numerics import rng_record

ROOT, WORKSPACE, torch = common.ROOT, common.WORKSPACE, common.torch
VARIANTS = ('baseline', 'exp03_independent', 'freqselect_h')
EXPERIMENTS = {name: p12.EXPERIMENT if name == 'exp03_independent'
               else 'exp00_baseline_paired_strict_v2' for name in VARIANTS}
SEEDS = (1234, 42, 2026)
SPLITS = ROOT / 'configs/splits/kvasir_clinicdb_holdout_v1'
DATASET = WORKSPACE / 'datasets/Polyp-Mixed-v1'
OUTPUT = WORKSPACE / 'work/joint_polyp_holdout_20261029'
OUT = WORKSPACE / 'analysis/20261029_joint_polyp_holdout'
ORIGINAL_TRAIN, ORIGINAL_SMOKE = common.train, common.smoke
ORIGINAL_APPEND, ORIGINAL_LOSSES = common.append_json, common.losses


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def locate(mode, variant, seed):
    paths = list((OUTPUT / 'experiments').glob(f'local_{mode}_{variant}_seed{seed}_*'))
    if len(paths) > 1:
        raise RuntimeError(f'Multiple runs require inspection: {paths}')
    return paths[0] if paths else None


def check_data():
    manifest = read(SPLITS / 'manifest.json')
    if manifest['counts'] != dict(train=1253, val=194, test=165):
        raise RuntimeError('Joint split counts changed')
    for name, digest in manifest['files_sha256'].items():
        if common.file_hash(SPLITS / name) != digest:
            raise RuntimeError(f'Frozen joint split changed: {name}')
    for name, digest in manifest['original_split_files_sha256'].items():
        if common.file_hash(ROOT / name) != digest:
            raise RuntimeError(f'Original split changed: {name}')
    with (SPLITS / 'samples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            for kind in ('image', 'mask'):
                original = WORKSPACE / row[kind + '_path']
                via_link = DATASET / ('images' if kind == 'image' else 'masks') / row['sample_id']
                if via_link.resolve() != original.resolve() or common.file_hash(original) != row[kind + '_sha256']:
                    raise RuntimeError(f'Dataset view/content changed: {via_link}')
    return manifest


def make_model(source, args):
    if args.variant == 'exp03_independent':
        return p12.make_model(source, args)
    if args.variant == 'freqselect_h':
        return shell.make_model(source, args)
    return common.make_model(source, args)


def configure(args, records, source):
    check_data()
    for path in (Path(__file__), Path(p12.__file__), Path(shell.__file__), ROOT / 'tools/check_baseline_numerics.py'):
        shutil.copy2(path, records / path.name)
    shutil.copytree(SPLITS, records / 'splits')
    if args.variant == 'exp03_independent':
        common.gradient_report, common.check_gradients = p12.gradient_report, p12.check_gradients
    elif args.variant == 'freqselect_h':
        common.gradient_report, common.check_gradients = shell.gradient_report, shell.check_gradients
        shutil.copy2(shell.CORE, source / 'networks/freqselect_core.py')
    common.set_context = lambda model, args, epoch, ids: None
    common.losses = shell.losses if args.variant == 'freqselect_h' else ORIGINAL_LOSSES
    expected = None
    if args.mode == 'train' and args.variant != 'baseline':
        baseline = locate('train', 'baseline', args.seed)
        if baseline is None or not (baseline / 'result.json').is_file():
            raise RuntimeError('The matching joint baseline must finish first')
        expected = lines(baseline / 'input_fingerprints.jsonl')

    def append(path, value):
        if Path(path).name == 'input_fingerprints.jsonl':
            index = 2 * value['epoch'] + (value['phase'] == 'valid')
            if expected is not None and value != expected[index]:
                raise RuntimeError(f'Joint paired input mismatch: {value}')
            frame = sys._getframe(1).f_locals
            model, optimizer, scheduler = (frame[key] for key in ('model', 'optimizer', 'scheduler'))
            digest = frame['paired'].state_digest
            before = rng_record(torch)
            moments = {name + '.' + key: value for name, parameter in model.named_parameters()
                       for key, value in optimizer.state.get(parameter, {}).items() if torch.is_tensor(value)}
            state = dict(epoch=value['epoch'], phase=value['phase'],
                position='after_phase_metrics_and_input_digest_before_scheduler_and_checkpoint',
                full_model_sha256=digest(model.state_dict()), buffers_sha256=digest(dict(model.named_buffers())),
                optimizer_tensors_sha256=digest(moments), optimizer_param_groups=optimizer.state_dict()['param_groups'],
                scheduler=scheduler.state_dict(), rng=before)
            if before != rng_record(torch):
                raise RuntimeError('Read-only observation changed RNG')
            ORIGINAL_APPEND(records / 'numerical_states.jsonl', state)
        ORIGINAL_APPEND(path, value)

    def smoke(*values):
        result = ORIGINAL_SMOKE(*values)
        model = values[0]
        temporary = records / 'smoke_full_state.pth'
        torch.save(model.state_dict(), temporary)
        model.load_state_dict(torch.load(temporary, map_location='cpu', weights_only=True), strict=True)
        temporary.unlink()
        common.write_json(records / 'full_reload.json', dict(strict_full_state_reload=True))
        return result

    common.append_json, common.smoke = append, smoke


def record_initial(model, paired, records, args):
    initial = dict(full_model_sha256=paired.state_digest(model.state_dict()),
        original_model_sha256=paired.state_digest({name: value for name, value in model.state_dict().items()
                                                  if '.frequency_enhancer.' not in name}), rng=rng_record(torch))
    baseline = locate(args.mode, 'baseline', args.seed)
    if baseline is not None and baseline != records:
        previous = read(baseline / 'study_initial_state.json')
        if any(initial[key] != previous[key] for key in ('original_model_sha256', 'rng')):
            raise RuntimeError('Original layers or complete initial RNG differ from joint baseline')
    common.write_json(records / 'study_initial_state.json', initial)


def before_execution(model, loaders, args, records):
    if args.variant == 'freqselect_h':
        config = read(records / 'effective_config.json')
        config['naf_reg_weight'] = 0
        common.write_json(records / 'effective_config.json', config)
    runtime = common.load_module('joint_strict_runtime', records / 'source/strict_runtime.py')
    common.write_json(records / 'initial_runtime.json', dict(runtime=runtime.runtime_record(), rng=rng_record(torch)))
    common.write_json(records / 'study_manifest.json', dict(
        split_manifest_sha256=common.file_hash(SPLITS / 'manifest.json'),
        source_files={str(path.relative_to(records / 'source')): common.file_hash(path)
                      for path in (records / 'source').rglob('*') if path.is_file() and '__pycache__' not in path.parts},
        reused_train_loop_sha256=common.file_hash(Path(common.__file__)),
        train_function_bytecode_sha256=common.hashlib.sha256(ORIGINAL_TRAIN.__code__.co_code).hexdigest()))
    if args.mode == 'smoke':
        if args.variant == 'exp03_independent':
            p12.audit_original_p12(model, loaders, args, records)
        elif args.variant == 'freqselect_h':
            shell.zero_projection_parity(model, loaders, args, records)


def validate(path, mode):
    result = read(path / ('smoke.json' if mode == 'smoke' else 'result.json'))
    if result['status'] != 'completed':
        raise RuntimeError(f'Incomplete run: {path}')
    manifest = read(path / 'study_manifest.json')
    if any(common.file_hash(path / 'source' / name) != digest for name, digest in manifest['source_files'].items()):
        raise RuntimeError('Source snapshot hash mismatch')
    if mode == 'smoke':
        if not (result['repeated_eval_equal'] and result['checkpoint_roundtrip_equal']
                and read(path / 'full_reload.json')['strict_full_state_reload']):
            raise RuntimeError('Smoke repeat/reload check failed')
    else:
        if len(lines(path / 'epoch_metrics.jsonl')) != 20 or len(lines(path / 'numerical_states.jsonl')) != 40:
            raise RuntimeError('Missing epochs/phase states')
        fingerprints = lines(path / 'input_fingerprints.jsonl')
        if len(fingerprints) != 40 or any(r['samples'] != (1248 if r['phase'] == 'train' else 194) for r in fingerprints):
            raise RuntimeError('Missing/incorrect input audit')
        if len(result['checkpoints']) != 3 or abs(result['best_iou'] - result['evaluation']['iou']) > 1e-6:
            raise RuntimeError('Checkpoint selection/reload mismatch')
        for name, digest in result['checkpoints'].items():
            if common.file_hash(OUTPUT / 'checkpoints' / path.name / name) != digest:
                raise RuntimeError('Checkpoint hash mismatch')
    return result


def evaluate_all():
    import gc
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from zoneinfo import ZoneInfo
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = locate('train', variant, seed)
            validate(path, 'train')
            source = path / 'source'
            args = argparse.Namespace(variant=variant, hiera_checkpoint=WORKSPACE / 'checkpoints/sam2_hiera_large.fb_r1024.safetensors', precision='bf16')
            # Each model imports its own archived source; no cached networks from another arm.
            for name in list(sys.modules):
                if name == 'networks' or name.startswith('networks.') or name in ('loader', 'paired_inputs'):
                    del sys.modules[name]
            sys.path.insert(0, str(source))
            paired = common.load_module('joint_evaluation_paired', source / 'paired_inputs.py')
            factories = common.definitions(source / 'kvasir_train.py', {'get_valid_transform'}, dict(A=A, ToTensorV2=ToTensorV2))
            factories.update(common.definitions(source / 'kvasir_test.py', {'compute_batch_metrics'}, dict(torch=torch)))
            model = make_model(source, args)
            checkpoint = OUTPUT / 'checkpoints' / path.name / 'kvasir_best_model.pth'
            before = common.file_hash(checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            model.cuda().eval()
            for split in ('val', 'test'):
                ids = (path / 'splits' / f'{split}.txt').read_text().splitlines()
                dataset = paired.PairedDataset(str(DATASET), ids, factories['get_valid_transform'](352), seed, 'valid')
                loader = paired.make_loader(dataset, 6, 2, pin_memory=True)
                per_image = []
                with torch.inference_mode(), common.autocast(args):
                    for images, masks, names in loader:
                        values = factories['compute_batch_metrics']((model(images.cuda()).float().sigmoid() > .5).float(), masks.cuda().unsqueeze(1))
                        for index, name in enumerate(names):
                            per_image.append(dict(sample_id=name, dataset=name.split('/')[0],
                                                  **{key: value[index] for key, value in values.items()}))
                metric_keys = [key for key in per_image[0] if key not in ('sample_id', 'dataset')]
                summary = {key: statistics.mean(row[key] for row in per_image) for key in metric_keys}
                by_source = {name: {key: statistics.mean(row[key] for row in per_image if row['dataset'] == name)
                                   for key in metric_keys} for name in ('Kvasir-SEG', 'CVC-ClinicDB')}
                if split == 'val' and abs(summary['iou'] - read(path / 'result.json')['best_iou']) > 1e-6:
                    raise RuntimeError('Final validation reevaluation mismatch')
                common.write_json(path / f'final_{split}_evaluation.json', dict(metrics=summary, by_dataset=by_source, per_image=per_image))
                rows.append(dict(seed=seed, variant=variant, split=split, **summary, by_dataset=by_source))
            if common.file_hash(checkpoint) != before:
                raise RuntimeError('Inference changed checkpoint')
            del model, loader, dataset
            sys.path.remove(str(source))
            gc.collect(); torch.cuda.empty_cache()
    aggregate = {}
    for variant in VARIANTS:
        chosen = [row for row in rows if row['variant'] == variant and row['split'] == 'test']
        aggregate[variant] = {key: dict(mean=statistics.mean(row[key] for row in chosen),
                                      sample_sd=statistics.stdev(row[key] for row in chosen)) for key in ('iou', 'dice')}
    differences = [dict(seed=seed, variant=variant,
        iou_delta_pp=100*(next(row['iou'] for row in rows if row['seed']==seed and row['variant']==variant and row['split']=='test')
                         - next(row['iou'] for row in rows if row['seed']==seed and row['variant']=='baseline' and row['split']=='test')))
        for seed in SEEDS for variant in VARIANTS[1:]]
    common.write_json(OUT / 'summary.json', dict(status='completed', runs=rows, test_aggregate=aggregate,
        paired_against_baseline=differences, limitations='Internal holdout; historical development used Kvasir. CVC test has 4 local sequences.',
        completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(ROOT / 'studies/clinicdb/joint_training_plan.json')
    check_data()
    watched = [Path(__file__), Path(common.__file__), Path(p12.__file__), Path(shell.__file__),
               p12.CORE, shell.CORE, common.NAF_FILE, ROOT / 'studies/clinicdb/joint_training_plan.json']
    hashes = {str(path): common.file_hash(path) for path in watched}
    if (OUT / 'plan.json').exists():
        raise FileExistsError('Existing supervisor output needs inspection; no automatic rerun')
    common.write_json(OUT / 'plan.json', dict(plan, implementation_sha256=hashes))
    completed = []
    jobs = [('smoke', variant, 1234) for variant in VARIANTS]
    jobs += [('train', variant, seed) for seed in SEEDS for variant in VARIANTS]
    for mode, variant, seed in jobs:
        if any(common.file_hash(Path(name)) != digest for name, digest in hashes.items()):
            raise RuntimeError('Implementation changed after launch')
        if locate(mode, variant, seed) is not None:
            raise RuntimeError('Existing worker output needs inspection; no retry or selection')
        command = [sys.executable, '-u', str(Path(__file__)), 'worker', '--mode', mode, '--variant', variant,
            '--seed', str(seed), '--epochs', '20', '--batch', '12', '--workers', '2', '--threads', '4',
            '--precision', 'bf16', '--strict', '--dataset', str(DATASET), '--split-dir', str(SPLITS), '--output-root', str(OUTPUT)]
        log = OUT / f'{mode}_{variant}_seed{seed}.console.log'
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            common.write_json(OUT / 'progress.json', dict(status=mode, variant=variant, seed=seed,
                child_pid=child.pid, supervisor_pid=os.getpid(), console=str(log), completed=completed))
            print('START', mode, variant, seed, child.pid, flush=True)
            if child.wait():
                raise RuntimeError(f'Worker failed; inspect {log}')
        path = locate(mode, variant, seed)
        result = validate(path, mode)
        completed.append(dict(mode=mode, variant=variant, seed=seed, records=str(path)))
        common.write_json(OUT / 'audit.json', dict(completed=completed))
        print('DONE', mode, variant, seed, result.get('best_iou'), flush=True)
    common.write_json(OUT / 'progress.json', dict(status='final_evaluation', completed=completed))
    evaluate_all()
    common.write_json(OUT / 'progress.json', dict(status='completed', completed=completed, summary=str(OUT / 'summary.json')))


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'worker':
            common.main(sys.argv[2:], study=sys.modules[__name__])
        else:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument('--run', action='store_true')
            args = parser.parse_args()
            if args.run:
                run()
            else:
                print('PLAN: B/NAF3/H, seeds1234/42/2026, 20epochs; smoke first; pooled194 validation, final165 test.')
    except BaseException as error:
        if '--run' in sys.argv and OUT.is_dir():
            previous = read(OUT / 'progress.json') if (OUT / 'progress.json').exists() else {}
            common.write_json(OUT / 'progress.json', dict(previous, status='failed', error=str(error)))
        raise
