"""Optional run settings applied only to a built copy of an experiment."""
import difflib
import hashlib
import json
from pathlib import Path


def apply_overrides(source, *, split_dir=None, epochs=None, patience=0, fp32=False,
                    dataset=None, hiera_checkpoint=None):
    if not any((split_dir, epochs, patience, fp32, dataset, hiera_checkpoint)):
        return
    paths=[source/n for n in ('run.py','config.json','kvasir_train.py','kvasir_test.py')]
    before={p:p.read_text() for p in paths}
    launcher=before[source/'run.py'];flags=[]
    if split_dir:
        split_dir=Path(split_dir).resolve()
        for name in ('train','val'):
            target=split_dir/(name+'.txt')
            if not target.is_file():raise FileNotFoundError(target)
            marker=f"str(HERE / 'configs/splits/kvasir/{name}.txt')"
            if marker not in launcher:raise RuntimeError('Launcher split argument changed')
            launcher=launcher.replace(marker,repr(str(target)))
        flags+=['--split-dir',str(split_dir)]
    config=json.loads(before[source/'config.json'])
    if epochs is not None:
        config['train_options']['epoch']=epochs;flags+=['--epochs',str(epochs)]
    for key,value,flag in (('DATASET',dataset,'--dataset'),('HIERA',hiera_checkpoint,'--hiera-checkpoint')):
        if value:
            value=str(Path(value).resolve());lines=launcher.splitlines(True)
            matches=[i for i,line in enumerate(lines) if line.startswith(key+' = ')]
            if len(matches)!=1:raise RuntimeError(f'Launcher constant changed: {key}')
            lines[matches[0]]=f'{key} = {value!r}\n';launcher=''.join(lines);flags += [flag,value]
            if key=='HIERA':config['train_options']['hiera_checkpoint']=value
    train=before[source/'kvasir_train.py']
    if patience:
        flags+=['--patience',str(patience)]
        marker='    since = time.time()\n';boundary='        print()\n\n    best_loss_iou ='
        if train.count(marker)!=1 or train.count(boundary)!=1:raise RuntimeError('Training epoch boundary changed')
        train=train.replace(marker,marker+'    _p51_peak, _p51_bad = float("-inf"), 0\n')
        stop=('        _p51_score = accuracy_list["valid"][-1]\n'
              '        if _p51_score > _p51_peak:\n'
              '            _p51_peak, _p51_bad = _p51_score, 0\n'
              '        else:\n'
              '            _p51_bad += 1\n'
              f'        if _p51_bad >= {patience}:\n'
              '            logging.info("Early stopping at epoch %d", epoch)\n'
              '            break\n')
        train=train.replace(boundary,'        print()\n'+stop+'\n    best_loss_iou =')
        launcher=launcher.replace("range(config['train_options']['epoch'])",
                                  "range(1 + max(r['epoch'] for r in fingerprints))")
    evaluation=before[source/'kvasir_test.py']
    if fp32:
        flags+=['--fp32']
        settings=('import torch\n'
                  'torch.set_default_dtype(torch.float32)\n'
                  'torch.set_float32_matmul_precision("highest")\n'
                  'torch.backends.cuda.matmul.allow_tf32 = False\n'
                  'torch.backends.cudnn.allow_tf32 = False\n')
        train=train.replace('import torch\n',settings,1);evaluation=evaluation.replace('import torch\n',settings,1)
    marker="    (records / 'command.txt').write_text"
    if launcher.count(marker)!=1:raise RuntimeError('Launcher recording point changed')
    launcher=launcher.replace(marker,
        '    metadata["runtime_overrides"] = '+repr(flags)+'\n'
        '    metadata["reproduce_from_repository"].extend(metadata["runtime_overrides"])\n'
        '    (records / "runtime_overrides.patch").write_text((HERE / "runtime_overrides.patch").read_text())\n'+marker)
    for name,value in (('run.py',launcher),('config.json',json.dumps(config,indent=2)+'\n'),
                       ('kvasir_train.py',train),('kvasir_test.py',evaluation)):
        (source/name).write_text(value)
    patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),
                 fromfile='a/'+p.name,tofile='b/'+p.name)) for p,old in before.items())
    (source/'runtime_overrides.patch').write_text(patch)
    manifest_path=source/'build_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['runtime_overrides']=flags;manifest['original_config_sha256']=manifest['config_sha256']
    for p in [*paths,source/'runtime_overrides.patch']:
        manifest['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest['config_sha256']=manifest['files']['config.json']
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
