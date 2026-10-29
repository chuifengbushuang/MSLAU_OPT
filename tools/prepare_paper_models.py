"""Fetch pinned author code/weights and isolated dependencies; never train.

Run with the existing work/local_train_env/bin/python. Default verifies existing
assets. --download and --install-dependencies prepare a fresh workspace without
updating the original environment. All downloads are public research assets.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEFINITION=ROOT/'studies/paper_models'
ASSETS=ROOT.parent/'work/paper_models_20261030'
URLS=dict(polyper='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth',
    eat='https://raw.githubusercontent.com/deepang-ai/EAT/c699f7e5e5f457d7fd5789f1e309e380048142f8/pretrained/pvt_v2_b2.pth',
    condseg='https://download.pytorch.org/models/resnet50-19c8e357.pth',
    asgnet='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--install-dependencies',action='store_true')
    args=parser.parse_args(argv)
    pins=json.loads((DEFINITION/'upstreams.json').read_text())
    ASSETS.mkdir(parents=True,exist_ok=True)
    for model in ['polyper','eat','condseg','asgnet']:
        pin=pins[model];directory=ASSETS/'vendor'/model
        if not directory.exists():
            if not args.download:raise FileNotFoundError(f'Missing {directory}; use --download')
            url=f'https://codeload.github.com/{pin["repo"]}/zip/{pin["commit"]}'
            archive=urllib.request.urlopen(url,timeout=120).read()
            files={}
            zip_source=zipfile.ZipFile(io.BytesIO(archive))
            for info in zip_source.infolist():
                relative=Path(*Path(info.filename).parts[1:])
                if not relative.parts or info.is_dir() or '__pycache__' in relative.parts or relative.suffix=='.pyc':continue
                if '..' in relative.parts or relative.is_absolute():raise ValueError('Invalid archive path')
                destination=directory/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                content=zip_source.read(info)
                destination.write_bytes(content);files[str(relative)]=hashlib.sha256(content).hexdigest()
            (directory/'upstream_manifest.json').write_text(json.dumps(dict(repo=pin['repo'],commit=pin['commit'],
                archive_sha256=hashlib.sha256(archive).hexdigest(),files=files),indent=2)+'\n')
        manifest=json.loads((directory/'upstream_manifest.json').read_text())
        if manifest['commit']!=pin['commit']:raise RuntimeError(f'Wrong revision for {model}')
        for name,sha in manifest['files'].items():
            if digest(directory/name)!=sha:raise RuntimeError(f'Changed author file: {model}/{name}')
        weight=ASSETS/pin['weights']
        if not weight.exists():
            if not args.download:raise FileNotFoundError(weight)
            weight.parent.mkdir(parents=True,exist_ok=True)
            temporary=weight.with_suffix('.download')
            with urllib.request.urlopen(URLS[model],timeout=120) as response,temporary.open('xb') as stream:
                while b:=response.read(4*1024*1024):stream.write(b)
            if digest(temporary)!=pin['weight_sha256']:raise RuntimeError('Downloaded weights have wrong SHA256')
            temporary.rename(weight)
        if digest(weight)!=pin['weight_sha256']:raise RuntimeError(f'Wrong pretrained weights: {model}')
        print('VERIFIED',model,pin['commit'],flush=True)
    dependencies=ASSETS/'deps'
    required={line.split('==')[0].lower():line.split('==')[1] for line in
        (DEFINITION/'requirements.txt').read_text().splitlines() if line and not line.startswith('#')}
    installed={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions(path=[str(dependencies)])}
    missing=[name+'=='+version for name,version in required.items() if installed.get(name)!=version]
    if missing:
        if not args.install_dependencies:raise RuntimeError(f'Missing isolated dependencies: {missing}')
        subprocess.run([sys.executable,'-m','pip','install','--target',str(dependencies),'--no-deps',*missing],check=True)
    print(json.dumps(dict(status='verified',models=4,dependencies=str(dependencies),training_started=False)))


if __name__=='__main__':main()
