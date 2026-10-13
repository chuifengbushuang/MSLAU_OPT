#!/usr/bin/env bash
# Read-only checks. Does not start training or claim any GPU is available.
# Usage: bash check_affe_environment.sh [repo] [python] [model_root]
repo_dir=${1:-/home/zsy/projects/mslau-net}
python_bin=${2:-/home/zsy/miniconda3/envs/mslau-net/bin/python}
model_root=${3:-/data/models/zsy/mslau-net}
nvidia-smi
ps -u zsy -u le -o user,pid,etime,args | grep -E 'python|train|kvasir' || true
if [ -d "$repo_dir" ]; then
    git -C "$repo_dir" branch --show-current
    git -C "$repo_dir" status --short
    git -C "$repo_dir" rev-parse HEAD
else
    echo "Repository directory not found: $repo_dir"
fi
if [ -x "$python_bin" ]; then
    "$python_bin" - <<'PY'
import sys
import torch
import timm
print('Python:', sys.executable, sys.version)
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('timm:', timm.__version__, 'CUDA available:', torch.cuda.is_available())
PY
else
    echo "Python executable not found: $python_bin"
    command -v python python3 conda || true
fi
if [ -d "$model_root/pretrained" ]; then
    find "$model_root/pretrained" -maxdepth 1 -iname '*hiera*' -ls
fi
if [ -d "$model_root/runs" ]; then
    find "$model_root/runs" -path '*p5a*' -name '*.pth' -print
fi
