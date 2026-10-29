"""Equivalent global max pooling with a deterministic first-maximum gradient."""
import torch
from torch import nn


class GlobalMaxPool2d(nn.Module):
    def forward(self,x):
        return x.flatten(2).max(dim=-1).values[...,None,None]


def replace_global_max_pool(module):
    count = 0
    for name,child in list(module.named_children()):
        if isinstance(child,nn.AdaptiveMaxPool2d) and child.output_size in (1,(1,1)):
            if child.return_indices:
                raise ValueError('Return-indices max pool requires a different adapter')
            setattr(module,name,GlobalMaxPool2d())
            count += 1
        else:
            count += replace_global_max_pool(child)
    return count
