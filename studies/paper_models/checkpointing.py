"""Activation checkpointing without double-updating BatchNorm running state."""
from contextlib import contextmanager, nullcontext
import types
import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_running_buffers(module):
    # Capture at recomputation entry, after the original forward's legitimate
    # update. BatchNorm training uses batch statistics; replay must leave exactly
    # the original running means/variances/counters behind.
    saved = [(b,b.detach().clone()) for child in module.modules()
             if isinstance(child,torch.nn.modules.batchnorm._BatchNorm)
             for b in child.buffers(recurse=False) if b is not None]
    try:
        yield
    finally:
        for buffer,value in saved:
            # Native BN updates running buffers without autograd version bumps.
            # Restore the same non-differentiated state without invalidating its
            # saved backward tensors. No parameter is modified here.
            buffer.data.copy_(value)


def checkpoint_module(module):
    original = module.forward
    def forward(self,*args,**kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(original,*args,use_reentrant=False,preserve_rng_state=True,
                context_fn=lambda:(nullcontext(),preserve_running_buffers(self)),**kwargs)
        return original(*args,**kwargs)
    module.forward = types.MethodType(forward,module)


def checkpoint_condseg(network):
    names = ['layer0','layer1','layer2','layer3','output_block',
        'decoder_small','decoder_middle','decoder_large']
    names += [f'dconv{i}' for i in range(1,5)]
    names += [f'cdfa{i}' for i in range(1,5)]
    names += [f'preprocess_{kind}{i}' for kind in ('fg','bg') for i in range(1,5)]
    for name in names:checkpoint_module(getattr(network,name))
    for branch in ('branch_fg','branch_bg','branch_uc'):
        checkpoint_module(getattr(network.aux_head,branch))
    return names+['aux_head.'+b for b in ('branch_fg','branch_bg','branch_uc')]
