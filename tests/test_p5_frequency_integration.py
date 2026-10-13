import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
import torch

from networks.mslau_net import SAM2UNetDecoder, MSLAU_net, HieraBlockAdapter
from networks.frequency_enhancement import load_frequency_state_dict
from networks.frequency_config import add_frequency_arguments, frequency_model_kwargs
from loss import StructureLoss_binary


@pytest.fixture(scope='module')
def old_code(tmp_path_factory):
    # An independent original decoder, not two instances of the edited implementation.
    path = tmp_path_factory.mktemp('baseline') / 'baseline.py'
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', '81b19e2:networks/mslau_net.py'], cwd=root, text=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('original_p51', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_features(batch=2):
    return [torch.randn(batch, c, s, s, requires_grad=True)
            for c, s in zip((144, 288, 576, 1152), (88, 44, 22, 11))]


def test_original_none_exact_and_checkpoint_migration(old_code):
    torch.manual_seed(1234)
    original = old_code.SAM2UNetDecoder().eval()
    torch.manual_seed(1234)
    unchanged = SAM2UNetDecoder().eval()
    assert set(original.state_dict()) == set(unchanged.state_dict())
    for key, value in original.state_dict().items():
        assert torch.equal(value, unchanged.state_dict()[key]), key
    audit = load_frequency_state_dict(unchanged, original.state_dict())
    assert not audit['missing_keys'] and not audit['unexpected_keys']
    features = synthetic_features(1)
    with torch.no_grad():
        expected = original(features, 352, return_aux=True)
        actual = unchanged(features, 352, return_aux=True)
    assert set(actual) == {'logits', 'aux_d2', 'aux_d3'}
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    enhanced = SAM2UNetDecoder(frequency_mode='naf2d').eval()
    audit = load_frequency_state_dict(enhanced, original.state_dict(), allow_legacy=True)
    assert audit['legacy_migration'] and len(audit['missing_keys']) == 20
    assert all(key.startswith('frequency_enhancer.') for key in audit['missing_keys'])
    load_frequency_state_dict(enhanced, enhanced.state_dict())
    for corrupt in [dict(list(original.state_dict().items())[1:]),
                    {**original.state_dict(), 'bad_key': torch.tensor(0)},
                    {**original.state_dict(), 'head.weight': torch.zeros(2)}]:
        with pytest.raises(RuntimeError, match='audit failed'):
            load_frequency_state_dict(enhanced, corrupt, allow_legacy=True)
    partial = dict(enhanced.state_dict())
    partial.pop('frequency_enhancer.stages.0.alpha_logit')
    with pytest.raises(RuntimeError):
        load_frequency_state_dict(enhanced, partial, allow_legacy=True)


def test_complete_decoder_structure_loss_and_gradients():
    model = SAM2UNetDecoder(frequency_mode='naf2d')
    features = synthetic_features()
    criterion = StructureLoss_binary()
    target = (torch.rand(2, 1, 352, 352) > .5).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        outputs = model(features, 352, return_aux=True, return_frequency=True,
                        return_frequency_diagnostics=step == 0)
        loss = sum(criterion(outputs[key], target) for key in ('logits', 'aux_d2', 'aux_d3'))
        if step:
            loss = loss + .01 * outputs['frequency_regularization']
        loss.backward()
        for key in ('logits', 'aux_d2', 'aux_d3'):
            assert outputs[key].shape == (2, 1, 352, 352)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if 'frequency_enhancer' in name or 'rfbs' in name:
                assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
    assert all(feature.grad.abs().sum() > 0 for feature in features)


def test_cli_defaults_and_invalid_combinations():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_mode', default='p5_sam2unet')
    add_frequency_arguments(parser)
    args = parser.parse_args([])
    assert frequency_model_kwargs(args)['frequency_mode'] == 'none'
    assert args.frequency_reg_weight == 0
    args = parser.parse_args(['--frequency_mode', 'naf2d', '--frequency_enabled_stages', '1,0,1,0'])
    assert frequency_model_kwargs(args)['frequency_enabled_stages'] == (True, False, True, False)
    with pytest.raises(SystemExit):
        parser.parse_args(['--frequency_enabled_stages', '1,0'])
    with pytest.raises(ValueError):
        frequency_model_kwargs(parser.parse_args(['--frequency_reg_weight', '.1']))
    with pytest.raises(ValueError):
        MSLAU_net(frequency_mode='naf2d')


@pytest.mark.skipif(os.environ.get('AFFE_FULL_SMOKE') != '1', reason='opt-in real Hiera synthetic smoke')
def test_real_hiera_freeze_adapter_optimizer_and_full_forward():
    device = os.environ.get('AFFE_DEVICE', 'cpu')
    torch.manual_seed(1234)
    model = MSLAU_net(img_size=352, encoder_name='sam2_hiera_large',
                      decoder_mode='p5_sam2unet', frequency_mode='naf2d',
                      hiera_checkpoint=os.environ.get('HIERA_CHECKPOINT')).to(device).train()
    assert sum(isinstance(m, HieraBlockAdapter) for m in model.modules()) == 48
    # Execute the parameter-group expressions from the actual training entry point.
    import ast
    source = ast.parse((Path(__file__).resolve().parents[1] / 'kvasir_train.py').read_text())
    groups = {}
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('encoder_params', 'decoder_params'):
                groups[node.targets[0].id] = eval(compile(ast.Expression(node.value), '<optimizer groups>', 'eval'), {'model': model})
    optimizer = torch.optim.AdamW([{'params': groups['encoder_params']}, {'params': groups['decoder_params']}], lr=1e-3)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.encoder.named_parameters() if not p.requires_grad}
    # Exact snapshots of all frozen tensors: move to CPU to limit GPU footprint.
    originals = {n: p.detach().cpu().clone() for n, p in frozen.items()}
    image = torch.randn(1, 3, 352, 352, device=device)
    target = (torch.rand(1, 1, 352, 352, device=device) > .5).float()
    criterion = StructureLoss_binary()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, return_aux=True, return_frequency=True)
        loss = sum(criterion(outputs[k], target) for k in ('logits', 'aux_d2', 'aux_d3'))
        if step:
            loss = loss + .01*outputs['frequency_regularization']
        loss.backward()
        assert all(outputs[k].shape == (1, 1, 352, 352) for k in ('logits', 'aux_d2', 'aux_d3'))
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                assert p.grad.abs().sum() > 0, name
            else:
                assert p.grad is None, name
        optimizer.step()
    assert all(torch.equal(p.detach().cpu(), originals[n]) for n, p in frozen.items())


@pytest.mark.skipif(not os.environ.get('P51_CHECKPOINT'), reason='real P5.1 checkpoint not supplied')
def test_real_checkpoint_audit():
    state = torch.load(os.environ['P51_CHECKPOINT'], map_location='cpu', weights_only=True)
    model = MSLAU_net(img_size=352, encoder_name='sam2_hiera_large', decoder_mode='p5_sam2unet')
    audit = load_frequency_state_dict(model, state)
    assert not audit['missing_keys'] and not audit['unexpected_keys']
    del model
    model = MSLAU_net(img_size=352, encoder_name='sam2_hiera_large', decoder_mode='p5_sam2unet', frequency_mode='naf2d')
    audit = load_frequency_state_dict(model, state, allow_legacy=True)
    assert len(audit['missing_keys']) == 20 and not audit['unexpected_keys']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_decoder_cuda_amp(dtype):
    model = SAM2UNetDecoder(frequency_mode='naf2d').cuda()
    features = [x.detach().cuda().requires_grad_() for x in synthetic_features(1)]
    with torch.autocast('cuda', dtype=dtype):
        output = model(features, 352, return_aux=True, return_frequency=True)
        loss = sum(output[k].float().square().mean() for k in ('logits', 'aux_d2', 'aux_d3'))
        loss = loss + .01*output['frequency_regularization']
    loss.backward()
    assert all(torch.isfinite(output[k]).all() for k in ('logits', 'aux_d2', 'aux_d3'))
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(torch.isfinite(x.grad).all() for x in features)


def test_all_disabled_decoder_matches_original_three_heads(old_code):
    original = old_code.SAM2UNetDecoder().eval()
    model = SAM2UNetDecoder(frequency_mode='naf2d', frequency_enabled_stages=(False,)*4).eval()
    load_frequency_state_dict(model, original.state_dict(), allow_legacy=True)
    features = synthetic_features(1)
    with torch.no_grad():
        expected = original(features, 352, return_aux=True)
        actual = model(features, 352, return_aux=True, return_frequency=True,
                       return_frequency_diagnostics=True)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert actual['frequency_regularization'] == 0
    assert actual['frequency_diagnostics'] == [None]*4


def test_regularizer_only_reaches_rfb_and_gates_not_alpha_or_decoder():
    model = SAM2UNetDecoder(frequency_mode='naf2d')
    features = synthetic_features(1)
    outputs = model(features, 352, return_aux=True, return_frequency=True)
    outputs['frequency_regularization'].backward()
    for name, parameter in model.named_parameters():
        if name.startswith('rfbs.') or ('frequency_enhancer.' in name and 'alpha_logit' not in name):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
               for x in features)
