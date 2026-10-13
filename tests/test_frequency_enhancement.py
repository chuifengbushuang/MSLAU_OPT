from itertools import product

import pytest
import torch

from networks.frequency_enhancement import (
    AdaptiveFrequencyEnhancement2D, FourStageFrequencyEnhancer,
    frequency_grid, symmetrize_special_columns,
)


@pytest.mark.parametrize('hw', [(88, 88), (44, 44), (22, 22), (11, 11), (13, 20), (1, 1)])
def test_fft_identity_and_half_spectrum(hw):
    x = torch.randn(2, 4, *hw)
    q = torch.fft.rfft2(x, norm='ortho')
    assert q.shape[-2:] == (hw[0], hw[1]//2+1)
    restored = torch.fft.irfft2(q, s=hw, norm='ortho')
    assert (restored-x).abs().max() < 2e-6
    module = AdaptiveFrequencyEnhancement2D(4)
    with torch.no_grad():
        module.gate_out.weight.zero_()
        module.gate_out.bias.zero_()
    torch.testing.assert_close(module(x), x, rtol=1e-5, atol=2e-6)


def test_four_stages_independence_disable_and_determinism():
    model = FourStageFrequencyEnhancer().eval()
    features = [torch.randn(2, 64, s, s) for s in (88, 44, 22, 11)]
    outputs = model(features)
    changed = model([features[0]*3, *features[1:]])
    for index, (x, y) in enumerate(zip(features, outputs)):
        assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
        assert not y.is_complex()
        assert not torch.equal(x, y)
        if index:
            assert torch.equal(changed[index], y)
    for a, b in zip(outputs, model(features)):
        assert torch.equal(a, b)
    parameter_sets = [{p.data_ptr() for p in stage.parameters()} for stage in model.stages]
    for i in range(4):
        for j in range(i):
            assert parameter_sets[i].isdisjoint(parameter_sets[j])
    disabled = FourStageFrequencyEnhancer(enabled_stages=(False, True, False, True))
    result = disabled(features)
    assert result[0] is features[0] and result[2] is features[2]
    all_off = FourStageFrequencyEnhancer(enabled_stages=(False,)*4)
    result, reg, diag = all_off.enhance_features(features, return_regularization=True, return_diagnostics=True)
    assert all(a is b for a, b in zip(result, features))
    assert reg == 0 and diag == [None]*4


def test_adaptation_not_noop_and_known_gain():
    model = AdaptiveFrequencyEnhancement2D(4)
    x = torch.randn(2, 4, 11, 18)
    weights = model.frequency_weights(x)
    assert (weights-model.frequency_weights(x*4)).abs().max() > 1e-5
    assert weights.std() > 1e-5
    with torch.no_grad():
        model.gate_out.weight.zero_()
        model.gate_out.bias.fill_(torch.atanh(torch.tensor(0.4)))
    # M=1.2; alpha=0.1 => effective gain 1.02, including DC and odd dimensions.
    torch.testing.assert_close(model(x), x*1.02, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize('hw', [(11, 11), (12, 18), (13, 20)])
def test_hermitian_consistency(hw):
    model = AdaptiveFrequencyEnhancement2D(4)
    x = torch.randn(2, 4, *hw)
    mask = model.frequency_weights(x)
    reverse = (-torch.arange(hw[0])) % hw[0]
    for column in ([0, -1] if hw[1] % 2 == 0 else [0]):
        torch.testing.assert_close(mask[..., column], mask[..., column].index_select(-1, reverse))
    # Test the projection itself with deliberately asymmetric masks.
    asymmetric = torch.rand_like(mask)
    symmetric = symmetrize_special_columns(asymmetric, hw[1])
    torch.testing.assert_close(symmetric[..., 1:-1], asymmetric[..., 1:-1])
    q = torch.fft.rfft2(x, norm='ortho') * mask
    roundtrip = torch.fft.rfft2(torch.fft.irfft2(q, s=hw, norm='ortho'), norm='ortho')
    torch.testing.assert_close(roundtrip, q, rtol=2e-5, atol=2e-6)


def test_initial_and_optimized_gradients_and_regularization():
    model = AdaptiveFrequencyEnhancement2D(4)
    x = torch.randn(2, 4, 11, 13, requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        x.grad = None
        output = model(x)
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        assert x.grad.abs().sum() > 0
        optimizer.step()
    model.zero_grad()
    _, reg, _ = model.forward_with_details(x, return_regularization=True)
    reg.backward()
    assert model.gate_in.weight.grad.abs().sum() > 0
    assert model.gate_out.weight.grad.abs().sum() > 0
    # The regularizer intentionally constrains M, not alpha.
    assert model.alpha_logit.grad is None


@pytest.mark.parametrize('value', [0., 1., 1e4, 1e-8, None])
@pytest.mark.parametrize('hw', [(11, 11), (12, 17)])
def test_fp32_stability_and_detached_diagnostics(value, hw):
    model = AdaptiveFrequencyEnhancement2D(4)
    x = torch.randn(2, 4, *hw) if value is None else torch.full((2, 4, *hw), value)
    x.requires_grad_()
    output, reg, diag = model.forward_with_details(x, return_regularization=True, return_diagnostics=True)
    assert output.dtype == x.dtype and output.shape == x.shape
    assert torch.isfinite(output).all() and torch.isfinite(reg)
    for v in diag.values():
        assert torch.isfinite(v).all() and not v.requires_grad
    (output.square().mean() + reg).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(torch.is_tensor(v) for v in vars(model).values())


def test_frequency_bands_and_empty_band_handling():
    yy, xx, radius, mult = frequency_grid(11, 12, device='cpu', dtype=torch.float32)
    assert radius[0, 0] == 0 and radius[-1, 0] == radius[1, 0]
    assert mult.sum() == 11*12
    model = AdaptiveFrequencyEnhancement2D(4)
    mask = torch.full((2, 4, 11, 7), 1.2)
    reg, means, counts = model._band_statistics(mask, radius, mult)
    torch.testing.assert_close(reg, torch.tensor(0.04))
    torch.testing.assert_close(means, torch.full((3,), 1.2))
    assert (counts > 0).all()
    x = torch.zeros(1, 4, 1, 1)
    _, reg, diag = model.forward_with_details(x, return_regularization=True, return_diagnostics=True)
    assert torch.isfinite(reg) and diag['band_counts'].tolist() == [1, 0, 0]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('value', [0., 1., 1e4, 1e-8, None])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp(dtype, value):
    model = AdaptiveFrequencyEnhancement2D(4).cuda()
    x = torch.randn(2, 4, 11, 17, device='cuda') if value is None else torch.full((2, 4, 11, 17), value, device='cuda')
    x = x.to(dtype).requires_grad_()
    with torch.autocast('cuda', dtype=dtype):
        output, reg, diag = model.forward_with_details(x, return_regularization=True, return_diagnostics=True)
        loss = output.float().square().mean() + reg
    assert output.dtype == dtype and torch.isfinite(output).all()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_configuration():
    with pytest.raises(ValueError):
        AdaptiveFrequencyEnhancement2D(band_edges=(.7, .3))
    with pytest.raises(ValueError):
        AdaptiveFrequencyEnhancement2D(alpha_init=0)
    with pytest.raises(ValueError):
        FourStageFrequencyEnhancer(enabled_stages=(True,)*3)
    with pytest.raises(ValueError):
        AdaptiveFrequencyEnhancement2D(4)(torch.rand(2, 3, 11, 11))


@pytest.mark.parametrize('enabled', list(product((False, True), repeat=4)))
def test_all_stage_masks_preserve_bypass_and_gradient_routes(enabled):
    model = FourStageFrequencyEnhancer(enabled_stages=enabled)
    features = [torch.randn(1, 64, s, s, requires_grad=True) for s in (88, 44, 22, 11)]
    outputs, reg, diagnostics = model.enhance_features(
        features, return_regularization=True, return_diagnostics=True)
    sum(y.square().mean() for y in outputs).backward()
    for active, stage, x, y, diag in zip(enabled, model.stages, features, outputs, diagnostics):
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        if active:
            assert diag is not None
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0 for p in stage.parameters())
        else:
            assert y is x and diag is None
            assert all(p.grad is None for p in stage.parameters())
    active_regs = [d['regularization'] for d in diagnostics if d is not None]
    expected = torch.stack(active_regs).mean() if active_regs else reg.new_zeros(())
    torch.testing.assert_close(reg.detach(), expected)


@pytest.mark.parametrize('training', [False, True])
def test_each_stage_has_no_forward_or_backward_cross_stage_dependency(training):
    model = FourStageFrequencyEnhancer().train(training)
    features = [torch.randn(1, 64, s, s, requires_grad=True) for s in (88, 44, 22, 11)]
    with torch.no_grad():
        reference = model(features)
        for changed_index in range(4):
            altered = list(features)
            altered[changed_index] = altered[changed_index] * 3
            outputs = model(altered)
            for index in range(4):
                if index != changed_index:
                    assert torch.equal(reference[index], outputs[index])
    for selected in range(4):
        model.zero_grad(set_to_none=True)
        for x in features:
            x.grad = None
        model(features)[selected].square().mean().backward()
        for index, (x, stage) in enumerate(zip(features, model.stages)):
            if index == selected:
                assert x.grad is not None and x.grad.abs().sum() > 0
                assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters())
            else:
                assert x.grad is None
                assert all(p.grad is None for p in stage.parameters())
