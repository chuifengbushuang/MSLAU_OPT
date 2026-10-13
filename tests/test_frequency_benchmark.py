import json

import torch

from networks.frequency_enhancement import AdaptiveFrequencyEnhancement2D
from tools.verify_frequency import measure, stage_evidence


def test_diagnostics_and_benchmark_do_not_retain_gradients_or_tensor_results():
    stage = AdaptiveFrequencyEnhancement2D(4)
    x = torch.randn(1, 4, 11, 13, requires_grad=True)
    evidence = stage_evidence(stage, x)
    json.dumps(evidence)  # Evidence must not keep tensors or their computation graphs.
    assert all(value > 0 for value in evidence['initial_gradient_l1'].values())
    assert x.grad is None and all(p.grad is None for p in stage.parameters())
    # Simulate an earlier backward before measuring forward/backward in turn.
    stage(x).square().mean().backward()
    calls = []

    def forward():
        assert x.grad is None and all(p.grad is None for p in stage.parameters())
        calls.append(torch.is_grad_enabled())
        return stage(x)

    result = measure(stage, forward, torch.device('cpu'), 1, 2, inputs=(x,))
    assert calls == [False]*3 + [True]*3
    assert set(result) == {'forward', 'forward_backward'}
    assert x.grad is None and all(p.grad is None for p in stage.parameters())
