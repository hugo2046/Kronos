"""固定参数诊断的数值及恢复契约。"""
import pytest
import torch
from state_adapter.sam_mechanism import bounded_ascent


def test_quadratic_ascent_radius_and_restore():
    p = torch.nn.Parameter(torch.tensor([1., 0.], dtype=torch.float64))
    p.grad = torch.tensor([7., 8.], dtype=torch.float64)
    original = p.detach().clone()
    def objective(backward):
        loss = p.square().sum()
        if backward:
            loss.backward()
        return float(loss.detach())
    r = bounded_ascent([p], objective, rho=.2)
    assert r['best_loss'] == pytest.approx(1.44)
    assert torch.linalg.vector_norm(r['delta'][0]).item() <= .2 + 1e-12
    assert torch.equal(p, original)
    assert torch.equal(p.grad, torch.tensor([7., 8.], dtype=torch.float64))


def test_zero_radius_and_rng_restore():
    p = torch.nn.Parameter(torch.tensor([1.]))
    rng = torch.get_rng_state().clone()
    def objective(backward):
        torch.rand(1)
        loss = p.square().sum()
        if backward:
            loss.backward()
        return float(loss.detach())
    r = bounded_ascent([p], objective, rho=0)
    assert r['best_loss'] == 1.
    assert torch.equal(r['delta'][0], torch.zeros_like(p))
    assert torch.equal(torch.get_rng_state(), rng)
    assert p.grad is None


def test_exception_restores_parameter_and_gradient():
    p = torch.nn.Parameter(torch.tensor([1.]))
    calls = 0
    def objective(backward):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise RuntimeError('injected')
        loss = p.square().sum()
        if backward:
            loss.backward()
        return float(loss.detach())
    with pytest.raises(RuntimeError, match='injected'):
        bounded_ascent([p], objective, rho=.2)
    assert p.item() == 1.
    assert p.grad is None


def test_sharpe_symmetric_decomposition():
    from state_adapter.sam_mechanism_report import sharpe_decomposition
    import numpy as np
    a = np.array([.01, -.02, .03, .02])
    b = np.array([.02, -.01, .01, .03])
    r = sharpe_decomposition(a, b)
    assert r['mean_effect'] + r['vol_effect'] == pytest.approx(r['difference'])


def test_contribution_reconciliation_rejects_mismatch():
    from state_adapter.sam_mechanism_report import reconcile
    import pandas as pd
    ix = pd.to_datetime(['2025-07-01', '2025-07-02'])
    report = pd.DataFrame({'return':[0., .02], 'cost':[0., .001]}, index=ix)
    rows = pd.DataFrame({'date':[ix[1]], 'instrument':['A'], 'gross_contribution':[.02],
                         'cost_contribution':[.001], 'net_contribution':[.019]})
    assert reconcile(report, rows) < 1e-12
    rows.loc[0, 'net_contribution'] = .018
    with pytest.raises(ValueError, match='对账'):
        reconcile(report, rows)


def test_original_g1_requires_no_adapter_state():
    from state_adapter.sam_mechanism import configure_state
    model = torch.nn.Linear(3, 2)
    state = torch.ones(4, 3)
    before = model.weight.detach().clone()
    assert configure_state(model, state) is False
    assert torch.equal(model.weight, before)
