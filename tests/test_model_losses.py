import math

import pytest
import torch

from laya_tiny.config import load_config
from laya_tiny.losses import IGNORE, hard_loss, kd_loss, total_loss
from laya_tiny.model import Calibrated, TinyEncoder, build_model, count_params

pytestmark = pytest.mark.unit
HEADS = {"department": 4, "urgency": 3, "churn_risk": 2}


def test_default_model_is_in_the_1_to_5M_budget(repo_root):
    cfg = load_config(repo_root / "configs")
    n = count_params(build_model(cfg["model"], cfg["model"]["tokenizer"]["vocab_size"], HEADS))
    assert 1e6 <= n <= 5e6


def test_forward_shapes_and_padding_invariance():
    torch.manual_seed(0)
    m = TinyEncoder(100, HEADS, layers=2, d_model=32, heads=4, max_len=16).eval()
    ids = torch.tensor([[2, 10, 11, 12]])
    out = m(ids, torch.ones_like(ids))
    assert [o.shape for o in out] == [(1, 4), (1, 3), (1, 2)]
    padded = torch.tensor([[2, 10, 11, 12, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    for a, b in zip(out, m(padded, mask)):
        assert torch.allclose(a, b, atol=1e-5)


def test_cls_pooling_and_calibrated_wrapper():
    m = TinyEncoder(50, HEADS, layers=1, d_model=16, heads=2, max_len=8, pooling="cls").eval()
    ids = torch.tensor([[2, 5, 6], [2, 7, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    probs = Calibrated(m, [1.0, 2.0, 0.5])(ids, mask)
    for p in probs:
        assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6)
    sharp = torch.softmax(m(ids, mask)[2] * 2.0, -1)
    assert torch.allclose(probs[2], sharp, atol=1e-6)


def test_kd_loss_is_zero_at_teacher_and_positive_elsewhere():
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    for t in (1.0, 2.0, 4.0):
        assert kd_loss(torch.log(p), p, t).item() == pytest.approx(0.0, abs=1e-6)
        assert kd_loss(torch.zeros(2, 3), p, t).item() > 0


def test_kd_loss_t_squared_scaling_matches_manual():
    p = torch.tensor([[0.6, 0.4]])
    z = torch.tensor([[0.0, 1.0]])
    t = 2.0
    tgt = torch.softmax(torch.log(p) / t, -1)
    s = torch.softmax(z / t, -1)
    manual = (tgt * (tgt.log() - s.log())).sum() * t * t
    assert kd_loss(z, p, t).item() == pytest.approx(manual.item(), rel=1e-5)


def test_hard_loss_ignores_missing_labels_but_keeps_graph():
    z = torch.randn(3, 4, requires_grad=True)
    loss = hard_loss(z, torch.full((3,), IGNORE))
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None
    assert hard_loss(z, torch.tensor([0, IGNORE, 2])).item() == pytest.approx(
        torch.nn.functional.cross_entropy(z[[0, 2]], torch.tensor([0, 2])).item(), rel=1e-5)


def test_total_loss_weights_heads():
    z = {"a": torch.zeros(2, 2), "b": torch.zeros(2, 3)}
    t = {"a": torch.tensor([[0.9, 0.1]] * 2), "b": torch.full((2, 3), 1 / 3)}
    hard = {"a": torch.tensor([0, IGNORE]), "b": torch.full((2,), IGNORE)}
    l1, parts = total_loss(z, t, hard, {"a": 1.0, "b": 1.0}, 1.0, 0.0)
    l2, _ = total_loss(z, t, hard, {"a": 2.0, "b": 1.0}, 1.0, 0.0)
    assert l2.item() == pytest.approx(2 * parts["kd/a"] + parts["kd/b"], rel=1e-5)
    assert parts["kd/b"] == pytest.approx(0.0, abs=1e-6) and l1.item() > 0
    l3, parts3 = total_loss(z, t, hard, {}, 1.0, 0.5)
    assert l3.item() == pytest.approx(l1.item() + 0.5 * math.log(2), rel=1e-4) and "ce/a" in parts3
