import numpy as np
import pytest

from laya_tiny.calibrate import fit_temperature, softmax
from laya_tiny.metrics import accuracy, agreement, ece, macro_f1, tv_distance

pytestmark = pytest.mark.unit


def test_basic_metrics():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1, 1])
    assert accuracy(p, y) == 0.75
    # class 0: tp1 fp1 fn0 -> 2/3 ; class 1: tp2 fp0 fn1 -> 4/5
    assert macro_f1(p, y, 2) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert agreement(p, p) == 1.0 and agreement(p, p[:, ::-1]) == 0.0
    assert tv_distance(p, p) == 0.0
    assert tv_distance(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == 1.0


def test_macro_f1_skips_absent_classes():
    p = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    assert macro_f1(p, np.array([0, 1]), 3) == 1.0


def test_ece_perfect_and_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    y = (rng.uniform(size=conf.size) < conf).astype(int)       # label 1 with prob = confidence
    p = np.stack([1 - conf, conf], 1)
    assert ece(p, y) < 0.02
    assert ece(np.array([[0.0, 1.0]] * 4), np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_fit_temperature_recovers_known_scale():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(4000, 4)) * 2
    target = softmax(true)
    t = fit_temperature(true * 3.0, target, 0.5, 5.0, 181)
    assert t == pytest.approx(3.0, rel=0.03)
