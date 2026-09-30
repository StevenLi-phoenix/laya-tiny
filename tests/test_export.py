import numpy as np
import pytest
import torch

from laya_tiny.export import compare, export_onnx, ort_session, quantize_int8
from laya_tiny.model import Calibrated, TinyEncoder

pytestmark = pytest.mark.unit
NAMES = ["department", "urgency", "churn_risk"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    m = TinyEncoder(300, dict(zip(NAMES, (4, 3, 2))), layers=2, d_model=32, heads=4, max_len=24).eval()
    w = Calibrated(m, [1.3, 0.8, 2.0]).eval()
    d = tmp_path_factory.mktemp("onnx")
    export_onnx(w, NAMES, d / "m.onnx", 17, 24)
    quantize_int8(d / "m.onnx", d / "m.int8.onnx")
    return w, d


@pytest.mark.parametrize("batch, seq", [(1, 5), (3, 24), (7, 11)])
def test_onnx_matches_torch_with_dynamic_shapes(exported, batch, seq):
    w, d = exported
    g = torch.Generator().manual_seed(batch * 100 + seq)
    ids = torch.randint(4, 300, (batch, seq), generator=g)
    mask = torch.ones_like(ids)
    mask[0, seq // 2:] = 0
    ids[0, seq // 2:] = 0
    with torch.no_grad():
        ref = [t.numpy() for t in w(ids, mask)]
    got = ort_session(d / "m.onnx").run(NAMES, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
    assert compare(ref, got)["max_abs_diff"] < 1e-4
    for p in got:
        assert np.allclose(p.sum(-1), 1.0, atol=1e-5)


def test_int8_is_smaller_and_close(exported):
    _, d = exported
    assert (d / "m.int8.onnx").stat().st_size < 0.5 * (d / "m.onnx").stat().st_size
    ids = np.random.default_rng(0).integers(4, 300, (16, 12)).astype(np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    a = ort_session(d / "m.onnx").run(NAMES, feeds)
    b = ort_session(d / "m.int8.onnx").run(NAMES, feeds)
    assert compare(a, b)["max_abs_diff"] < 0.05
