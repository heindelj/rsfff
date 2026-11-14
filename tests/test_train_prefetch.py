"""Batch transfer of train_ionct: pack_to_device (one copy per dtype, views of the device buffer) and
prefetch_batches (background collate + side-stream copy) give the same batches as collate + to_device."""
import pytest
import torch

import rsfff.train.train_ionct as T

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fake_batch(k):
    g = torch.Generator().manual_seed(k)
    return {"B": 3 + k, "nmax": 5, "L": torch.randn(3, 5, 8, generator=g, dtype=torch.float64),
            "mask": torch.rand(3, 5, generator=g) < 0.5, "species": torch.randint(0, 4, (11,), generator=g),
            "ch_index": torch.randint(0, 11, (2, 7), generator=g), "empty": torch.zeros(0, 8, dtype=torch.float64),
            "A_pp": None, "labels": {"pol": torch.randn(3, generator=g, dtype=torch.float64),
                                     "ct_mask": torch.tensor([True, False, True])},
            "env": {"x": torch.randn(11, 3, generator=g, dtype=torch.float32)}, "lst": [torch.arange(4), 7]}


def _same(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


def test_pack_to_device_is_a_no_op_on_cpu():
    bt = _fake_batch(0)
    out, bufs = T.pack_to_device(bt, torch.device("cpu"))
    assert out is bt and bufs == []


def test_prefetch_on_cpu_is_collate(monkeypatch):
    monkeypatch.setattr(T, "collate", lambda chunk: _fake_batch(chunk[0]))
    got = list(T.prefetch_batches([[0], [1], [2]], torch.device("cpu")))
    for k, bt in enumerate(got):
        _same(bt, _fake_batch(k))


@cuda
def test_pack_to_device_matches_to_device():
    bt = _fake_batch(1)
    out, bufs = T.pack_to_device(bt, torch.device("cuda"))
    torch.cuda.synchronize()
    assert len(bufs) == 4                                       # float64, bool, int64, float32
    _same(out, bt)
    assert out["L"].device.type == "cuda" and out["A_pp"] is None and out["lst"][1] == 7


@cuda
def test_prefetch_batches_on_cuda(monkeypatch):
    monkeypatch.setattr(T, "collate", lambda chunk: _fake_batch(chunk[0]))
    seen = []
    for k, bt in enumerate(T.prefetch_batches([[j] for j in range(6)], torch.device("cuda"))):
        x = bt["L"] * 2                                         # used on the consumer stream
        seen.append(float(x.sum()))
        _same(bt, _fake_batch(k))
    # GPU and CPU reductions sum in different orders: compare to rounding (the batch contents are checked exactly above)
    assert seen == pytest.approx([float((_fake_batch(k)["L"] * 2).sum()) for k in range(6)], rel=1e-12, abs=1e-12)


@cuda
def test_prefetch_stops_cleanly_on_break(monkeypatch):
    monkeypatch.setattr(T, "collate", lambda chunk: _fake_batch(chunk[0]))
    gen = T.prefetch_batches([[j] for j in range(20)], torch.device("cuda"))
    next(gen)
    gen.close()


def test_prefetch_reraises_collate_errors(monkeypatch):
    def bad(chunk):
        raise RuntimeError("boom")
    monkeypatch.setattr(T, "collate", bad)
    with pytest.raises(RuntimeError, match="boom"):
        list(T.prefetch_batches([[0]], torch.device("cuda" if torch.cuda.is_available() else "cpu")))
