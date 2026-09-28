"""Adapter contract and temporal truncation checks without model weights."""
import sys
import types
import numpy as np
import pytest
from humob26.forecasters.timesfm3 import TimesFM3Adapter
from humob26.forecast_runner import forecast_window


def test_torch_device_forwarded_and_quantiles(monkeypatch):
    calls = []
    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()
        def predict_batch(self, contexts, horizon, return_quantiles):
            assert return_quantiles
            for c in contexts:
                q = np.tile(np.arange(9), (horizon, 1)) + c[-1]
                yield types.SimpleNamespace(forecast=q[:,4], quantiles=q)
    monkeypatch.setitem(sys.modules, 'timesfm3.torch', types.SimpleNamespace(TimesFM3Forecaster=Model))
    adapter = TimesFM3Adapter.from_pretrained('weights', backend='torch', device='cpu')
    med, lo, hi = adapter.predict([np.array([10, 20]), np.array([30,40])], 3)
    assert calls == [('weights', {'device':'cpu'})]
    np.testing.assert_array_equal(med[:,0], [24,44])
    np.testing.assert_array_equal(lo[:,0], [20,40])
    np.testing.assert_array_equal(hi[:,0], [28,48])
    assert med.shape == (2,3)


@pytest.mark.parametrize('leg,cut', [('fwd',14),('fwd',28),('fwd',56),('bwd',30),('bwd',60),('bwd',120)])
def test_context_cuts_retain_days_nearest_gap_only(leg, cut):
    class Recorder:
        def __init__(self): self.calls = []
        def predict(self, contexts, horizon):
            self.calls.append(np.array(contexts))
            out = np.zeros((len(contexts), horizon))
            return out, out, out
    model = Recorder()
    # Backward context is stored in reverse chronological order.
    ctx = {'keys':np.array([1]), 'is_diag':np.array([True]),
           'fwd':np.arange(92)[None,:], 'bwd':np.arange(214)[::-1][None,:],
           'f_off':np.array([1,2]), 'b_off':np.array([2,1])}
    forecast_window(model, ctx, **{f'context_days_{leg}':cut})
    for i,k in enumerate(('fwd','bwd')):
        expected = ctx[k][:,-cut:] if k == leg else ctx[k]
        np.testing.assert_array_equal(model.calls[i], expected)
