"""TimesFM-3 adapter: one cell at a time (`predict`) or several cells in a
single multivariate call (`predict_joint`, for the checkpoints that support
it). The framework is imported lazily inside each method, per backend, so
this module can be listed in the registry without either being installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Forecaster, register

QUANTILE_LO_IDX, QUANTILE_HI_IDX = 0, 8   # the 0.1 and 0.9 quantile columns


def _predict_batch_or_loop(model, contexts, horizon):
    """Use the batched call if the backend offers one; fall back to one
    call per series otherwise. Returns (median, q10, q90)."""
    try:
        outs = list(model.predict_batch(contexts, horizon=horizon, return_quantiles=True))
    except TypeError:
        outs = [model.predict(c, horizon=horizon, return_quantiles=True) for c in contexts]
    q = np.stack([o.quantiles for o in outs])
    return np.stack([o.forecast for o in outs]), q[:, :, QUANTILE_LO_IDX], q[:, :, QUANTILE_HI_IDX]


@register("timesfm3")
class TimesFM3Adapter(Forecaster):
    name = "timesfm3"

    def __init__(self, model, backend):
        self._model = model
        self.backend = backend

    @classmethod
    def from_pretrained(cls, weights_dir, backend="mlx", device=None):
        if backend == "mlx":
            from timesfm3.mlx import TimesFM3Forecaster
            return cls(TimesFM3Forecaster.from_pretrained(weights_dir), backend)
        if backend == "torch":
            from timesfm3.torch import TimesFM3Forecaster
            return cls(TimesFM3Forecaster.from_pretrained(str(weights_dir), device=device), backend)
        raise ValueError(f"unknown backend {backend!r}")

    def predict(self, contexts, horizon: int):
        return _predict_batch_or_loop(self._model, list(contexts), horizon)

    def predict_joint(self, matrix, horizon: int):
        out = next(self._model.predict_batch([np.asarray(matrix, dtype=np.float32)],
                                              horizon=horizon, return_quantiles=True))
        return out.forecast, out.quantiles[:, :, QUANTILE_LO_IDX], out.quantiles[:, :, QUANTILE_HI_IDX]

    @property
    def max_variates(self):
        """The checkpoint's joint-mode width, read from its own config."""
        return getattr(self._model, "max_variates", None)


def checkpoint_max_variates(weights_dir):
    """The `max_variates` a TimesFM-3 checkpoint's config.json declares,
    without loading the model -- used to size a joint-mode call."""
    cfg = json.loads((Path(weights_dir) / "config.json").read_text())
    return cfg["transformer_config"]["transformer"]["max_variates"]
