"""Amazon Chronos-2 adapter: each series forecast independently
(`cross_learning=False`), one call producing the 0.1/0.5/0.9 quantiles.

Chronos-2 does not itself enforce that its quantile outputs are ordered at
every step (`sort_quantiles` in this package's CLI applies the monotone
rearrangement afterwards, if asked for).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster, register

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
BATCH_SIZE = 256


@register("chronos2")
class Chronos2Adapter(Forecaster):
    name = "chronos2"

    def __init__(self, pipeline):
        self._pipeline = pipeline

    @classmethod
    def from_pretrained(cls, weights_dir, device="cpu"):
        import torch
        from chronos import Chronos2Pipeline
        # Fixed before the pipeline loads, matching how this adapter's
        # numerics were verified: Chronos-2's quantile estimation samples
        # internally, so the seed must be set before any call for the
        # result to be reproducible run to run.
        torch.manual_seed(0)
        pipeline = Chronos2Pipeline.from_pretrained(weights_dir, device_map=device, torch_dtype=torch.float32)
        return cls(pipeline)

    def predict(self, contexts, horizon: int):
        quantiles, _ = self._pipeline.predict_quantiles(
            [np.asarray(r, dtype=np.float32) for r in contexts], prediction_length=horizon,
            quantile_levels=list(QUANTILE_LEVELS), batch_size=BATCH_SIZE, cross_learning=False)
        q = np.stack([x[0].float().numpy() for x in quantiles])       # (n, horizon, 3)
        return q[:, :, 1], q[:, :, 0], q[:, :, 2]


def sort_quantiles(median, q10, q90):
    """Monotone rearrangement: sort (q10, median, q90) at every point so a
    model that does not itself enforce quantile ordering still returns an
    ordered triple. A no-op wherever the triple was already ordered."""
    q = np.sort(np.stack([q10, median, q90]), axis=0)
    return q[1], q[0], q[2]
