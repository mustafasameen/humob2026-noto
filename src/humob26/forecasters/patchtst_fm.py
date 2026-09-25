"""IBM PatchTST-FM-r2 adapter: `predict` through the package's own
`forward()`, and `infill` for reconstructing an interior masked span in one
pass (both contexts and the gap between them seen together).

`infill` mirrors the package's own single-step reconstruction path with an
arbitrary prediction mask instead of one fixed at the series' end, so it
must be read against that path (`tsfm_public`'s
`forecast_single_step_fast`) whenever the package's internals change;
`verify_infill_matches_forward` is the check that the two still agree
wherever both are defined (the mask at the end).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster, register

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
BATCH_SIZE = 64


def _official(model, contexts, horizon, device):
    """The package's own forward() on a list of 1-D series."""
    import torch

    outs = []
    for i in range(0, len(contexts), BATCH_SIZE):
        x = torch.tensor(np.asarray(contexts[i:i + BATCH_SIZE], dtype=np.float32), device=device).unsqueeze(-1)
        with torch.no_grad():
            o = model(past_values=x, prediction_length=horizon, quantile_levels=list(QUANTILE_LEVELS),
                      return_dict=True)
        outs.append(o.quantile_outputs[..., 0].float().cpu().numpy())
    q = np.concatenate(outs)
    return q[:, 1], q[:, 0], q[:, 2]


def _infill_raw(model, xs, pred_mask, device):
    """Reconstruct the positions where `pred_mask` is True. Returns the
    three quantiles over the WHOLE series, shape (n, 3, L).

    Line for line the package's own masked single-step reconstruction path,
    except that the prediction mask is supplied rather than fixed at the
    series' end.
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from tsfm_public.models.patchtst_fm import modeling_patchtst_fm as mp

    cfg, bb = model.config, model.backbone
    qi = [cfg.quantile_levels.index(q) for q in QUANTILE_LEVELS]
    outs = []
    for i in range(0, len(xs), BATCH_SIZE):
        x = torch.tensor(np.asarray(xs[i:i + BATCH_SIZE], dtype=np.float32), device=device).unsqueeze(-1)
        pm = torch.tensor(np.asarray(pred_mask[i:i + BATCH_SIZE]), device=device, dtype=torch.bool).unsqueeze(-1)
        x = torch.where(pm, torch.full_like(x, float("nan")), x)
        x_mean = x.nanmean(dim=1, keepdim=True)
        x_in = torch.where(torch.isnan(x), x_mean.expand_as(x), x)
        B, L, _ = x_in.shape
        if L > cfg.context_length:
            raise ValueError("series longer than the model's context")
        miss = torch.zeros_like(pm)
        pad = torch.zeros_like(pm)
        left = cfg.context_length - L
        if left:
            inputs = torch.cat((x_mean.repeat((1, left, 1)), x_in), dim=1)
            pm = F.pad(pm, (0, 0, left, 0), mode="constant", value=False)
            pad = F.pad(pad, (0, 0, left, 0), mode="constant", value=True)
            miss = F.pad(miss, (0, 0, left, 0), mode="constant", value=False)
        else:
            inputs = x_in
        r = lambda t: rearrange(t, "B T N -> (B N) T")
        with torch.no_grad(), mp.get_autocast_context(inputs.device):
            mo = bb(inputs=r(inputs), pred_mask=r(pm), miss_mask=r(miss), pad_mask=r(pad), return_loss=False,
                    output_hidden_states=False, override_patch_stride=None, context_length=L, attn_window=None)
            o = mo.quantile_outputs
        if bb.config.patch_stride is not None and bb.config.patch_stride != bb.config.d_patch:
            o = bb._overlap_add(o, original_length=bb.config.context_length, stride=bb.config.patch_stride)
        o = bb.norm_fn.inverse_transform(o.permute(0, 2, 1))
        o = rearrange(o, "(B N) Q T -> B Q T N", B=B)[:, :, left:left + L, 0]
        outs.append(o[:, qi, :].float().cpu().numpy())
    return np.concatenate(outs)


@register("patchtst_fm")
class PatchTSTFMAdapter(Forecaster):
    name = "patchtst_fm"

    def __init__(self, model, device):
        self._model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, weights_dir, device="cpu"):
        from tsfm_public import PatchTSTFMForPrediction
        model = PatchTSTFMForPrediction.from_pretrained(weights_dir).to(device).eval()
        return cls(model, device)

    def predict(self, contexts, horizon: int):
        return _official(self._model, list(contexts), horizon, self.device)

    def infill(self, series, mask):
        rec = _infill_raw(self._model, np.asarray(series), np.asarray(mask), self.device)
        return rec[:, 1, :], rec[:, 0, :], rec[:, 2, :]

    def infill_end_matches_forward(self, contexts, horizon, fill_length):
        """I-1: infilling a mask placed at the series' end reproduces
        `predict()` exactly, on the same contexts. Returns the worst
        relative difference across the median and both quantiles."""
        contexts = list(contexts)
        med, lo, hi = self.predict(contexts, horizon)
        xs = np.concatenate([contexts, np.zeros((len(contexts), fill_length), np.float32)], axis=1)
        pmask = np.zeros(xs.shape, bool)
        pmask[:, contexts[0].shape[0]:] = True
        rec_med, rec_lo, rec_hi = self.infill(xs, pmask)
        c = contexts[0].shape[0]
        rec_med, rec_lo, rec_hi = rec_med[:, c:c + horizon], rec_lo[:, c:c + horizon], rec_hi[:, c:c + horizon]
        return max(
            np.abs(rec_med - med).max() / (1 + np.abs(med).max()),
            np.abs(rec_lo - lo).max() / (1 + np.abs(lo).max()),
            np.abs(rec_hi - hi).max() / (1 + np.abs(hi).max()),
        )
