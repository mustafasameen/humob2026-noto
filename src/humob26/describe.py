"""Data-description statistics, each computed directly from the raw
dataset rather than assumed. Used to describe the dataset in the paper, not
to fit or validate the model -- nothing here feeds back into the pipeline.
"""
from __future__ import annotations

import math

import numpy as np

from .calendar import daterange
from .config import (ANCHOR_AFTER_END, ANCHOR_AFTER_START, ANCHOR_BEFORE_DAYS,
                     ANCHOR_BEFORE_END, ANCHOR_BEFORE_START, N_EVAL_CELLS, NA_DAYS)
from .data import iter_days, to_eval_box
from .foundation import BLOCKS


def describe(data_path):
    """One streaming pass over the raw dataset, computing every statistic
    this command reports. Returns a dict of DataFrames/scalars; the CLI
    decides how to print or save them."""
    n_days = 0
    eval_pair_days = {}          # key -> count of positive days, training only
    anchor_pair_days = {"late_jan": {}, "april": {}}
    diag_sq_mass = np.zeros(N_EVAL_CELLS)
    na_days = set(NA_DAYS)
    late_jan_dates = set(daterange(ANCHOR_BEFORE_START, ANCHOR_BEFORE_END)[-ANCHOR_BEFORE_DAYS:])
    april_dates = set(daterange(ANCHOR_AFTER_START, ANCHOR_AFTER_END))
    block_days = {i: 0 for i in range(len(BLOCKS))}
    anchor_levels = {"late_jan": {"diag": [], "off": []}, "april": {"diag": [], "off": []}}

    for date, arr in iter_days(data_path, keep_oob=True):
        if arr is None:
            continue
        n_days += 1

        box = to_eval_box(date, arr)
        is_diag = box.is_diag
        for b, block in enumerate(BLOCKS):
            if block[0] <= date <= block[-1]:
                block_days[b] += 1
        if date not in na_days:
            for k in box.key:
                eval_pair_days[int(k)] = eval_pair_days.get(int(k), 0) + 1
            if date in late_jan_dates or date in april_dates:
                tag = "late_jan" if date in late_jan_dates else "april"
                for k in box.key:
                    anchor_pair_days[tag][int(k)] = anchor_pair_days[tag].get(int(k), 0) + 1
                anchor_levels[tag]["diag"].append(box.val[is_diag])
                anchor_levels[tag]["off"].append(box.val[~is_diag])
            dk, dv = box.key[is_diag], box.val[is_diag]
            diag_sq_mass[dk // N_EVAL_CELLS] += dv ** 2

    order = np.argsort(diag_sq_mass)[::-1]
    top5_share = float(diag_sq_mass[order[:5]].sum() / diag_sq_mass.sum())

    def pair_summary(pair_days):
        counts = np.array(list(pair_days.values()))
        return dict(n_pairs_ever_positive=int(counts.size),
                   median_positive_days=float(np.median(counts)) if counts.size else float("nan"))

    def anchor_level_summary(tag):
        diag = np.concatenate(anchor_levels[tag]["diag"]) if anchor_levels[tag]["diag"] else np.empty(0)
        off = np.concatenate(anchor_levels[tag]["off"]) if anchor_levels[tag]["off"] else np.empty(0)
        return dict(mean_diag=float(diag.mean()) if diag.size else float("nan"),
                   mean_offdiag=float(off.mean()) if off.size else float("nan"))

    return dict(
        days_per_block={f"block_{i}_{b[0]}_{b[-1]}": n for i, (b, n) in enumerate(zip(BLOCKS, block_days.values()))},
        n_days_total=n_days,
        training_pairs=pair_summary(eval_pair_days),
        late_jan_anchor_pairs=pair_summary(anchor_pair_days["late_jan"]),
        april_anchor_pairs=pair_summary(anchor_pair_days["april"]),
        diagonal_top5_share=top5_share,
        late_jan_anchor_level=anchor_level_summary("late_jan"),
        april_anchor_level=anchor_level_summary("april"),
    )


def format_report(stats):
    """A plain-text rendering of `describe()`'s output for `--out FILE`."""
    lines = []
    lines.append("days observed per block:")
    for k, v in stats["days_per_block"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"days observed total: {stats['n_days_total']}")
    for label, key in (("training", "training_pairs"), ("late-January anchor", "late_jan_anchor_pairs"),
                      ("April anchor", "april_anchor_pairs")):
        s = stats[key]
        lines.append(f"{label} evaluation-region pairs ever positive: {s['n_pairs_ever_positive']} "
                     f"(median positive days: {s['median_positive_days']:.1f})")
    lines.append(f"diagonal squared-mass share in the top 5 cells: {stats['diagonal_top5_share']:.4f}")
    for label, key in (("late-January anchor", "late_jan_anchor_level"), ("April anchor", "april_anchor_level")):
        s = stats[key]
        lines.append(f"{label} mean level: diagonal={s['mean_diag']:.4f} off-diagonal={s['mean_offdiag']:.6f}")
    return "\n".join(lines)
