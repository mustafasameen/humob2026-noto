"""Forecaster-slot comparison ("compare-slot"): swap only the model behind
the diagonal forecast blend of the `final` construction and see whether the
pooled rolling-window score still improves, holding the base construction,
the off-diagonal, and the windows fixed. A second, separate section scores
the three earliest-January windows as a "shock-origin" set (their
before-anchor sits closest to the one known discontinuity in the record),
with a forward/backward leg split that isolates which direction of a joint
forecast is doing the work.

Every arm here means the SAME final construction with only its diagonal
forecast source changed; the off-diagonal is identical across arms by
construction; a mismatch there is a fatal error, not a soft warning, since
if it fires something upstream of the slot itself has drifted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import combine, metric, windows as windows_mod

QUANTILE_MARGIN = 1e-3          # tolerance for a quantile-ordering check
CROSSING_LEVEL_FLOOR = 1.0      # a crossing on a smaller series is ignored
CROSSING_MAX_EXCESS = 0.05      # ... unless it exceeds this many units anywhere


def load_forecast(arm_dir, window_name):
    """A forecast file's fwd/bwd(/quantile) arrays as float64, keyed the
    way every other array in this package is."""
    f = np.load(f"{arm_dir}/fc_{window_name}.npz")
    return {k: np.asarray(f[k], dtype=np.float64) for k in f.files if k != "mv_rows"}


def mean_forecast(arm_dirs, window_name):
    """Elementwise mean of several arms' fwd/bwd arrays (their `--mean`
    ensemble). Quantiles are not averaged (the ensemble arm carries only a
    median)."""
    fs = [load_forecast(d, window_name) for d in arm_dirs]
    return {k: np.mean([f[k] for f in fs], axis=0) for k in ("fwd", "bwd")}


def slot_median(ctx, forecast, shift=0):
    """The inverse-distance-blended median forecast for every target day,
    with an optional +/-1-day index shift (the offset check probes whether
    the declared day-to-column indexing, not a shifted one, is what scores
    best)."""
    fo, bo = ctx["f_off"], ctx["b_off"]
    wb = (fo / (fo + bo))[:, None]

    def pick(a, off):
        idx = np.clip(off - 1 + shift, 0, a.shape[1] - 1)
        return np.stack([a[:, idx[i]] for i in range(len(off))])

    med = np.clip((1 - wb) * pick(forecast["fwd"], fo) + wb * pick(forecast["bwd"], bo), 0, None)
    return ctx["keys"], [str(t) for t in ctx["T"]], med


def check_forecast_shapes(reference_fc, arm_fc, ctx, lenient=False):
    """Gate: an arm's forecast file matches the reference's shape, is
    finite on the diagonal, and its quantiles do not cross by more than a
    small, size-scaled tolerance. `lenient` downgrades a crossing to a note
    instead of a failure (used for a raw, pre-sorted quantile sensitivity
    arm)."""
    diag = ctx["is_diag"]
    notes = []
    for leg in ("fwd", "bwd"):
        if arm_fc[leg].shape != reference_fc[leg].shape:
            return False, f"{leg} shape {arm_fc[leg].shape} vs {reference_fc[leg].shape}"
        if not np.isfinite(arm_fc[leg][diag]).all():
            return False, f"{leg} non-finite"
        if f"{leg}_q10" in arm_fc:
            lo, hi, med = arm_fc[f"{leg}_q10"][diag], arm_fc[f"{leg}_q90"][diag], arm_fc[leg][diag]
            excess = np.maximum(lo - med, med - hi) - QUANTILE_MARGIN * (1 + np.abs(med))
            bad = (excess > 0).any(axis=1)
            level = ctx[leg][diag].mean(axis=1)
            if not lenient and ((bad & (level >= CROSSING_LEVEL_FLOOR)).any() or excess.max() > CROSSING_MAX_EXCESS):
                return False, f"{leg} quantiles cross on a non-trivial series (max excess {excess.max():.4f})"
            if bad.any():
                notes.append(f"{leg}: crossing on {int(bad.sum())} near-zero series (max excess {excess.max():.4f})")
    return True, "; ".join(notes)


def final_with_slot(ctx, forecast, target_dates, base_diag, base_offdiag, shift=0):
    """The `final` construction with the diagonal forecast taken from
    `forecast` instead of the default one."""
    keys, days, med = slot_median(ctx, forecast, shift)
    if days != list(target_dates):
        raise ValueError("forecast days differ from the window's target dates")
    flows = {d: metric.DayFlows(d, keys[med[i] > 0], med[i][med[i] > 0]) for i, d in enumerate(days)}
    return combine.weighted_sum(combine.diagonal_part(flows, True), base_diag, base_offdiag,
                                weights=[0.5, 0.5, 1.0])


def bonferroni_percentiles(n_arms, family_alpha=0.05):
    """Two-sided percentile bounds for a pooled CI, Bonferroni-corrected
    across `n_arms` simultaneous comparisons."""
    alpha = family_alpha / max(n_arms, 1)
    return 100 * alpha / 2, 100 * (1 - alpha / 2)


def cluster_percentile(records, reps, pct, seed=0):
    """`bootstrap.date_cluster_bootstrap`'s resampling, returning raw
    percentiles of the pooled statistic rather than a symmetric CI (needed
    here because the Bonferroni bounds are not symmetric around 50%)."""
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(records.date.unique()))
    grouped = records.groupby("date")
    sum_by_date, n_by_date = grouped["diff"].sum(), grouped["diff"].size()
    boot = [sum_by_date.loc[s].sum() / n_by_date.loc[s].sum()
            for s in (rng.choice(dates, dates.size, replace=True) for _ in range(reps))]
    return np.percentile(boot, pct)


def build_base(days, na, all_dates, window):
    """The eval-box `final`-without-a-diagonal-forecast base: the
    empirical-Bayes interpolation blended with the RTS smoother, split into
    its diagonal and off-diagonal parts (what every arm's diagonal slot is
    added to)."""
    keys, train, calib = combine.eval_box_context(days, window, all_dates)
    base = combine.build_anchor_rts_eb(days, window, na, keys, train, calib)
    return combine.diagonal_part(base, True), combine.diagonal_part(base, False), keys, train, calib


NO_HARM_WINDOWS = ("may_jun", "late_jan", "april")
OFFSET_WINDOWS = ("may_jun", "april")
FLOOR_PCT = 0.05


def _forecast_getter(reference_dir, arm_dirs, means):
    """A `name -> {window -> forecast dict}` resolver: plain arms load their
    own directory, a `--mean` arm averages its ingredients' arrays (which
    may themselves be "reference" or any other named arm)."""
    def dir_of(name):
        if name == "reference":
            return reference_dir
        if name in arm_dirs:
            return arm_dirs[name]
        raise KeyError(f"unknown arm or mean ingredient {name!r}")

    def get(name, window_name):
        if name in means:
            return mean_forecast([dir_of(n) for n in means[name]], window_name)
        return load_forecast(dir_of(name), window_name)
    return get


def run(days, na, all_dates, contexts_dir, reference_dir, arm_dirs, means=None,
       window_names=None, reps=4000, lenient_arms=()):
    """Score every arm (plain or `--mean`) against `reference` on every
    window, with the shape/finite/quantile-order gate, the off-diagonal
    identity check across arms, and the +/-1-day offset check on the two
    designated windows. Returns (window_scores, daily_rows, gate_rows)."""
    means = means or {}
    arm_names = list(arm_dirs) + list(means)
    window_names = window_names or list(windows_mod.VALIDATION_WINDOW_NAMES)
    get_forecast = _forecast_getter(reference_dir, arm_dirs, means)

    per_win, rows, gates = [], [], []
    for name in window_names:
        window = windows_mod.get_window(name, days, na)
        obs = {d: days[d] for d in window.target}
        ctx = np.load(f"{contexts_dir}/ctx_{name}.npz", allow_pickle=True)
        base_diag, base_offdiag, keys, train, calib = build_base(days, na, all_dates, window)

        # gate 1 (internal consistency): the reference arm's slot-based
        # reconstruction must reproduce this package's own `final` arm,
        # built independently through combine.py, using the SAME forecast.
        from . import foundation
        reference_fc = get_forecast("reference", name)
        forecast_diag = foundation.read_forecast(f"{contexts_dir}/ctx_{name}.npz",
                                                 f"{reference_dir}/fc_{name}.npz")
        expected = combine.build_final(days, window, na, keys, train, calib, forecast_diag)
        got = final_with_slot(ctx, reference_fc, window.target, base_diag, base_offdiag)
        g1 = abs(metric.daily_combined(metric.score(obs, got)[1]).mean()
                - metric.daily_combined(metric.score(obs, expected)[1]).mean())
        gates.append(dict(gate="1_reference_matches_final", arm="reference", window=name,
                          value=g1, ok=bool(g1 <= 1e-12), note=""))
        if g1 > 1e-12:
            raise SystemExit(f"GATE 1 FAIL on {name}: reference slot {g1!r} vs combine.build_final")

        combined, diag_c, off_c = {}, {}, {}
        for arm in ["reference"] + arm_names:
            fc = reference_fc if arm == "reference" else get_forecast(arm, name)
            ok, why = check_forecast_shapes(reference_fc, fc, ctx, lenient=arm in lenient_arms)
            gates.append(dict(gate="3_files", arm=arm, window=name, value=np.nan, ok=ok, note=why))
            if not ok:
                raise SystemExit(f"GATE 3 FAIL {arm} {name}: {why}")
            pred = final_with_slot(ctx, fc, window.target, base_diag, base_offdiag)
            per_day = metric.score(obs, pred)[1].set_index("date")
            combined[arm] = metric.daily_combined(per_day)
            diag_c[arm], off_c[arm] = metric.nrmse_components(per_day[["sse_diag", "sse_offdiag"]], window.target)
            if arm != "reference" and name in OFFSET_WINDOWS:
                for shift in (-1, 1):
                    shifted = final_with_slot(ctx, fc, window.target, base_diag, base_offdiag, shift=shift)
                    cs = metric.daily_combined(metric.score(obs, shifted)[1]).mean()
                    gates.append(dict(gate="4_offset", arm=arm, window=name, value=cs - combined[arm].mean(),
                                      ok=bool(combined[arm].mean() < cs), note=f"shift {shift:+d}"))

        spread = max(off_c.values()) - min(off_c.values())
        if spread > 1e-12:
            raise SystemExit(f"{name}: off-diagonal differs across arms by {spread:.2e}")

        for arm in arm_names:
            per_win.append(dict(window=name, geometry=window.geometry, n_days=len(window.target), arm=arm,
                                reference=combined["reference"].mean(), combined=combined[arm].mean(),
                                pct_better=100 * (1 - combined[arm].mean() / combined["reference"].mean()),
                                diag_reference=diag_c["reference"], diag=diag_c[arm], off=off_c[arm]))
            for d in window.target:
                rows.append(dict(arm=arm, window=name, date=d, roll=name.startswith("roll_"),
                                 diff=combined[arm][d] - combined["reference"][d], reference=combined["reference"][d]))
    return pd.DataFrame(per_win), pd.DataFrame(rows), pd.DataFrame(gates)


def summarize(window_scores, daily_rows, gate_rows, arm_names, reps=4000):
    """The per-arm verdict table: pooled rolling-window CI at the
    Bonferroni level for `len(arm_names)` simultaneous comparisons, no
    significant loss on the no-harm windows, a positive point estimate on
    `may_jun`, and the offset gate."""
    pooled_pct = bonferroni_percentiles(len(arm_names))
    off_fail = gate_rows[(gate_rows.gate == "4_offset") & ~gate_rows.ok.astype(bool)]
    summary = []
    for arm in arm_names:
        rolled = daily_rows[(daily_rows.arm == arm) & daily_rows.roll]
        lo = cluster_percentile(rolled, reps, pooled_pct[0])
        hi = cluster_percentile(rolled, reps, pooled_pct[1])
        est = rolled["diff"].mean()
        pct = 100 * -est / rolled.reference.mean()
        harm = {}
        for w in NO_HARM_WINDOWS:
            sub = daily_rows[(daily_rows.arm == arm) & (daily_rows.window == w)].set_index("date")["diff"]
            rng = np.random.default_rng(0)
            bs = [sub.loc[rng.choice(sub.index, sub.size, replace=True)].mean() for _ in range(reps)]
            blo, bhi = np.percentile(bs, [2.5, 97.5])
            harm[w] = bool(blo > 0)
        may_jun_pct = window_scores[(window_scores.arm == arm)
                                    & (window_scores.window == "may_jun")].pct_better.iloc[0]
        offset_ok = arm not in set(off_fail.arm)
        passed = bool(hi < 0 and pct > FLOOR_PCT and not any(harm.values()) and may_jun_pct > 0 and offset_ok)
        summary.append(dict(arm=arm, pooled_diff=est, ci_lo=lo, ci_hi=hi, pooled_pct=pct,
                            may_jun_pct=may_jun_pct, **{f"harm_{w}": harm[w] for w in NO_HARM_WINDOWS},
                            offset_gate=offset_ok, verdict="PASS" if passed else "FAIL"))
    return pd.DataFrame(summary)


def run_shock_origin(days, na, all_dates, contexts_dir, reference_dir, joint_dir, reps=4000):
    """Reference-vs-joint-mode check restricted to the three windows whose
    before-anchor sits closest to the record's one known discontinuity
    (`windows.SHOCK_ORIGIN_WINDOW_NAMES`), plus a forward/backward leg
    decomposition: does the joint gain come from its forward leg, its
    backward leg, or both?
    """
    from . import windows as wmod
    per, rows, legs = [], [], []
    for name in wmod.SHOCK_ORIGIN_WINDOW_NAMES:
        window = wmod.get_window(name, days, na)
        obs = {d: days[d] for d in window.target}
        ctx = np.load(f"{contexts_dir}/ctx_{name}.npz", allow_pickle=True)
        base_diag, base_offdiag, *_ = build_base(days, na, all_dates, window)
        reference_fc = load_forecast(reference_dir, name)
        joint_fc = load_forecast(joint_dir, name)
        fwd_only = dict(joint_fc, bwd=reference_fc["bwd"])
        bwd_only = dict(reference_fc, bwd=joint_fc["bwd"])
        fwd_only["fwd"], bwd_only["fwd"] = joint_fc["fwd"], reference_fc["fwd"]

        combined = {}
        for tag, fc in (("reference", reference_fc), ("joint", joint_fc),
                        ("joint_fwd_only", fwd_only), ("joint_bwd_only", bwd_only)):
            pred = final_with_slot(ctx, fc, window.target, base_diag, base_offdiag)
            combined[tag] = metric.daily_combined(metric.score(obs, pred)[1])

        def pct(tag):
            return 100 * (1 - combined[tag].mean() / combined["reference"].mean())

        fo, bo = ctx["f_off"], ctx["b_off"]
        is_new = name != "late_jan"
        per.append(dict(window=name, n_days=len(window.target), first=window.target[0], last=window.target[-1],
                        reference=combined["reference"].mean(), joint=combined["joint"].mean(),
                        joint_better_pct=pct("joint"), new=is_new))
        legs.append(dict(window=name, n_days=len(window.target), mean_backward_weight=float((fo / (fo + bo)).mean()),
                         fwd_ctx_days=int(ctx["fwd"].shape[1]), bwd_ctx_days=int(ctx["bwd"].shape[1]),
                         joint_both_pct=pct("joint"), joint_fwd_only_pct=pct("joint_fwd_only"),
                         joint_bwd_only_pct=pct("joint_bwd_only")))
        for d in window.target:
            rows.append(dict(window=name, date=d, diff=combined["joint"][d] - combined["reference"][d],
                             reference=combined["reference"][d], new=is_new))

    per_df, rows_df, legs_df = pd.DataFrame(per), pd.DataFrame(rows), pd.DataFrame(legs)
    new_rows = rows_df[rows_df.new]
    est = new_rows["diff"].mean()
    pct_new = 100 * -est / new_rows.reference.mean()
    rng = np.random.default_rng(0)
    v = new_rows["diff"].values
    boot = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(reps)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    keep = bool(est <= 0)
    all_rows_mean = rows_df["diff"].mean()
    pct_all = 100 * -all_rows_mean / rows_df.reference.mean()
    summary = pd.DataFrame([dict(test="shock_origin_new_windows", n_days=len(new_rows), est=est, ci_lo=lo, ci_hi=hi,
                                 pct=pct_new, pct_with_late_jan=pct_all, verdict="KEEP" if keep else "REVERT")])
    return per_df, summary, legs_df
