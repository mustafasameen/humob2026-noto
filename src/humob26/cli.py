"""Command-line interface: `python -m humob26 <command> ...`.

  prepare       parse the raw dataset once into a cache directory.
  contexts      export bidirectional forecasting contexts for every window.
  forecast      run a registered forecaster adapter over exported contexts.
  evaluate      score one or more arms on the validation windows, with
                per-day and per-window tables and bootstrap comparisons.
  compare-slot  the forecaster-slot bake-off: swap only the diagonal
                forecast source and see whether the pooled score still
                improves, plus a separate shock-origin check.
  sweep         re-score a small grid of `--set key=value` overrides
                against the submitted configuration.
  describe      data-description statistics computed from the raw dataset.
  submit        build a submission file for the real prediction gap.

None of `contexts`/`evaluate`/`compare-slot`/`sweep`/`submit` touch the raw
dataset path after `prepare`; they read only from the cache directory it
wrote. Every command reads and writes files under public names only
(`ctx_<window>.npz`, `fc_<window>.npz`, ...): there is no mechanism in this
package for pointing it at a differently named file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import combine, config, data, foundation, metric, submission, windows
from .bootstrap import date_cluster_bootstrap, paired_bootstrap, paired_bootstrap_components

# `pandas`, `bakeoff` and `describe` are imported lazily inside the commands
# that need them (evaluate, compare-slot, sweep, describe): `forecast` runs
# inside a model's own environment, which may have only numpy installed
# alongside that model's framework, and must not need pandas to work.


class Pipeline:
    """Builds any named arm on any window, caching the (keys, train, calib)
    modelling context per window so it is computed once even when several
    arms share it. `settings` (a config.Settings) overrides the default
    model constants for a sensitivity run; the default reproduces them."""

    def __init__(self, days, na, all_dates, full_days=None, all_full_dates=None,
                forecasts_dir=None, settings=None):
        self.days, self.na, self.all_dates = days, na, all_dates
        self.full_days, self.all_full_dates = full_days, all_full_dates
        self.na_full = set(config.NA_DAYS)
        self.forecasts_dir = forecasts_dir
        self.settings = settings or config.Settings()
        self._eval_ctx_cache = {}
        self._full_ctx_cache = {}

    def eval_ctx(self, window):
        if window.name not in self._eval_ctx_cache:
            self._eval_ctx_cache[window.name] = combine.eval_box_context(self.days, window, self.all_dates)
        return self._eval_ctx_cache[window.name]

    def full_ctx(self, window):
        if window.name not in self._full_ctx_cache:
            self._full_ctx_cache[window.name] = combine.full_grid_context(
                self.full_days, window, self.all_full_dates)
        return self._full_ctx_cache[window.name]

    def forecast_diag(self, window):
        ctx_path = Path(self.forecasts_dir) / f"ctx_{window.name}.npz"
        fc_path = Path(self.forecasts_dir) / f"fc_{window.name}.npz"
        if not ctx_path.exists() or not fc_path.exists():
            raise FileNotFoundError(f"window {window.name!r}: expected forecast files {ctx_path} and {fc_path}")
        return foundation.read_forecast(ctx_path, fc_path)

    def build(self, arm, window):
        s = self.settings
        if arm == "april_mean":
            return combine.build_april_mean(self.days, window, self.na)
        if arm == "novdec_mean":
            return combine.build_novdec_mean(self.days, window, self.na)
        if arm == "anchor_rts_fullgrid":
            keys, train, calib = self.full_ctx(window)
            return combine.build_anchor_rts_fullgrid(self.full_days, window, self.na_full, keys, train, calib, s)
        keys, train, calib = self.eval_ctx(window)
        if arm == "anchor":
            return combine.build_anchor(self.days, window, self.na, keys, calib, s)
        if arm == "anchor_eb":
            return combine.build_anchor_eb(self.days, window, self.na, keys, calib, s)
        if arm == "anchor_rts":
            return combine.build_anchor_rts(self.days, window, self.na, keys, train, calib, s)
        if arm == "anchor_rts_eb":
            return combine.build_anchor_rts_eb(self.days, window, self.na, keys, train, calib, s)
        if arm == "anchor_rts_fm":
            return combine.build_anchor_rts_fm(self.days, window, self.na, keys, train, calib,
                                               self.forecast_diag(window), s)
        if arm == "final":
            return combine.build_final(self.days, window, self.na, keys, train, calib,
                                       self.forecast_diag(window), s)
        raise ValueError(f"unknown arm {arm!r}; choose from {combine.ARM_NAMES}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_prepare(args):
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    days, na = data.load_eval_box(args.data, cache=cache_dir / "eval_box.npz")
    full = data.load_full_grid(args.data, cache=cache_dir / "full_grid.npz")
    print(f"evaluation box : {len(days)} days ({len(na)} not-a-number) -> {cache_dir / 'eval_box.npz'}")
    print(f"full grid      : {len(full)} days -> {cache_dir / 'full_grid.npz'}")


def cmd_contexts(args):
    cache_dir = Path(args.cache)
    days, na = data.load_eval_box(None, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    names = args.windows or list(windows.ALL_WINDOW_NAMES) + list(windows.SHIFT_WINDOW_NAMES)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        w = windows.get_window(name, days, na)
        ctx = foundation.build_context(days, na, all_dates, w)
        foundation.save_context(out_dir / f"ctx_{name}.npz", ctx)
        print(f"{name:<14} target={w.target[0]}..{w.target[-1]} ({len(w.target)} d)  series={ctx['keys'].size}")


def cmd_evaluate(args):
    import pandas as pd
    cache_dir = Path(args.cache)
    days, na = data.load_eval_box(None, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    names = args.windows or list(windows.VALIDATION_WINDOW_NAMES)
    settings = _settings_from_args(args)
    compare_pairs = [tuple(p.split(":", 1)) for p in (args.compare or DEFAULT_COMPARE_PAIRS)]
    arms = set(args.arms or combine.ARM_NAMES)
    for a, b in compare_pairs:
        arms |= {a, b}
    arms = [a for a in combine.ARM_NAMES if a in arms]        # stable, canonical order

    full_days = all_full_dates = None
    if "anchor_rts_fullgrid" in arms:
        full_days = data.load_full_grid(None, cache=cache_dir / "full_grid.npz")
        na_full = set(config.NA_DAYS)
        all_full_dates = sorted(d for d in full_days if d not in na_full)

    pipe = Pipeline(days, na, all_dates, full_days, all_full_dates, args.forecasts, settings)

    daily_rows, window_rows = [], []
    per_day_by = {}
    for name in names:
        w = windows.get_window(name, days, na, settings.anchor_before_days)
        obs = {d: days[d] for d in w.target}
        combined_by_arm = {}
        for arm in arms:
            pred = pipe.build(arm, w)
            summary, per_day = metric.score(obs, pred)
            per_day_idx = per_day.set_index("date")
            per_day_by[(name, arm)] = per_day_idx
            daily = metric.daily_combined(per_day_idx)
            for d in w.target:
                daily_rows.append(dict(
                    window=name, date=d, arm=arm, combined=daily[d],
                    nrmse_diag=per_day_idx.loc[d, "rmse_diag_dense"] / config.NORM_DIAG,
                    nrmse_offdiag=per_day_idx.loc[d, "rmse_offdiag_dense"] / config.NORM_OFFDIAG))
            window_rows.append(dict(window=name, arm=arm, n_days=summary["n_days"],
                                    combined=summary["combined_dense"],
                                    nrmse_diag=summary["nrmse_diag_dense"],
                                    nrmse_offdiag=summary["nrmse_offdiag_dense"]))
            combined_by_arm[arm] = summary["combined_dense"]
        print(f"{name:<14} " + "  ".join(f"{a}={combined_by_arm[a]:.6f}" for a in arms))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily_rows).to_csv(out_dir / "daily_scores.csv", index=False)
    pd.DataFrame(window_rows).to_csv(out_dir / "window_scores.csv", index=False)
    print(f"wrote {out_dir / 'daily_scores.csv'}, {out_dir / 'window_scores.csv'}")

    comp_rows = []
    for arm_a, arm_b in compare_pairs:
        for name in names:
            if (name, arm_a) not in per_day_by or (name, arm_b) not in per_day_by:
                continue
            w = windows.get_window(name, days, na, settings.anchor_before_days)
            parts = paired_bootstrap_components(per_day_by[(name, arm_a)], per_day_by[(name, arm_b)],
                                                w.target, reps=args.reps)
            diag_a, offdiag_a = metric.nrmse_components(per_day_by[(name, arm_a)], w.target)
            a_value = {"combined": (diag_a + offdiag_a) / 2, "diag": diag_a, "offdiag": offdiag_a}
            for component, (est, lo, hi) in parts.items():
                comp_rows.append(dict(window=name, arm_a=arm_a, arm_b=arm_b, component=component,
                                      mean_diff=est, pct_of_a=100 * est / a_value[component],
                                      ci_lo=lo, ci_hi=hi, reps=args.reps))

        roll_present = [n for n in windows.ROLL_WINDOW_NAMES if n in names]
        if len(roll_present) == len(windows.ROLL_WINDOW_NAMES) and \
                all((n, arm_a) in per_day_by and (n, arm_b) in per_day_by for n in roll_present):
            dates_all, diffs_all = [], []
            for name in windows.ROLL_WINDOW_NAMES:
                w = windows.get_window(name, days, na, settings.anchor_before_days)
                da = metric.daily_combined(per_day_by[(name, arm_a)])
                db = metric.daily_combined(per_day_by[(name, arm_b)])
                for d in w.target:
                    dates_all.append(d)
                    diffs_all.append(da[d] - db[d])
            est, lo, hi, _ = date_cluster_bootstrap(dates_all, diffs_all, reps=args.reps)
            comp_rows.append(dict(window="pooled_roll", arm_a=arm_a, arm_b=arm_b, component="combined",
                                  mean_diff=est, pct_of_a=np.nan, ci_lo=lo, ci_hi=hi, reps=args.reps))
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(out_dir / "comparisons.csv", index=False)
        print(f"wrote {out_dir / 'comparisons.csv'}")


DEFAULT_COMPARE_PAIRS = ("april_mean:anchor", "anchor:anchor_rts", "anchor_rts_fullgrid:anchor_rts",
                        "anchor_rts:anchor_rts_fm", "anchor_rts_fm:final", "april_mean:final",
                        "novdec_mean:final")


def _settings_from_args(args):
    overrides = {}
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        overrides.update(config.Settings.parse(key, value))
    return config.Settings(**overrides)


def cmd_submit(args):
    cache_dir = Path(args.cache)
    days, na = data.load_eval_box(None, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    full_days = data.load_full_grid(None, cache=cache_dir / "full_grid.npz")
    na_full = set(config.NA_DAYS)
    all_full_dates = sorted(d for d in full_days if d not in na_full)
    settings = _settings_from_args(args)

    w = windows.test_gap_window(settings.anchor_before_days)

    fg_keys, fg_train, fg_calib = combine.full_grid_context(full_days, w, all_full_dates)
    fg_pred = combine.build_fullgrid_prediction(full_days, w, na_full, fg_keys, fg_train, fg_calib, settings)
    out_path = Path(args.out)
    fullgrid_path = Path(args.fullgrid_out) if args.fullgrid_out else out_path.with_name(out_path.stem + ".fullgrid.tsv")
    submission.write_fullgrid_tsv(fg_pred, w.target, fullgrid_path)
    print(f"full-grid build -> {fullgrid_path}")

    keys, train, calib = combine.eval_box_context(days, w, all_dates)
    ctx_path = Path(args.forecasts) / f"ctx_{w.name}.npz"
    fc_path = Path(args.forecasts) / f"fc_{w.name}.npz"
    forecast_diag = foundation.read_forecast(ctx_path, fc_path)
    final_pred = combine.build_final(days, w, na, keys, train, calib, forecast_diag, settings)

    base_lines = submission.read_tsv_lines(fullgrid_path)
    submission.splice_inbox(final_pred, base_lines, out_path)
    ok = submission.verify_inbox_written(out_path, final_pred, w.target)
    diffs = submission.verify_outbox_unchanged(out_path, base_lines, w.target)
    print(f"submission -> {out_path}  in-box round-trip {'PASS' if ok else 'FAIL'}  "
          f"out-of-box entries changed: {diffs}")
    if not ok or diffs:
        sys.exit("submission failed its own round-trip check")

    if args.validator:
        result = subprocess.run([sys.executable, str(args.validator), str(out_path)],
                                capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())


def cmd_forecast(args):
    from . import forecast_runner
    from .forecasters import get as get_forecaster
    from .forecasters import load_all as load_all_forecasters

    load_all_forecasters()
    adapter_cls = get_forecaster(args.model)
    kwargs = {}
    if args.backend is not None:
        kwargs["backend"] = args.backend
    if args.device is not None:
        kwargs["device"] = args.device
    forecaster = adapter_cls.from_pretrained(args.weights, **kwargs)

    for line in forecast_runner.run_planted_gate(forecaster, check_infill=args.infill):
        print(line)

    summaries = forecast_runner.run_all(
        forecaster, args.contexts, args.out, joint_top=args.joint_top, infill=args.infill,
        diagonal_only=args.diagonal_only, context_days_fwd=args.context_days_fwd,
        context_days_bwd=args.context_days_bwd, sort_quantiles=args.sort_quantiles)
    for name, n_series in summaries:
        print(f"{name:<14} series {n_series}")
    print(f"wrote {len(summaries)} forecast files -> {args.out}")


def cmd_compare_slot(args):
    from . import bakeoff
    cache_dir = Path(args.cache)
    days, na = data.load_eval_box(None, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    arm_dirs = dict(a.split("=", 1) for a in (args.arms or []))
    means = {}
    for spec in args.mean or []:
        name, ingredients = spec.split("=", 1)
        means[name] = ingredients.split(",")
    window_names = args.windows or list(windows.VALIDATION_WINDOW_NAMES)
    lenient = set(args.lenient or [])

    window_scores, daily_rows, gate_rows = bakeoff.run(
        days, na, all_dates, args.contexts, args.reference, arm_dirs, means,
        window_names, args.reps, lenient)
    summary = bakeoff.summarize(window_scores, daily_rows, gate_rows, list(arm_dirs) + list(means), args.reps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_scores.to_csv(out_dir / "bakeoff_windows.csv", index=False)
    summary.to_csv(out_dir / "bakeoff_summary.csv", index=False)
    gate_rows.to_csv(out_dir / "bakeoff_gates.csv", index=False)
    print(window_scores.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"wrote bakeoff_windows.csv, bakeoff_summary.csv, bakeoff_gates.csv -> {out_dir}")

    joint_dir = arm_dirs.get(args.shock_origin_joint_arm) if args.shock_origin_joint_arm else None
    if joint_dir:
        per, shock_summary, legs = bakeoff.run_shock_origin(days, na, all_dates, args.contexts,
                                                             args.reference, joint_dir, args.reps)
        per.to_csv(out_dir / "shock_origin.csv", index=False)
        shock_summary.to_csv(out_dir / "shock_origin_summary.csv", index=False)
        legs.to_csv(out_dir / "shock_origin_legs.csv", index=False)
        print(per.to_string(index=False))
        print(shock_summary.to_string(index=False))
        print(f"wrote shock_origin.csv, shock_origin_summary.csv, shock_origin_legs.csv -> {out_dir}")


def cmd_sweep(args):
    import pandas as pd
    cache_dir = Path(args.cache)
    days, na = data.load_eval_box(None, cache=cache_dir / "eval_box.npz")
    all_dates = sorted(d for d in days if d not in na)
    full_days = data.load_full_grid(None, cache=cache_dir / "full_grid.npz")
    na_full = set(config.NA_DAYS)
    all_full_dates = sorted(d for d in full_days if d not in na_full)
    window_names = args.windows or list(windows.VALIDATION_WINDOW_NAMES)
    reference_arm, reference_row = args.reference.split(":") if ":" in args.reference else (args.reference, None)

    grid = json.loads(Path(args.grid).read_text()) if args.grid else {"submitted": {}}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    submitted_scores = None
    for run_name, overrides in grid.items():
        settings = config.Settings(**{k: (tuple(v) if k == "tau_range" else v) for k, v in overrides.items()})
        pipe = Pipeline(days, na, all_dates, full_days, all_full_dates, args.forecasts, settings)
        rows = []
        for name in window_names:
            w = windows.get_window(name, days, na, settings.anchor_before_days)
            obs = {d: days[d] for d in w.target}
            pred = pipe.build(args.arm, w)
            summary, per_day = metric.score(obs, pred)
            per_day_idx = per_day.set_index("date")
            for d in w.target:
                rows.append(dict(window=name, date=d, combined=metric.daily_combined(per_day_idx)[d]))
        df = pd.DataFrame(rows)
        if run_name == "submitted":
            submitted_scores = df
        pooled = None
        roll_present = [n for n in windows.ROLL_WINDOW_NAMES if n in window_names]
        if submitted_scores is not None and len(roll_present) == len(windows.ROLL_WINDOW_NAMES):
            merged = df.merge(submitted_scores, on=["window", "date"], suffixes=("", "_submitted"))
            roll = merged[merged.window.isin(windows.ROLL_WINDOW_NAMES)]
            if len(roll):
                est, lo, hi, _ = date_cluster_bootstrap(roll.date.tolist(),
                                                        (roll.combined - roll.combined_submitted).tolist(),
                                                        reps=args.reps)
                pooled = dict(pooled_diff_vs_submitted=est, ci_lo=lo, ci_hi=hi)
        table = df.groupby("window").combined.mean().reset_index()
        table.to_csv(out_dir / f"sweep_{run_name}.csv", index=False)
        note = f"  pooled diff vs submitted: {pooled}" if pooled else ""
        print(f"{run_name:<20} overrides={overrides}{note}")
        if run_name == "submitted" and args.assert_reference:
            ref_df = pd.read_csv(args.assert_reference)
            merged = table.merge(ref_df, on="window", suffixes=("", "_ref"))
            worst = (merged.combined - merged.combined_ref).abs().max()
            print(f"  submitted-configuration reproduction check: max abs diff {worst:.3e} "
                 f"-> {'PASS' if worst <= 1e-12 else 'FAIL'}")
            if worst > 1e-12:
                sys.exit("sweep's submitted-configuration row does not reproduce the phase-1 reference")
    print(f"wrote sweep_<name>.csv -> {out_dir}")


def cmd_describe(args):
    from . import describe as describe_mod
    stats = describe_mod.describe(args.data)
    text = describe_mod.format_report(stats)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(prog="humob26", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prepare", help="parse the raw dataset into a cache directory")
    pp.add_argument("--data", required=True, help="path to the raw dataset TSV")
    pp.add_argument("--cache", required=True, help="cache directory to write")
    pp.set_defaults(func=cmd_prepare)

    pc = sub.add_parser("contexts", help="export bidirectional forecasting contexts")
    pc.add_argument("--cache", required=True)
    pc.add_argument("--out", required=True, help="directory to write ctx_<window>.npz files")
    pc.add_argument("--windows", nargs="+", default=None,
                    help=f"default: all {len(windows.ALL_WINDOW_NAMES)} windows")
    pc.set_defaults(func=cmd_contexts)

    pf = sub.add_parser("forecast", help="run a registered forecaster adapter over exported contexts")
    pf.add_argument("--model", required=True, help="registered forecaster name, e.g. timesfm3/chronos2/patchtst_fm")
    pf.add_argument("--backend", default=None, choices=["mlx", "torch"])
    pf.add_argument("--weights", required=True, help="path to the model's pretrained weights")
    pf.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    pf.add_argument("--contexts", required=True, help="directory of ctx_<window>.npz files")
    pf.add_argument("--out", required=True, help="directory to write fc_<window>.npz files")
    pf.add_argument("--joint-top", type=int, default=None, metavar="K",
                    help="forecast the K largest-context diagonal series with one joint call")
    pf.add_argument("--infill", action="store_true",
                    help="reconstruct the whole before/gap/after block in one pass instead of two directional calls")
    pf.add_argument("--diagonal-only", action="store_true",
                    help="only forecast diagonal series; other rows are left at 0")
    pf.add_argument("--context-days-fwd", type=int, default=None, metavar="N")
    pf.add_argument("--context-days-bwd", type=int, default=None, metavar="M")
    pf.add_argument("--sort-quantiles", action="store_true",
                    help="monotone-rearrange (q10, median, q90) after forecasting")
    pf.set_defaults(func=cmd_forecast)

    pe = sub.add_parser("evaluate", help="score arms on validation windows")
    pe.add_argument("--cache", required=True)
    pe.add_argument("--forecasts", required=False, help="directory holding ctx_*/fc_*.npz forecast files")
    pe.add_argument("--out", required=True, help="directory to write score tables")
    pe.add_argument("--arms", nargs="+", default=None, choices=list(combine.ARM_NAMES),
                    help=f"default: every arm named in --compare, else all of {combine.ARM_NAMES}")
    pe.add_argument("--windows", nargs="+", default=None,
                    help=f"default: all {len(windows.VALIDATION_WINDOW_NAMES)} validation windows")
    pe.add_argument("--reps", type=int, default=config.BOOTSTRAP_REPS)
    pe.add_argument("--compare", action="append", default=None, metavar="A:B",
                    help="arm pair to bootstrap-compare; repeatable (default: the 7-pair cumulative build-up)")
    pe.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    help="override a model constant for this run only, e.g. --set rts_rank=8")
    pe.set_defaults(func=cmd_evaluate)

    pcs = sub.add_parser("compare-slot", help="forecaster-slot bake-off against a reference forecaster")
    pcs.add_argument("--cache", required=True)
    pcs.add_argument("--contexts", required=True, help="directory of ctx_<window>.npz files")
    pcs.add_argument("--reference", required=True, help="directory of the reference arm's fc_<window>.npz files")
    pcs.add_argument("--arms", action="append", default=None, metavar="NAME=DIR",
                    help="a challenger arm and its forecast directory; repeatable")
    pcs.add_argument("--mean", action="append", default=None, metavar="NAME=A,B,C",
                    help="define an arm as the elementwise mean of other arms (or \"reference\"); repeatable")
    pcs.add_argument("--lenient", nargs="+", default=None,
                    help="arm names whose quantile-crossing gate is a note, not a failure")
    pcs.add_argument("--shock-origin-joint-arm", default=None, metavar="NAME",
                    help="also run the shock-origin forward/backward leg check using this arm as the joint model")
    pcs.add_argument("--windows", nargs="+", default=None)
    pcs.add_argument("--reps", type=int, default=config.BOOTSTRAP_REPS)
    pcs.add_argument("--out", required=True, help="directory to write bake-off score tables")
    pcs.set_defaults(func=cmd_compare_slot)

    psw = sub.add_parser("sweep", help="re-score a small grid of --set overrides")
    psw.add_argument("--cache", required=True)
    psw.add_argument("--forecasts", required=False)
    psw.add_argument("--arm", default="final", choices=list(combine.ARM_NAMES))
    psw.add_argument("--grid", required=True, help="JSON {run_name: {setting: value, ...}}")
    psw.add_argument("--reference", default="submitted", help="the grid entry name the pooled diff is measured against")
    psw.add_argument("--assert-reference", default=None, metavar="CSV",
                    help="a per-window score CSV the \"submitted\" grid entry must reproduce to 1e-12")
    psw.add_argument("--windows", nargs="+", default=None)
    psw.add_argument("--out", required=True)
    psw.add_argument("--reps", type=int, default=config.BOOTSTRAP_REPS)
    psw.set_defaults(func=cmd_sweep)

    pd_ = sub.add_parser("describe", help="data-description statistics computed from the raw dataset")
    pd_.add_argument("--data", required=True, help="path to the raw dataset TSV (or a cache directory's source)")
    pd_.add_argument("--cache", required=False, help="unused; kept for a uniform command line")
    pd_.add_argument("--out", required=False, default=None, help="text file to also write the report to")
    pd_.set_defaults(func=cmd_describe)

    ps = sub.add_parser("submit", help="build a submission file for the real prediction gap")
    ps.add_argument("--cache", required=True)
    ps.add_argument("--forecasts", required=True, help="directory holding the test_gap ctx_*/fc_*.npz pair")
    ps.add_argument("--out", required=True, help="submission file to write")
    ps.add_argument("--fullgrid-out", default=None,
                    help="where to write the intermediate full-grid file (default: <out>.fullgrid.tsv)")
    ps.add_argument("--validator", default=None, help="path to the official validator script")
    ps.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    help="override a model constant (breaks byte-exact reproduction; for a sensitivity run only)")
    ps.set_defaults(func=cmd_submit)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
