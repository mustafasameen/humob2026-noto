"""Experiment 2: municipality shares using the published submission and mapping.

Municipality numerators contain only evaluation-box within-cell flows. All raw
OD rows, including cross-boundary flows, enter the denominator. The outside-grid
comparison uses the within-bucket (-1_-1 -> -1_-1) flow as its numerator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, evidence, windows
from .calendar import daterange

MUNICIPALITIES = tuple(evidence.MUNI_CENTROID)
OUTSIDE = 'outside_grid'
GROUPS = MUNICIPALITIES + (OUTSIDE,)
JANUARY = tuple(daterange('20240122', '20240131'))
APRIL = tuple(daterange('20240401', '20240430'))
CALENDAR = tuple(daterange('20231101', '20241031'))


def mapping_table():
    cells = np.arange(config.N_EVAL_CELLS)
    assignment = evidence.cell_to_muni(cells)
    return pd.DataFrame({'cell': cells,
                         'y': cells // config.N_EVAL_X + config.EVAL_Y[0],
                         'x': cells % config.N_EVAL_X + config.EVAL_X[0],
                         'municipality': [assignment[int(c)] for c in cells]})


def aggregate_day(date, arr, assignment):
    """Return the all-row total and disjoint within-cell group numerators."""
    count = arr['count']
    if not np.isfinite(count).all() or (count < 0).any():
        raise ValueError(f'{date}: invalid flow values')
    total = float(count.sum())
    if total <= 0:
        raise ValueError(f'{date}: daily total must be positive')
    flow = {name: 0.0 for name in GROUPS}
    f = data.to_eval_box(date, arr)
    for key, value in zip(f.key[f.is_diag], f.val[f.is_diag]):
        flow[assignment[int(key // config.N_EVAL_CELLS)]] += float(value)
    outside_diag = ((arr['oy'] == -1) & (arr['ox'] == -1)
                    & (arr['dy'] == -1) & (arr['dx'] == -1))
    flow[OUTSIDE] = float(count[outside_diag].sum())
    if sum(flow.values()) > total + 1e-8:
        raise ValueError(f'{date}: group numerators exceed all-row total')
    return total, flow


def read_observed(path, assignment):
    rows, totals, missing, seen = {}, [], [], set()
    for date, arr in data.iter_days(path, keep_oob=True):
        if date in seen:
            raise ValueError(f'duplicate raw date {date}')
        seen.add(date)
        if arr is None:
            missing.append(date)
            continue
        total, flow = aggregate_day(date, arr, assignment)
        rows[date] = (total, flow)
        totals.append({'date': date, 'all_row_total': total,
                       'outside_within_flow': flow[OUTSIDE],
                       'outside_any_endpoint_flow': float(arr['count'][
                           (arr['oy'] == -1) | (arr['dy'] == -1)].sum())})
    if not rows:
        raise ValueError('no observed days')
    values = np.array([x['all_row_total'] for x in totals])
    fixed_total = float(values.mean())
    if not np.allclose(values, fixed_total, rtol=1e-12, atol=1e-8):
        raise ValueError('raw daily totals are not constant; cannot use one denominator in gap')
    return rows, pd.DataFrame(totals), missing, fixed_total


def build_daily(raw_path, submission_path):
    mapping = mapping_table()
    assignment = dict(zip(mapping.cell, mapping.municipality))
    observed, totals, missing, fixed_total = read_observed(raw_path, assignment)
    reconstructed = {}
    for date, arr in data.iter_days(submission_path, keep_oob=True):
        if date in reconstructed or arr is None:
            raise ValueError(f'invalid/duplicate submission date {date}')
        _, flow = aggregate_day(date, arr, assignment)
        reconstructed[date] = flow
    expected = set(windows.test_gap_window().target)
    if set(reconstructed) != expected:
        raise ValueError('submission must contain exactly the 58 original scored gap dates')
    if set(observed) & expected:
        raise ValueError('raw observations unexpectedly include a submission target')
    output = []
    for date in CALENDAR:
        for group in GROUPS:
            if date in observed:
                denominator, flows = observed[date]
                numerator, status, source = flows[group], 'observed', 'observed'
            elif date in reconstructed and group != OUTSIDE:
                denominator = fixed_total
                numerator, status, source = reconstructed[date][group], 'reconstructed', 'reconstructed'
            else:
                denominator, numerator, source = fixed_total, np.nan, 'unavailable'
                status = ('excluded_by_challenge' if date in config.EXCLUDED_TEST_DAYS else
                          'outside_bucket_not_predicted' if date in reconstructed else
                          'missing_observation' if date in missing else 'not_available')
            output.append({'date': date, 'municipality': group, 'source': source,
                           'status': status, 'within_flow': numerator,
                           'denominator': denominator, 'share': numerator / denominator})
    return pd.DataFrame(output), mapping, totals


def bootstrap_ratios(before, after, reps=config.BOOTSTRAP_REPS, seed=config.BOOTSTRAP_SEED):
    """Independent day resampling within anchors; shared indices across groups.

    Input shape: days x groups. Return ratio and percentile CI by group, plus
    the count of undefined bootstrap ratios. Zero denominators are never hidden
    by dropping replicates; their affected intervals are reported as undefined.
    """
    before, after = np.asarray(before, float), np.asarray(after, float)
    if before.ndim != 2 or after.ndim != 2 or before.shape[1] != after.shape[1]:
        raise ValueError('anchor arrays must have matching group dimensions')
    if len(before) == 0 or len(after) == 0 or not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError('anchors must contain finite observed days')
    if (before < 0).any() or (after < 0).any() or reps < 1:
        raise ValueError('shares must be nonnegative and reps positive')
    rng = np.random.default_rng(seed)
    b = before[rng.integers(len(before), size=(reps, len(before)))].mean(axis=1)
    a = after[rng.integers(len(after), size=(reps, len(after)))].mean(axis=1)
    ratios = np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    bm, am = before.mean(axis=0), after.mean(axis=0)
    estimate = np.divide(am, bm, out=np.full_like(am, np.nan), where=bm > 0)
    bad = (~np.isfinite(ratios)).sum(axis=0)
    lo, hi = np.full(len(bm), np.nan), np.full(len(bm), np.nan)
    valid = bad == 0
    if valid.any():
        lo[valid], hi[valid] = np.percentile(ratios[:, valid], [2.5, 97.5], axis=0)
    return estimate, lo, hi, bad


def first_halfway(before, after, dates, shares):
    """First available gap date reaching >=50% of the signed anchor change."""
    if not np.isfinite([before, after]).all():
        raise ValueError('anchor shares must be finite')
    dates, shares = list(dates), list(shares)
    if len(dates) != len(shares) or len(set(dates)) != len(dates):
        raise ValueError('gap dates must be unique and align with shares')
    delta = after - before
    if np.isclose(before, after, rtol=1e-12, atol=1e-15):
        return None, 'no_change'
    available = [(d, s) for d, s in sorted(zip(dates, shares)) if np.isfinite(s)]
    if not available:
        return None, 'no_reconstruction'
    for date, value in available:
        if (value - before) / delta >= 0.5:
            return date, 'reached'
    return None, 'not_reached'


def summarize(daily, reps=config.BOOTSTRAP_REPS, seed=config.BOOTSTRAP_SEED):
    observed = daily[daily.source == 'observed']
    wide = observed.pivot(index='date', columns='municipality', values='share').reindex(columns=GROUPS)
    before = wide.loc[wide.index.isin(JANUARY)]
    after = wide.loc[wide.index.isin(APRIL)]
    ratio, lo, hi, bad = bootstrap_ratios(before.values, after.values, reps, seed)
    bm, am = before.mean(), after.mean()
    rows = []
    for i, group in enumerate(GROUPS):
        pred = daily[(daily.municipality == group) & (daily.source == 'reconstructed')]
        half_date, half_status = first_halfway(bm[group], am[group], pred.date, pred.share)
        if group == OUTSIDE:
            half_date, half_status = None, 'outside_bucket_not_predicted'
        rows.append({'municipality': group, 'january_mean_share': bm[group],
                     'april_mean_share': am[group], 'change_share': am[group] - bm[group],
                     'april_over_january': ratio[i], 'ratio_ci_lo': lo[i], 'ratio_ci_hi': hi[i],
                     'ratio_status': 'zero_january_mean' if bm[group] == 0 else
                                     'undefined_bootstrap_denominator' if bad[i] else 'defined',
                     'undefined_bootstrap_replicates': int(bad[i]),
                     'january_observed_days': len(before), 'april_observed_days': len(after),
                     'halfway_share': (am[group] + bm[group]) / 2,
                     'halfway_date': half_date, 'halfway_status': half_status,
                     'reconstructed_days': len(pred), 'bootstrap_reps': reps, 'bootstrap_seed': seed})
    return pd.DataFrame(rows)


def plot_shares(daily, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, MonthLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = {'nanao': 'Nanao', 'wajima': 'Wajima', 'suzu': 'Suzu', 'anamizu': 'Anamizu',
             'noto_cho': 'Noto town', 'shika': 'Shika', 'nakanoto': 'Nakanoto', 'hakui': 'Hakui',
             OUTSIDE: 'Outside-grid bucket (observed only)'}
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
    calendar = pd.date_range('2023-11-01', '2024-04-30')
    for ax, group in zip(axes.flat, GROUPS):
        df = daily[(daily.municipality == group) & (daily.date <= '20240430')].copy()
        df.index = pd.to_datetime(df.date)
        ax.axvspan(pd.Timestamp('2024-02-01'), pd.Timestamp('2024-03-31'), color='#eaf0f5', zorder=0)
        for source, color in [('observed', '#16465c'), ('reconstructed', '#bc541e')]:
            series = df.loc[df.source == source, 'share'].reindex(calendar) * 100
            ax.plot(calendar, series, color=color, lw=1.25, marker='.', markersize=2)
        row = summary[summary.municipality == group].iloc[0]
        if group != OUTSIDE:
            ax.axhline(row.halfway_share * 100, color='#747b80', ls=':', lw=1)
            if row.halfway_status == 'reached':
                date = pd.Timestamp(row.halfway_date)
                ax.axvline(date, color='#bc541e', ls='--', lw=.8, alpha=.7)
                ax.text(.03, .95, f'First halfway: {date:%b %d}', transform=ax.transAxes,
                        va='top', fontsize=8, color='#954217')
        ax.set_title(names[group], loc='left', fontsize=11, fontweight='bold')
        ax.set_xlim(calendar[0], calendar[-1])
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b'))
        ax.tick_params(labelsize=8)
        ax.set_ylabel('Share of total (%)', fontsize=8)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=.18)
    fig.suptitle('Municipality activity shares around the missing period', x=.065, ha='left',
                 fontsize=17, fontweight='bold', y=.98)
    fig.text(.065, .943, 'Within-cell flows / all-row daily total  |  November 2023 - April 2024  |  Each panel has its own scale', fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color='#16465c', label='Observed'),
                        Line2D([0], [0], color='#bc541e', label='Reconstructed'),
                        Line2D([0], [0], color='#747b80', ls=':', label='Halfway between anchor means'),
                        Patch(facecolor='#eaf0f5', label='February-March gap')],
               loc='lower left', bbox_to_anchor=(.055, .035), ncol=4, frameon=False, fontsize=9)
    fig.text(.065, .022, 'Missing observations and the two excluded gap dates are left blank. Mapping uses nearest municipality centroids.', fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, top=.89, bottom=.13, hspace=.38, wspace=.3)
    fig.savefig(output / 'municipality_shares.png', dpi=180)
    fig.savefig(output / 'municipality_shares.pdf')
    plt.close(fig)
