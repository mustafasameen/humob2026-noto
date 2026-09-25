"""Fixed constants for the HuMob Challenge 2026 pipeline.

Everything that is a *number the method depends on* lives here, in one
place, so that every other module reads it from a single source instead of
repeating literals.  Nothing in this file depends on any other module in
the package.

The task is to predict a contiguous 58-day gap of origin-destination (OD)
flow on a 100 x 70 grid of 2 km cells, given flows observed before and
after the gap.  Two things follow from that setup and recur everywhere
below: dates are handled as an ordered, gap-aware axis (never a plain
calendar), and every OD pair is addressed either in "full-grid" space
(all 7,000 cells) or in "evaluation-box" space (the 1,476-cell region the
metric actually scores).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
# Grid ids are strings "y_x" with x in 1..100 and y in 1..70; both axes
# increase with the coordinate.  Cells are not square (about 1.80 km
# east-west by 2.32 km north-south at the evaluation box's latitude), but
# no distance-based computation is used in this phase, so only the counts
# below matter.
GRID_NX = 100
GRID_NY = 70
GRID_N_CELLS = GRID_NX * GRID_NY          # 7,000; linear id = (y-1)*NX + (x-1)
OUT_OF_BOUNDS_ID = "-1_-1"

# Evaluation boundary (inclusive) fixed by the challenge scoring region.
EVAL_X = (30, 70)
EVAL_Y = (35, 70)
N_EVAL_X = EVAL_X[1] - EVAL_X[0] + 1               # 41
N_EVAL_Y = EVAL_Y[1] - EVAL_Y[0] + 1               # 36
N_EVAL_CELLS = N_EVAL_X * N_EVAL_Y                 # 1,476
N_EVAL_OFFDIAG_PAIRS = N_EVAL_CELLS * (N_EVAL_CELLS - 1)   # 2,177,100

# ---------------------------------------------------------------------------
# Metric normalisers
# ---------------------------------------------------------------------------
# Combined NRMSE = mean(NRMSE_diag, NRMSE_offdiag), each RMSE divided by its
# own normaliser.  Convention is "dense": every one of the N_EVAL_CELLS
# diagonal pairs and N_EVAL_OFFDIAG_PAIRS off-diagonal pairs counts on every
# scored day, an absent pair contributing zero.
NORM_DIAG = 26.57
NORM_OFFDIAG = 0.0176

# ---------------------------------------------------------------------------
# Calendar exceptions
# ---------------------------------------------------------------------------
# Days present in the raw file but marked not-a-number for data-quality
# reasons; they must never be averaged into a level as an observed zero.
NA_DAYS = (
    "20231126", "20231130", "20231201", "20231203", "20231204", "20231205",
    "20231214", "20240118", "20240123", "20240124", "20240202", "20240305",
    "20240408", "20240426", "20240529", "20240708",
)
# Within the scored gap, these two are excluded from scoring altogether.
EXCLUDED_TEST_DAYS = ("20240202", "20240305")

# ---------------------------------------------------------------------------
# The prediction gap and its anchor windows
# ---------------------------------------------------------------------------
GAP_START = "20240201"
GAP_END = "20240331"
# The before-anchor is the last 10 observed days immediately before the gap.
ANCHOR_BEFORE_START = "20240101"
ANCHOR_BEFORE_END = "20240131"
ANCHOR_BEFORE_DAYS = 10
# The anchor after the gap is the full following month.
ANCHOR_AFTER_START = "20240401"
ANCHOR_AFTER_END = "20240430"

# ---------------------------------------------------------------------------
# Day-type factor shrinkage
# ---------------------------------------------------------------------------
# Per-cell weekday/holiday factors are pulled toward the pooled (mass
# weighted, cross-cell) shape so that a handful of noisy observations
# cannot set a whole cell's weekly profile.  0.35 is the fixed shrink used
# off the diagonal (and as the plain, non-adaptive variant everywhere);
# on the diagonal an empirical-Bayes weight replaces this fixed value
# cell by cell (see anchors.daytype_factors_eb).
DAYTYPE_SHRINK = 0.35

# ---------------------------------------------------------------------------
# State-space (RTS) smoother
# ---------------------------------------------------------------------------
# Number of spatial factors the smoother reduces the grid to before
# bridging the gap in factor space and reconstructing.
RTS_RANK = 12
# Clip the reconstructed level to this multiple of the largest observed
# value for the same pair, guarding against the smoother extrapolating a
# trend past what is physically plausible.
RTS_CLIP_CAP = 3.0

# ---------------------------------------------------------------------------
# Model blend weights
# ---------------------------------------------------------------------------
# The two-anchor interpolation and the RTS smoother are combined in equal
# measure everywhere they are blended.
ANCHOR_RTS_BLEND = 0.5
# On the diagonal only, the interpolation+RTS blend is itself averaged
# 50/50 with a foundation-model forecast.
FM_DIAGONAL_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Recovery-weight time-constant prior
# ---------------------------------------------------------------------------
# The two-anchor blend weight on the diagonal is not a single assumed curve
# shape; it is a point prediction under an exponential relaxation model,
# level(t) = A - (A - L0) * exp(-(t - t0) / tau), averaged over a prior on
# the relaxation time constant tau because tau is not identified from the
# data at hand.
#   TAU_DATA_FIT_RANGE   the candidate range considered for tau.
#   TAU_LITERATURE_RANGE a narrower sub-range used as the actual prior
#                        support (Yabe et al., J. R. Soc. Interface
#                        17:20190532 (2020), Fig. 1), which sits inside the
#                        range above.
TAU_DATA_FIT_RANGE = (4.0, 30.0)
TAU_LITERATURE_RANGE = (7.588, 26.88)

# RECOVERY_WEIGHT_TABLE holds, for gap days t = 0 .. 59, the normalised
# exponential recovery weight
#     w(t) = (exp(-t / tau) - exp(-T / tau)) / (1 - exp(-T / tau))
# with T = 60 (the first after-anchor day), averaged over tau on a fine
# uniform grid spanning TAU_LITERATURE_RANGE. w(0) = 1 (all weight on the
# before-anchor) and w(59) is a small positive residual rather than exactly
# 0, since the model does not assume the after-anchor sits exactly at the
# relaxation's asymptote. Recomputing that average here confirms the
# formula and the general shape (matches this stored table to about 1e-3);
# the table itself is kept as a fixed numeric input to the method, like a
# fitted coefficient, so the submission reproduces exactly regardless of
# how that average is taken.
RECOVERY_WEIGHT_TABLE = (
    1.0, 0.9346297786289975, 0.873918444995706, 0.8174918114516923,
    0.7650088531489676, 0.7161585433961704, 0.6706570074690605, 0.6282449615939921,
    0.5886854073979682, 0.5517615553018944, 0.5172749531683578, 0.48504379904041967,
    0.4549014190578619, 0.42669489364260244, 0.400283816832646, 0.3755391752380076,
    0.352342334513896, 0.33058412251501845, 0.3101639994269979, 0.2909893061815905,
    0.27297458336486974, 0.2560409536337176, 0.24011556137625215, 0.22513106399562824,
    0.21102516977225624, 0.19774021777426704, 0.18522279574663908, 0.17342339232156842,
    0.16229608026169987, 0.1517982277782966, 0.14189023526251898, 0.13253529503332293,
    0.12369917194338131, 0.11535000289780677, 0.1074581135318842, 0.09999585046585116,
    0.09293742770905876, 0.08625878592443981, 0.07993746338876453, 0.07395247759613702,
    0.06828421655290842, 0.06291433890279145, 0.057825682102548355, 0.05300217794208733,
    0.04842877476899638, 0.04409136583721569, 0.039976723253352615, 0.036072437042705206,
    0.03236685890087481, 0.02884905023643376, 0.025508734145882454, 0.02233625099444723,
    0.01932251730552451, 0.016458987688035893, 0.013737619554907664, 0.011150840407607486,
    0.00869151748132388, 0.00635292956322439, 0.004128740812406217, 0.002012976424845995,
)

# ---------------------------------------------------------------------------
# Bootstrap defaults
# ---------------------------------------------------------------------------
BOOTSTRAP_REPS = 4000
BOOTSTRAP_SEED = 0

# ---------------------------------------------------------------------------
# Overridable settings (sensitivity sweeps)
# ---------------------------------------------------------------------------
# Every model constant a sweep is allowed to touch, gathered into one
# object so `cli.py`'s `--set key=value` and `sweep` can override them for a
# single run without changing the defaults every other command uses. The
# defaults here reproduce the submitted configuration exactly.
from dataclasses import dataclass, field   # noqa: E402


@dataclass(frozen=True)
class Settings:
    rts_rank: int = RTS_RANK
    anchor_rts_blend: float = ANCHOR_RTS_BLEND
    daytype_shrink: float = DAYTYPE_SHRINK
    eb_enabled: bool = True
    anchor_before_days: int = ANCHOR_BEFORE_DAYS
    fm_diagonal_weight: float = FM_DIAGONAL_WEIGHT
    tau_range: tuple = TAU_LITERATURE_RANGE
    single_tau: float = None          # bypasses the tau-averaged table entirely when set
    # Where the recovery curve runs on a window: "target" spans the target days
    # (the gap's own geometry whenever both anchors touch the target);
    # "anchors" spans the day after the before-anchor to the day before the
    # after-anchor, so a window whose anchors do not touch it sits where it
    # would in the gap.
    curve_span: str = "target"

    @classmethod
    def parse(cls, key, value):
        """Coerce a `--set key=value` string pair to this field's type."""
        types = {f: type(getattr(cls(), f)) for f in cls.__dataclass_fields__}
        if key == "tau_range":
            lo, hi = value.split(",")
            return {"tau_range": (float(lo), float(hi))}
        if key not in types:
            raise KeyError(f"unknown setting {key!r}; choose from {sorted(types)}")
        cast = bool if types[key] is bool else (float if types[key] in (float, type(None)) else types[key])
        parsed = value if cast is str else cast(value) if not (cast is bool) else value.lower() in ("1", "true", "yes")
        return {key: parsed}
