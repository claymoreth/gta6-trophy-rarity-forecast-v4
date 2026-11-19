#!/usr/bin/env python3
"""GTA VI trophy rarity forecast — V4 FINAL (rebuild, authored in-session).

Python 3.13, standard library only, offline, deterministic.
Reads /input/<id>.in (schema makefun/gta6-trophies-case/v1), writes
/output/<id>.out as {"trophies": [{"id": n, "q": [q05,q25,q50,q75,q95]}]}
with q in percent [0, 100], non-decreasing, one row per trophy.

Method: six models in logit space (A logistic saturation, B damped logit
trend, C hierarchical group, D cross-sectional, E persistence, F empirical
growth pattern), combined by a walk-forward calibrated ensemble; quantiles
from the best of three methods by walk-forward pinball loss.
All times in hours since launch 2026-11-19T00:00Z = t=0.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z_TAU = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
EPS = 1e-9

# Walk-forward cutoffs (hours before t_last) and horizons (hours ahead).
WF_CUTOFFS = [48.0, 24.0, 12.0, 6.0]
WF_HORIZONS = [6.0, 12.0, 24.0, 48.0]
LAMBDAS_B = [0.005, 0.01, 0.02, 0.05]
DAMPINGS_F = [1.0, 0.8, 0.6, 0.4]
SIGMA_SCALES = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]

# Group ceiling weak priors (percent, as fractions). Penalty only, never caps.
GROUP_CEILINGS = {
    "story": (0.60, 0.90),
    "collection": (0.02, 0.15),
    "gold": (0.03, 0.10),
    "completion": (0.01, 0.05),
    "unknown": (0.01, 0.90),
}

# Default parameters (used when walk-forward has no samples, e.g. practice).
DEF_WEIGHTS = {"A": 0.20, "B": 0.20, "C": 0.10, "D": 0.10, "E": 0.30, "F": 0.10}
DEF_LAMBDA_B = 0.01
DEF_DAMPING_F = 0.8
DEF_SIGMA_SCALE = 1.6
DEF_SIGMA_FIT = 0.65


def clip_frac(p):
    """Clip a fraction to [0.0001, 0.9999]."""
    if p < 0.0001:
        return 0.0001
    if p > 0.9999:
        return 0.9999
    return p


def logit(p):
    """log(p/(1-p)) for a fraction p in (0, 1)."""
    p = clip_frac(p)
    return math.log(p / (1.0 - p))


def sigmoid(y):
    """Stable sigmoid; exponent clipped to [-30, 30]."""
    if y > 30.0:
        y = 30.0
    elif y < -30.0:
        y = -30.0
    return 1.0 / (1.0 + math.exp(-y))


def parse_iso(s):
    """Parse an ISO-8601 UTC timestamp to an aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_launch(dt):
    return (dt - LAUNCH).total_seconds() / 3600.0


def round4(x):
    """Round half-up to 4 decimals; whole numbers emitted as int."""
    d = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if d == d.to_integral_value():
        return int(d)
    return float(d)


def median(vals):
    """Median of a non-empty list of floats."""
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def pinball_loss(y, q, tau):
    """Pinball loss 2*max(tau*d, (tau-1)*d), d = y - q (logit scale)."""
    d = y - q
    a = tau * d
    b = (tau - 1.0) * d
    return 2.0 * (a if a > b else b)


def classify_group(trophy):
    """Group a trophy by type from its name/description (deterministic)."""
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    if "100% completion" in desc or "100% completion" in text:
        return "completion"
    if "gold medal" in desc or "gold medal" in text:
        return "gold"
    if "chapter" in desc or "chapter" in text:
        return "story"
    if ("complete" in desc and ("stunt" in desc or "collect" in desc
                                or "catch" in desc or "fish" in desc)):
        return "collection"
    if "stunt" in desc or "collect" in desc:
        return "collection"
    return "unknown"


def build_series(case):
    """Build per-trophy time series from a case.

    Returns (trophies, series, t_at, t_target) where series[tid] is a list
    of (t_hours, rate_percent) sorted by time, skipping null rates.
    """
    trophies = case["trophies"]
    history = case.get("history") or []
    t_at = hours_since_launch(parse_iso(case["at"]))
    t_target = hours_since_launch(parse_iso(case["target"]))
    series = {}
    for idx, tr in enumerate(trophies):
        tid = tr["id"]
        pts = []
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            r = rates[idx]
            if r is None:
                continue
            t = hours_since_launch(parse_iso(read["at"]))
            pts.append((t, float(r)))
        pts.sort(key=lambda p: p[0])
        series[tid] = pts
    return trophies, series, t_at, t_target


# ---------------------------------------------------------------- models

def model_e_logit(pts, fallback_frac=0.01):
    """Persistence: logit of the last observed rate (fraction)."""
    if not pts:
        return logit(fallback_frac)
    return logit(pts[-1][1] / 100.0)


def model_b_logit(pts, delta, lam):
    """Damped logit trend. Returns predicted logit at t_last + delta."""
    if not pts:
        return None
    ys = [logit(r / 100.0) for _, r in pts]
    t_last = pts[-1][0]
    y_last = ys[-1]
    # Robust median slope per hour over the last 12h (fallback: all slopes).
    recent = []
    all_slopes = []
    for i in range(1, len(pts)):
        dt = pts[i][0] - pts[i - 1][0]
        if dt <= 0:
            continue
        s = (ys[i] - ys[i - 1]) / dt
        all_slopes.append(s)
        if pts[i][0] >= t_last - 12.0:
            recent.append(s)
    pool = recent if recent else all_slopes
    v = median(pool) if pool else 0.0
    if lam <= 0:
        return y_last + v * delta
    return y_last + v * (1.0 - math.exp(-lam * delta)) / lam


def _logistic_p(L, k, t, t0):
    """L / (1 + exp(-k (t - t0))) with a clipped exponent."""
    x = -k * (t - t0)
    if x > 30.0:
        x = 30.0
    elif x < -30.0:
        x = -30.0
    return L / (1.0 + math.exp(x))


def fit_logistic(pts, group_ceiling=None, fast=False):
    """Fit p(t) = L / (1 + exp(-k (t - t0))) by coarse-to-fine grid.

    Minimizes squared error in logit + 0.01 (L - last)^2 (+ a weak
    penalty when L exceeds the group ceiling prior). Returns
    (L, k, t0, sse) or None when fewer than 2 points are available.
    fast=True uses a reduced grid (walk-forward calibration only).
    """
    if len(pts) < 2:
        return None
    # Subsample to at most 40 points for the SSE (the curve is smooth;
    # this keeps a 361-read final case fast without changing the fit).
    if len(pts) > 40:
        idx = sorted(set(int(round(i * (len(pts) - 1) / 39.0))
                         for i in range(40)))
        fit_pts = [pts[i] for i in idx]
    else:
        fit_pts = pts
    last_frac = pts[-1][1] / 100.0
    ys = [logit(r / 100.0) for _, r in fit_pts]
    ts = [t for t, _ in fit_pts]
    t_first, t_last = ts[0], ts[-1]

    def sse(L, k, t0):
        total = 0.0
        for (t, _r), y in zip(fit_pts, ys):
            y_pred = logit(_logistic_p(L, k, t, t0))
            d = y_pred - y
            total += d * d
        total += 0.01 * (L - last_frac) ** 2
        if group_ceiling is not None:
            _lo_c, hi_c = group_ceiling
            if L > hi_c:
                total += 0.1 * (L - hi_c) ** 2
        return total

    if fast:
        L_vals = []
        L = last_frac
        while L <= min(0.999, last_frac + 0.5) + 1e-12:
            L_vals.append(L)
            L += 0.10
        if not L_vals:
            L_vals = [last_frac]
        k_vals = [0.001, 0.005, 0.02, 0.08, 0.2]
        t0_vals = [t_first - 48.0, 0.5 * (t_first + t_last),
                   t_last + 48.0]
    else:
        L_vals = []
        L = last_frac
        while L <= min(0.999, last_frac + 0.5) + 1e-12:
            L_vals.append(L)
            L += 0.05
        if not L_vals:
            L_vals = [last_frac]
        k_vals = [0.001 * (200.0 ** (i / 7.0)) for i in range(8)]
        span = (t_last + 48.0) - (t_first - 48.0)
        t0_vals = [t_first - 48.0 + span * i / 4.0 for i in range(5)]

    scored = []
    for L in L_vals:
        for k in k_vals:
            for t0 in t0_vals:
                scored.append((sse(L, k, t0), L, k, t0))
    scored.sort(key=lambda x: x[0])
    top = scored[:3]
    if fast:
        e_best, L_best, k_best, t0_best = top[0]
        return (L_best, k_best, t0_best, e_best)

    # Fine refine around the top-3 at 1/5 step.
    span = (t_last + 48.0) - (t_first - 48.0)
    t0_step = span / 4.0 / 5.0
    k_step = 200.0 ** (1.0 / 7.0 / 5.0)
    refined = []
    for (_e0, L0, k0, t00) in top:
        for dL in (-0.01, 0.0, 0.01):
            L1 = L0 + dL
            if L1 < last_frac or L1 > 0.999:
                continue
            for dk in (1.0 / k_step, 1.0, k_step):
                k1 = k0 * dk
                if k1 < 0.001 or k1 > 0.2:
                    continue
                for dt0 in (-t0_step, 0.0, t0_step):
                    t01 = t00 + dt0
                    if t01 < t_first - 72.0 or t01 > t_last + 72.0:
                        continue
                    refined.append((sse(L1, k1, t01), L1, k1, t01))
    if refined:
        refined.sort(key=lambda x: x[0])
        e_best, L_best, k_best, t0_best = refined[0]
    else:
        e_best, L_best, k_best, t0_best = top[0]
    return (L_best, k_best, t0_best, e_best)


def model_a_logit(pts, t_target, group_ceiling=None, fast=False, fit=None):
    """Logistic saturation prediction (logit) at t_target.

    Pass a precomputed fit (from fit_logistic) to reuse it.
    """
    if not pts:
        return None
    if len(pts) < 2:
        return model_e_logit(pts)
    if fit is None:
        fit = fit_logistic(pts, group_ceiling=group_ceiling, fast=fast)
    if fit is None:
        return model_e_logit(pts)
    L, k, t0, _e = fit
    return logit(_logistic_p(L, k, t_target, t0))


def _delta24(pts):
    """Logit change per 24h over the trophy's recent history (data given)."""
    if len(pts) < 2:
        return 0.0
    ys = [logit(r / 100.0) for _, r in pts]
    t_last = pts[-1][0]
    y_last = ys[-1]
    # Observation closest to (but not after) t_last - 24h.
    target = t_last - 24.0
    best_i = 0
    for i in range(len(pts)):
        if pts[i][0] <= target:
            best_i = i
        else:
            break
    t0 = pts[best_i][0]
    span = t_last - t0
    if span <= 0:
        return 0.0
    return (y_last - ys[best_i]) / (span / 24.0)


def build_group_stats(trophies, series, cutoff_t, a_fits=None):
    """Group medians from data strictly before cutoff_t.

    Returns dict group -> {"L": median L, "k": median k,
    "delta24": {tid: median 24h logit change of OTHER trophies}}.
    L/k medians come from the per-trophy logistic fits (a_fits, keyed by
    trophy id); when a_fits is None they are fitted here (fast grid).
    """
    by_group = {}
    for tr in trophies:
        tid = tr["id"]
        g = classify_group(tr)
        by_group.setdefault(g, []).append(tid)
    stats = {}
    for g, tids in by_group.items():
        Ls, ks = [], []
        d24 = {}
        for tid in tids:
            pts = [p for p in series[tid] if p[0] < cutoff_t]
            if len(pts) >= 2:
                fit = None
                if a_fits is not None:
                    fit = a_fits.get(tid)
                if fit is None:
                    fit = fit_logistic(pts, fast=True)
                if fit is not None:
                    Ls.append(fit[0])
                    ks.append(fit[1])
            d24[tid] = _delta24(pts)
        # Median 24h change of OTHER trophies, per trophy.
        others = {}
        for tid in tids:
            vals = [d24[o] for o in tids if o != tid]
            others[tid] = median(vals) if vals else 0.0
        stats[g] = {
            "L": median(Ls) if Ls else 0.05,
            "k": median(ks) if ks else 0.01,
            "delta24_others": others,
        }
    return stats


def model_c_logit(pts, t_target, group, stats, fit=None):
    """Hierarchical group logistic: own fit, shrunk to group median.

    Reuses the Model A fit when passed (the group ceiling weak prior is
    applied as a post-hoc pull when L exceeds it).
    """
    if not pts:
        return None
    if len(pts) < 2:
        return model_e_logit(pts)
    if fit is None:
        ceiling = GROUP_CEILINGS.get(group)
        fit = fit_logistic(pts, group_ceiling=ceiling, fast=False)
    if fit is None:
        return model_e_logit(pts)
    L, k, t0, _e = fit
    # Weak group-ceiling prior: pull L toward the ceiling when it exceeds it.
    ceiling = GROUP_CEILINGS.get(group)
    if ceiling is not None and L > ceiling[1]:
        L = ceiling[1] + 0.5 * (L - ceiling[1])
    n = len(pts)
    if n < 4:
        g = stats.get(group)
        if g is not None:
            w = n / (n + 3.0)
            L = w * L + (1.0 - w) * g["L"]
            k = w * k + (1.0 - w) * g["k"]
    return logit(_logistic_p(L, k, t_target, t0))


def model_d_logit(pts, delta, group, stats):
    """Cross-sectional: move from last toward the group ceiling."""
    if not pts:
        return None
    last_frac = pts[-1][1] / 100.0
    g = stats.get(group)
    if g is None:
        return model_e_logit(pts)
    Lg, kg = g["L"], g["k"]
    p = last_frac + (Lg - last_frac) * (1.0 - math.exp(-kg * delta))
    return logit(p)


def model_f_logit(pts, delta, tid, group, stats, damping):
    """Empirical growth pattern from the group's median 24h change."""
    if not pts:
        return None
    y_last = logit(pts[-1][1] / 100.0)
    g = stats.get(group)
    if g is None:
        return y_last
    dmed = g["delta24_others"].get(tid, 0.0)
    return y_last + dmed * (delta / 24.0) * damping


def _actual_at(pts, t_want):
    """Observed logit at (or nearest before) t_want; None if none."""
    best = None
    for (t, r) in pts:
        if t <= t_want:
            best = (t, r)
        else:
            break
    if best is None:
        return None
    return logit(best[1] / 100.0)


# ---------------------------------------------------------------- calibration

def _horizon_weights(losses):
    """Inverse-loss weights with a 0.05 prior on every model."""
    models = ["A", "B", "C", "D", "E", "F"]
    inv = {}
    for m in models:
        loss = losses.get(m)
        if loss is None:
            inv[m] = 0.0
        else:
            inv[m] = 1.0 / (loss + EPS)
    total = sum(inv.values())
    if total <= 0:
        return dict(DEF_WEIGHTS)
    w = {m: inv[m] / total for m in models}
    # Add the 0.05 prior to every model, then renormalize.
    w = {m: w[m] + 0.05 for m in models}
    s = sum(w.values())
    return {m: w[m] / s for m in models}


def calibrate(trophies, series, t_at):
    """Walk-forward calibration. Returns (config, diag).

    config: {"weights": {h: {model: w}}, "weights_group": {h: {group: {model: w}}},
             "lambda_b": {h: lam}, "damping_f": {h: d},
             "sigma_fit": float, "sigma_scale": float,
             "quantile_method": "gaussian" | "empirical",
             "empirical_q": {tau: residual logit}}
    diag: sample counts and per-model losses for the report.
    """
    # Last observation time across the case.
    t_last = None
    for tr in trophies:
        pts = series[tr["id"]]
        if pts and (t_last is None or pts[-1][0] > t_last):
            t_last = pts[-1][0]
    diag = {"cutoffs_used": [], "samples_per_horizon": {h: 0 for h in WF_HORIZONS},
            "losses": {}, "n_models": 0}
    if t_last is None:
        return _default_config(), diag

    # Collect (pred, actual) per model per horizon, plus per-group.
    # err[model][h] = list of abs errors; err_g[model][h][group] = list.
    err = {m: {h: [] for h in WF_HORIZONS} for m in "ABCDEF"}
    err_g = {m: {h: {} for h in WF_HORIZONS} for m in "ABCDEF"}
    # For lambda/damping selection: err_b[lam][h], err_f[damp][h].
    err_b = {lam: {h: [] for h in WF_HORIZONS} for lam in LAMBDAS_B}
    err_f = {d: {h: [] for h in WF_HORIZONS} for d in DAMPINGS_F}
    # Residuals for quantile calibration: (h, residual, group).
    residuals = []

    for cutoff in [t_last - c for c in WF_CUTOFFS]:
        # Training data strictly before the cutoff.
        train = {}
        any_train = False
        for tr in trophies:
            pts = [p for p in series[tr["id"]] if p[0] < cutoff]
            train[tr["id"]] = pts
            if pts:
                any_train = True
        if not any_train:
            continue
        diag["cutoffs_used"].append(cutoff)
        # Fit Model A once per trophy for this cutoff; reuse the fit for
        # the group stats and for Model A at every horizon. Fit Model C
        # separately (it carries the group-ceiling weak prior).
        a_fits = {}
        c_fits = {}
        for tr in trophies:
            pts = train[tr["id"]]
            if len(pts) >= 2:
                a_fits[tr["id"]] = fit_logistic(pts, fast=True)
                c_fits[tr["id"]] = fit_logistic(
                    pts, group_ceiling=GROUP_CEILINGS.get(classify_group(tr)),
                    fast=True)
        stats = build_group_stats(trophies, train, cutoff, a_fits=a_fits)
        for h in WF_HORIZONS:
            t_want = cutoff + h
            for tr in trophies:
                tid = tr["id"]
                full = series[tid]
                # Actual target: a NEW read after the cutoff, at/before t_want.
                new_read = [p for p in full if cutoff < p[0] <= t_want]
                if not new_read:
                    continue
                actual = logit(new_read[-1][1] / 100.0)
                pts = train[tid]
                if not pts:
                    continue
                g = classify_group(tr)
                delta = t_want - pts[-1][0]
                t_tgt = pts[-1][0] + delta
                preds = {}
                preds["E"] = model_e_logit(pts)
                preds["B"] = model_b_logit(pts, delta, DEF_LAMBDA_B)
                preds["A"] = model_a_logit(pts, t_tgt, fast=True,
                                           fit=a_fits.get(tid))
                preds["C"] = model_c_logit(pts, t_tgt, g, stats,
                                           fit=c_fits.get(tid))
                preds["D"] = model_d_logit(pts, delta, g, stats)
                # F with each damping (for selection) + default for the model.
                for d in DAMPINGS_F:
                    pf = model_f_logit(pts, delta, tid, g, stats, d)
                    if pf is not None:
                        err_f[d][h].append(abs(pf - actual))
                preds["F"] = model_f_logit(pts, delta, tid, g, stats,
                                           DEF_DAMPING_F)
                for lam in LAMBDAS_B:
                    pb = model_b_logit(pts, delta, lam)
                    if pb is not None:
                        err_b[lam][h].append(abs(pb - actual))
                for m, p in preds.items():
                    if p is None:
                        continue
                    e = abs(p - actual)
                    err[m][h].append(e)
                    err_g[m][h].setdefault(g, []).append(e)
                    diag["samples_per_horizon"][h] += 1
                # Residual of the (default-weight) ensemble median, for
                # quantile calibration.
                ens = sum(DEF_WEIGHTS[m] * preds[m] for m in "ABCDEF"
                          if preds.get(m) is not None)
                wsum = sum(DEF_WEIGHTS[m] for m in "ABCDEF"
                           if preds.get(m) is not None)
                if wsum > 0:
                    residuals.append((h, actual - ens / wsum, g))

    # Mean losses per model per horizon.
    losses = {}
    for m in "ABCDEF":
        losses[m] = {}
        for h in WF_HORIZONS:
            v = err[m][h]
            losses[m][h] = (sum(v) / len(v)) if v else None
    diag["losses"] = losses
    diag["n_models"] = 6

    # Weights per horizon (global + per group where >= 10 samples).
    weights, weights_group = {}, {}
    for h in WF_HORIZONS:
        lh = {m: losses[m][h] for m in "ABCDEF"}
        if all(v is None for v in lh.values()):
            weights[h] = dict(DEF_WEIGHTS)
            weights_group[h] = {}
            continue
        weights[h] = _horizon_weights(lh)
        wg = {}
        groups = set()
        for m in "ABCDEF":
            groups |= set(err_g[m][h].keys())
        for g in groups:
            lg = {}
            ok = True
            for m in "ABCDEF":
                v = err_g[m][h].get(g, [])
                if len(v) < 10:
                    ok = False
                    break
                lg[m] = sum(v) / len(v)
            if ok:
                wg[g] = _horizon_weights(lg)
        weights_group[h] = wg

    # Lambda B and damping F per horizon (lowest mean abs error).
    lambda_b, damping_f = {}, {}
    for h in WF_HORIZONS:
        best_lam, best_e = DEF_LAMBDA_B, None
        for lam in LAMBDAS_B:
            v = err_b[lam][h]
            if not v:
                continue
            e = sum(v) / len(v)
            if best_e is None or e < best_e:
                best_e, best_lam = e, lam
        lambda_b[h] = best_lam
        best_d, best_e = DEF_DAMPING_F, None
        for d in DAMPINGS_F:
            v = err_f[d][h]
            if not v:
                continue
            e = sum(v) / len(v)
            if best_e is None or e < best_e:
                best_e, best_d = e, d
        damping_f[h] = best_d

    # Quantile calibration from ensemble residuals.
    if residuals:
        res_vals = [r for (_h, r, _g) in residuals]
        mean_r = sum(res_vals) / len(res_vals)
        var = sum((r - mean_r) ** 2 for r in res_vals) / len(res_vals)
        sigma_fit = math.sqrt(var) if var > 0 else DEF_SIGMA_FIT
        if sigma_fit <= 0:
            sigma_fit = DEF_SIGMA_FIT
        # Empirical residual quantiles.
        sres = sorted(res_vals)
        emp_q = {}
        for tau in LEVELS:
            idx = tau * (len(sres) - 1)
            lo = int(math.floor(idx))
            hi = min(lo + 1, len(sres) - 1)
            frac = idx - lo
            emp_q[tau] = sres[lo] * (1.0 - frac) + sres[hi] * frac
        # Compare the three methods by mean pinball loss.
        def method_loss(kind, scale=1.0):
            tot, n = 0.0, 0
            for (h, r, _g) in residuals:
                # r = actual - pred, so actual logit = pred + r.
                # Quantile q_tau (logit offset from pred):
                for tau in LEVELS:
                    if kind == "empirical":
                        off = emp_q[tau]
                    else:
                        st = math.sqrt(sigma_fit ** 2 +
                                      (sigma_fit ** 2) * (h / 24.0)) * scale
                        # z for this tau
                        zi = Z_TAU[LEVELS.index(tau)]
                        off = zi * st
                    # pinball: d = actual - q = r - off
                    tot += pinball_loss(r, off, tau)
                    n += 1
            return tot / n if n else float("inf")
        loss_gauss = method_loss("gaussian", 1.0)
        loss_emp = method_loss("empirical")
        best_scale, best_loss = 1.0, loss_gauss
        for sc in SIGMA_SCALES:
            l = method_loss("gaussian", sc)
            if l < best_loss:
                best_loss, best_scale = l, sc
        if loss_emp <= best_loss and loss_emp <= loss_gauss:
            qmethod, qscale = "empirical", best_scale
        else:
            qmethod, qscale = "gaussian", best_scale
        config = {
            "weights": weights, "weights_group": weights_group,
            "lambda_b": lambda_b, "damping_f": damping_f,
            "sigma_fit": sigma_fit, "sigma_scale": qscale,
            "quantile_method": qmethod, "empirical_q": emp_q,
        }
    else:
        config = _default_config()
    return config, diag


def _default_config():
    return {
        "weights": {h: dict(DEF_WEIGHTS) for h in WF_HORIZONS},
        "weights_group": {h: {} for h in WF_HORIZONS},
        "lambda_b": {h: DEF_LAMBDA_B for h in WF_HORIZONS},
        "damping_f": {h: DEF_DAMPING_F for h in WF_HORIZONS},
        "sigma_fit": DEF_SIGMA_FIT, "sigma_scale": DEF_SIGMA_SCALE,
        "quantile_method": "gaussian", "empirical_q": {},
    }


def _weights_for_delta(config, delta):
    """Weights for an arbitrary horizon delta (hours)."""
    w48 = config["weights"][48.0]
    if delta <= 6.0:
        return config["weights"][6.0]
    if delta <= 48.0:
        # Linear interpolation between bracketing horizons.
        lo = max(h for h in WF_HORIZONS if h <= delta)
        hi = min(h for h in WF_HORIZONS if h >= delta)
        if lo == hi:
            return config["weights"][lo]
        f = (delta - lo) / (hi - lo)
        wl, wh = config["weights"][lo], config["weights"][hi]
        return {m: wl[m] * (1.0 - f) + wh[m] * f for m in "ABCDEF"}
    # Beyond 48h: blend the 48h weights toward the default.
    f = 48.0 / delta
    return {m: w48[m] * f + DEF_WEIGHTS[m] * (1.0 - f) for m in "ABCDEF"}


def _param_for_delta(param_map, delta, default):
    """Per-horizon parameter (lambda/damping) for an arbitrary delta."""
    if delta <= 6.0:
        return param_map[6.0]
    if delta <= 48.0:
        lo = max(h for h in WF_HORIZONS if h <= delta)
        hi = min(h for h in WF_HORIZONS if h >= delta)
        if lo == hi:
            return param_map[lo]
        # Nearest horizon (params are discrete choices).
        return param_map[lo] if (delta - lo) <= (hi - delta) else param_map[hi]
    return param_map[48.0]


# ---------------------------------------------------------------- predict

def predict_case(case):
    """Predict all trophies for one case. Returns (output_rows, diag)."""
    trophies, series, t_at, t_target = build_series(case)
    delta = t_target - t_at
    config, diag = calibrate(trophies, series, t_at)
    # Fit Model A once per trophy on the full history (full grid); reuse
    # the fit for the group stats and for Model A. Fit Model C separately
    # (it carries the group-ceiling weak prior).
    a_fits_full = {}
    c_fits_full = {}
    for tr in trophies:
        pts = series[tr["id"]]
        if len(pts) >= 2:
            a_fits_full[tr["id"]] = fit_logistic(pts)
            c_fits_full[tr["id"]] = fit_logistic(
                pts, group_ceiling=GROUP_CEILINGS.get(classify_group(tr)))
    stats = build_group_stats(trophies, series, t_at, a_fits=a_fits_full)
    weights_delta = _weights_for_delta(config, delta)
    lambda_delta = _param_for_delta(config["lambda_b"], delta, DEF_LAMBDA_B)
    damping_delta = _param_for_delta(config["damping_f"], delta,
                                     DEF_DAMPING_F)
    sigma_fit = config["sigma_fit"]
    sigma_scale = config["sigma_scale"]
    sigma_time = sigma_fit * math.sqrt(max(delta, 0.0) / 24.0)
    sigma_total = math.sqrt(sigma_fit ** 2 + sigma_time ** 2) * sigma_scale
    # Horizon scaling for empirical offsets (reference: 24h).
    sigma_total_24 = math.sqrt(sigma_fit ** 2 + sigma_fit ** 2) * sigma_scale
    emp_scale = (sigma_total / sigma_total_24) if sigma_total_24 > 0 else 1.0

    rows = []
    for tr in trophies:
        tid = tr["id"]
        pts = series[tid]
        g = classify_group(tr)
        # Per-group weights at the nearest horizon, else global weights.
        w = weights_delta
        if delta <= 48.0:
            h_near = min(WF_HORIZONS, key=lambda h: abs(h - delta))
            wg = config["weights_group"].get(h_near, {}).get(g)
            if wg is not None:
                w = wg
        if not pts:
            # No history at all: fall back to the group median rate.
            fb = stats.get(g, {}).get("L", 0.01)
            ens_logit = logit(fb)
        else:
            preds = {}
            preds["E"] = model_e_logit(pts)
            preds["B"] = model_b_logit(pts, delta, lambda_delta)
            preds["A"] = model_a_logit(pts, t_target,
                                       group_ceiling=GROUP_CEILINGS.get(g),
                                       fit=a_fits_full.get(tid))
            preds["C"] = model_c_logit(pts, t_target, g, stats,
                                       fit=c_fits_full.get(tid))
            preds["D"] = model_d_logit(pts, delta, g, stats)
            preds["F"] = model_f_logit(pts, delta, tid, g, stats,
                                       damping_delta)
            num, den = 0.0, 0.0
            for m in "ABCDEF":
                p = preds.get(m)
                if p is None:
                    continue
                wm = w.get(m, DEF_WEIGHTS[m])
                num += wm * p
                den += wm
            ens_logit = num / den if den > 0 else model_e_logit(pts)
        # Quantiles.
        qs = []
        for i, tau in enumerate(LEVELS):
            if config["quantile_method"] == "empirical" and config["empirical_q"]:
                off = config["empirical_q"].get(tau, Z_TAU[i] * sigma_total)
                off = off * emp_scale
            else:
                off = Z_TAU[i] * sigma_total
            q_logit = ens_logit + off
            q_pct = sigmoid(q_logit) * 100.0
            qs.append(q_pct)
        # Clip to [0.01, 99.99] percent, enforce non-decreasing.
        qs = [min(99.99, max(0.01, q)) for q in qs]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tid, "q": [round4(q) for q in qs]})
    diag["delta_hours"] = delta
    diag["sigma_fit"] = sigma_fit
    diag["sigma_scale"] = sigma_scale
    diag["sigma_total"] = sigma_total
    diag["quantile_method"] = config["quantile_method"]
    diag["lambda_b"] = lambda_delta
    diag["damping_f"] = damping_delta
    diag["weights"] = weights_delta
    return rows, diag


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    args = list(argv if argv is not None else sys.argv[1:])
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            in_dir = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        else:
            i += 1
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows, _diag = predict_case(case)
        out = {"trophies": rows}
        text = json.dumps(out, separators=(",", ":")) + "\n"
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
