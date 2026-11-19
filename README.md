# GTA VI Trophy Rarity Forecast — V4 FINAL

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/claymoreth/gta6-trophy-rarity-forecast-v4/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/claymoreth/gta6-trophy-rarity-forecast-v4/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z.

## Program

`main.py` — Python 3.13, standard library only, offline, deterministic, under 8 seconds for 55 trophies on 1 CPU / 1024 MB.

It reads `/input/<id>.in` (schema `makefun/gta6-trophies-case/v1`: 55 trophies, history of Exophase reads before the checkpoint) and writes `/output/<id>.out` as one JSON object:

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

Quantiles are in percent [0, 100], non-decreasing, one row per trophy in input order.

## Method

All times in hours since launch 2026-11-19T00:00Z = t=0. All fitting in logit scale y = log(p/(1-p)), p clipped [0.0001, 0.9999]. Stable sigmoid (exponent clipped to [-30, 30]).

Six models, combined by a walk-forward calibrated ensemble:

- **A — Logistic Saturation.** p(t) = L / (1 + exp(-k (t - t0))). Coarse-to-fine grid (no brute force): coarse L in [last, min(0.999, last+0.5)] step 0.05, k log-grid 8 points 0.001–0.2/h, t0 5 points in [t_first-48h, t_last+48h]; fine refine top-3 at 1/5 step. Minimize squared error in logit + penalty 0.01 (L - last)^2. Under 600 evals per trophy. Falls back to B, then E.
- **B — Damped Logit Trend.** y(t) = y_last + v (1 - exp(-lambda delta)) / lambda. v is the robust MEDIAN logit slope per hour over the last 12h (not OLS). Lambda tried [0.005, 0.01, 0.02, 0.05]/h, chosen by walk-forward.
- **C — Hierarchical Group.** Groups by trophy type: story/chapter, collection/animal/stunt, gold-medal/skill, completion/100%, unknown. Group ceilings (story 0.6–0.9, collection 0.02–0.15, gold 0.03–0.10, completion 0.01–0.05) are WEAK PRIORS only — a penalty, never a hard cap. Group k and L medians computed only from data before each fold's cutoff. Trophies with fewer than 4 observations shrink to the group median with weight n/(n+3).
- **D — Cross-sectional.** For thin trophies: p50 = last + (L_group - last) (1 - exp(-k_group delta)), using L_group and k_group from Model C in the same fold.
- **E — Persistence.** p = last value. Often best for trophies above 95%.
- **F — Empirical Growth Pattern.** y_target = y_last + delta_median_group (delta_target / 24h) damping. delta_median_group is the median 24h logit change of other trophies in the same group, only from data before the cutoff. Damping is chosen from [1.0, 0.8, 0.6, 0.4] by walk-forward (not hardcoded); fallback 0.8 when data is insufficient.

**Ensemble weighting.** Walk-forward cutoffs [t_last-48h, t_last-24h, t_last-12h, t_last-6h]; train on data before the cutoff; predict +6h, +12h, +24h, +48h. Pinball loss in logit per model per horizon per group. Folds without an actual target are not counted. Weights w_m = (1 / (loss_m + eps)) / sum(1 / (loss + eps)) + 0.05 prior to every model. Weights per horizon, and per group when group samples >= 10, else global. Models with fewer than 5 valid samples in a horizon/group get no calibrated weight (global / persistence fallback). Final prediction is the weighted average in LOGIT, then sigmoid.

**Quantiles.** Three methods compared in validation, lowest mean pinball loss wins: (1) Gaussian with sigma_fit, (2) empirical residual quantiles, (3) Gaussian with calibrated sigma_scale in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]. sigma_total = sqrt(sigma_fit^2 + sigma_time^2) * sigma_scale; sigma_fit from walk-forward residuals (not in-sample only); sigma_time grows with sqrt(delta). q_logit = logit(p50) + z * sigma_total (Gaussian) or + empirical residual quantile; z = [-1.645, -0.674, 0, 0.674, 1.645]. Enforced q05 < q25 < q50 < q75 < q95 (eps 0.0001), clipped to [0.01, 99.99] percent.

**Validation integrity.** Every walk-forward fold uses ONLY data before its cutoff for all preprocessing, grouping, shrinkage, fitting, weighting, and sigma calibration. Chronological folds. No model is selected by final-target performance. Sample counts per model, group, and horizon are reported. If a complex model does not consistently beat persistence or damped trend, its weight is reduced automatically. Every model produces a valid prediction for every trophy, with fallback when fitting fails.

## Key data insight

Exophase refreshes its counts on its own schedule. The practice history is constant (638 players tracked, every rate identical), but the practice targets are post-refresh: 3,868 players tracked, and every rate except the one at 100% LOWER than before (dilution — the denominator jumped ~6x while earners barely grew). The ensemble's wide quantile bands anticipate this refresh dilution; the house baseline's narrow band does not.

## Validation

See `validation/VALIDATION.md` and `validation/validation-results.json` for the walk-forward table, ensemble vs house baseline, edge cases, determinism proof, and runtime.

## License

MIT. See LICENSE.
