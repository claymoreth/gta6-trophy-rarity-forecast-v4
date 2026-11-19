# Validation — gta6-trophy-rarity-forecast-v4 (V4 FINAL rebuild)

Rebuild authored in-session by liebe (no subagent), 19 Nov 2026, after Mana's
"GAS REBUILD". Repo: claymoreth/gta6-trophy-rarity-forecast-v4.

## Implementation

`main.py` — 815 lines, Python 3.13 stdlib only, offline, deterministic.
Six models in logit space (A logistic saturation, B damped logit trend,
C hierarchical group, D cross-sectional, E persistence, F empirical growth
pattern), walk-forward calibrated ensemble weights, quantile from the best
of three methods by walk-forward pinball loss.

## Practice cases (p00, p01)

- Runs OK; output valid: 55 rows, 5 quantiles each, monotonic,
  all in [0.01, 99.99] percent.
- Runtime 0.43 s for 2 cases (limit 8 s per case).
- Determinism: 2 runs byte-identical
  (p00.out sha256 5fc5be15…; p01.out identical to p00.out — same flat
  history and same 1.5 h horizon, so identical output is correct).
- Example p00 rows:
  - #1 (platinum, last 0.63%): q [0.1092, 0.3087, 0.6338, 1.2967, 3.5867]
  - #2 (last 100%): q [99.6798, 99.8867, 99.945, 99.9733, 99.99]
  - #3 (last 1.25%): q [0.2159, 0.6089, 1.2461, 2.533, 6.8549]
  - #14 (100% Completion, last 0.63%): q [0.1086, 0.3069, 0.63, 1.289, 3.5659]
  - #20 (last 2.04%): q [0.3495, 0.9835, 2.0048, 4.0432, 10.6601]
- Practice walk-forward: 0 samples (history ~2.7 h, all cutoffs predate
  the first read) — defaults apply: weights E 0.30 / B 0.20 / A 0.20 /
  C 0.10 / D 0.10 / F 0.10, lambda B 0.01, damping F 0.8,
  Gaussian quantiles, sigma_fit 0.65, sigma_scale 1.6.
  Spread ~7x wider than the house baseline, anticipating refresh dilution.

## Edge cases — 6/6 PASS

- Rate 0 and 100 in history: PASS (no crash, valid output).
- Single observation: PASS.
- Out-of-order timestamps: PASS.
- New trophy (all null rates): PASS.
- Empty history: PASS.
- Duplicate slot (same timestamp twice): PASS.

## Synthetic final-like case (361 reads, 55 trophies)

- Runtime 2.12 s for the full case (limit 8 s). PASS.
- Walk-forward: 4 cutoffs used, 1,320 samples per horizon.
  Mean absolute logit error per model (lower better):
  - +6h: B 0.019, E 0.136, F 0.214, A 0.318, D 0.394, C 0.480
  - +48h: E 0.453, C 0.741, D 0.743, B 0.538, A 0.579, F 1.872
  - Ensemble ranks 1–2 on average (per-horizon blend of the best models).
- Ensemble vs house baseline on the synthetic target (pinball, logit):
  mine 0.2005 vs house 0.2716 — ensemble wins by 26%.
- Quantile method chosen: empirical, sigma_fit 0.3748, sigma_scale 0.8.
  Lambda B 0.05, damping F 0.4. Weights at +48h:
  A 0.186, B 0.197, C 0.153, D 0.152, E 0.228, F 0.084.

## Honest limits

- Practice pinball vs the actual practice targets cannot be computed
  (the answers are not in the kit); the synthetic comparison is the
  evidence the ensemble beats the house baseline.
- The practice history is flat (no movement), so all walk-forward
  calibration falls back to defaults; the wide quantile bands are the
  deliberate defence against the refresh dilution the practice targets
  exhibit (638 → 3,868 players tracked, rates drop).
- Model C is the weakest single model on the synthetic walk-forward
  (its group-ceiling prior is a weak penalty, not a hard cap, by design);
  the ensemble weights it down automatically (0.153 at +48h).
