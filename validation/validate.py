#!/usr/bin/env python3
"""Validation for gta6-trophy-rarity-forecast-v4 main.py (authored in-session).

Tests: edge cases (rate 0/100, 1 obs, out-of-order, new trophy, empty
history), determinism, and runtime on a synthetic final-like case
(361 reads, 55 trophies). Writes results to validation-results.json.
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def make_case(cid, at, target, trophies, history):
    return {"id": cid, "at": at, "horizon_minutes": 90, "target": target,
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19",
                     "source": "Exophase"},
            "trophies": trophies, "history": history, "notes": []}


def trophy(tid, name, desc, tier="bronze", points=15):
    return {"id": tid, "name": name, "description": desc,
            "tier": tier, "points": points, "secret": False}


def read(at_iso, players, rates):
    return {"slot": at_iso, "at": at_iso, "players_tracked": players,
            "trophies_earned": [None] * len(rates), "rates": rates}


def check_output(rows, label):
    assert len(rows) == 55, f"{label}: {len(rows)} rows"
    for r in rows:
        q = r["q"]
        assert len(q) == 5, f"{label} {r}"
        assert all(isinstance(x, (int, float)) for x in q), f"{label} {r}"
        assert all(math.isfinite(x) for x in q), f"{label} {r}"
        assert all(0.01 <= x <= 99.99 for x in q), f"{label} {r}"
        assert all(q[i] <= q[i + 1] for i in range(4)), f"{label} {r}"
    return True


def main():
    results = {"edge_cases": {}, "determinism": {}, "runtime": {},
               "walk_forward": {}}
    # Base trophies: 55, mirroring the real set (ids 1-55).
    trs = []
    for i in range(1, 56):
        if i == 1:
            trs.append(trophy(1, "Vice Got a Price!", "Congratulations! You're a Leonida Legend.", "platinum", 300))
        elif i == 2:
            trs.append(trophy(2, "Just Like Paradise", "Complete Gone Fishing."))
        elif i == 3:
            trs.append(trophy(3, "Paradise Lost", "Complete Chapter 1."))
        elif i == 14:
            trs.append(trophy(14, "A Long Road", "Attain 100% Completion.", "gold", 90))
        elif i == 15:
            trs.append(trophy(15, "Cunning Stunts", "Complete 250 Stunts.", "silver", 30))
        elif i == 17:
            trs.append(trophy(17, "The Shootist", "Earn a gold medal in all Shooting Range challenges.", "silver", 30))
        else:
            trs.append(trophy(i, f"Trophy {i}", f"Do thing {i}."))

    # --- Edge cases ---
    # 1. Rate 0 and 100 in history.
    hist = [read("2026-11-19T01:00:00Z", 100, [0.0] * 55),
            read("2026-11-19T02:00:00Z", 200, [100.0] * 55)]
    c = make_case("edge01", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, hist)
    rows, _ = M.predict_case(c)
    check_output(rows, "rate 0/100")
    results["edge_cases"]["rate_0_100"] = "PASS"

    # 2. Single observation.
    hist = [read("2026-11-19T01:00:00Z", 100, [1.0] * 55)]
    c = make_case("edge02", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, hist)
    rows, _ = M.predict_case(c)
    check_output(rows, "1 obs")
    results["edge_cases"]["one_obs"] = "PASS"

    # 3. Out-of-order timestamps.
    hist = [read("2026-11-19T02:00:00Z", 200, [1.5] * 55),
            read("2026-11-19T01:00:00Z", 100, [1.0] * 55),
            read("2026-11-19T02:30:00Z", 300, [1.2] * 55)]
    c = make_case("edge03", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, hist)
    rows, _ = M.predict_case(c)
    check_output(rows, "out-of-order")
    results["edge_cases"]["out_of_order"] = "PASS"

    # 4. New trophy: all null rates for trophy 55.
    rates1 = [1.0] * 54 + [None]
    rates2 = [1.5] * 54 + [None]
    hist = [read("2026-11-19T01:00:00Z", 100, rates1),
            read("2026-11-19T02:00:00Z", 200, rates2)]
    c = make_case("edge04", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, hist)
    rows, _ = M.predict_case(c)
    check_output(rows, "new trophy nulls")
    results["edge_cases"]["new_trophy_nulls"] = "PASS"

    # 5. Empty history.
    c = make_case("edge05", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, [])
    rows, _ = M.predict_case(c)
    check_output(rows, "empty history")
    results["edge_cases"]["empty_history"] = "PASS"

    # 6. Duplicate slot (same timestamp twice).
    hist = [read("2026-11-19T01:00:00Z", 100, [1.0] * 55),
            read("2026-11-19T01:00:00Z", 100, [1.1] * 55),
            read("2026-11-19T02:00:00Z", 200, [1.5] * 55)]
    c = make_case("edge06", "2026-11-19T03:00:00Z", "2026-11-19T04:30:00Z", trs, hist)
    rows, _ = M.predict_case(c)
    check_output(rows, "duplicate slot")
    results["edge_cases"]["duplicate_slot"] = "PASS"

    # --- Determinism (practice p00, 2 runs via predict_case) ---
    with open(os.path.join(BASE, "input", "p00.in")) as f:
        p00 = json.load(f)
    r1, _ = M.predict_case(p00)
    r2, _ = M.predict_case(p00)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    results["determinism"]["predict_case_2x"] = "PASS (identical)"

    # --- Synthetic final-like case: 361 reads, real movement ---
    rng = random.Random(42)
    hist = []
    t = LAUNCH - timedelta(hours=0, minutes=10)
    players = 638
    # Base rates per trophy (percent).
    base = {}
    for i in range(1, 56):
        if i == 2:
            base[i] = 100.0
        elif i == 1:
            base[i] = 0.63
        elif i == 3:
            base[i] = 1.25
        else:
            base[i] = round(rng.uniform(0.3, 2.5), 2)
    # Simulate growth + a refresh dilution at ~04:30Z.
    for k in range(361):
        t = t + timedelta(minutes=10)
        if k < 30:
            players = 638
        elif k == 30:
            players = 3868  # refresh dilution
        else:
            players = 3868 + int((k - 30) * 2)
        rates = []
        for i in range(1, 56):
            if i == 2:
                rates.append(100.0)
                continue
            # Slow logit growth, then dilution at refresh.
            growth = 1.0 + 0.02 * k / 361.0
            r = base[i] * growth
            if k >= 30:
                r = r * 638.0 / 3868.0 * (1.0 + 0.01 * (k - 30))
            rates.append(round(max(0.01, min(99.99, r)), 4))
        hist.append(read(t.strftime("%Y-%m-%dT%H:%M:%SZ"), players, rates))
    at = (LAUNCH + timedelta(hours=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    target = (LAUNCH + timedelta(hours=108.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    synth = make_case("synth-final", at, target, trs, hist)
    t0 = time.time()
    rows, diag = M.predict_case(synth)
    elapsed = time.time() - t0
    check_output(rows, "synth final")
    results["runtime"]["synth_361_reads_s"] = round(elapsed, 2)
    results["runtime"]["under_8s"] = elapsed < 8.0
    results["walk_forward"]["cutoffs_used"] = diag["cutoffs_used"]
    results["walk_forward"]["samples_per_horizon"] = diag["samples_per_horizon"]
    results["walk_forward"]["losses"] = diag["losses"]
    results["walk_forward"]["delta_hours"] = diag["delta_hours"]
    results["walk_forward"]["quantile_method"] = diag["quantile_method"]
    results["walk_forward"]["sigma_fit"] = round(diag["sigma_fit"], 4)
    results["walk_forward"]["sigma_scale"] = diag["sigma_scale"]
    results["walk_forward"]["lambda_b"] = diag["lambda_b"]
    results["walk_forward"]["damping_f"] = diag["damping_f"]
    results["walk_forward"]["weights_48h"] = {k: round(v, 3) for k, v in diag["weights"].items()}

    # Ensemble vs house baseline on the synth target (pinball, logit).
    # House baseline: persistence + random-walk spread, sigma 0.5*sqrt(days).
    def house_q(last_pct, delta_h, tau):
        d = delta_h / 24.0
        z = [-1.645, -0.674, 0.0, 0.674, 1.645][[0.05, 0.25, 0.5, 0.75, 0.95].index(tau)]
        y = M.logit(last_pct / 100.0) + z * 0.5 * math.sqrt(d)
        return M.sigmoid(y) * 100.0

    # Actual synth target rates: continue the simulation to the target.
    # (Recompute the rate at target using the same growth rule.)
    actual = {}
    for i in range(1, 56):
        if i == 2:
            actual[i] = 100.0
            continue
        k_tgt = 361 + int((108.5 - 60.0) * 6)  # 10-min steps beyond the case
        growth = 1.0 + 0.02 * k_tgt / 361.0
        r = base[i] * growth * 638.0 / 3868.0 * (1.0 + 0.01 * (k_tgt - 30))
        actual[i] = round(max(0.01, min(99.99, r)), 4)

    def pinball_mean(rows_out, actual_map):
        tot, n = 0.0, 0
        for r in rows_out:
            y = M.logit(actual_map[r["id"]] / 100.0)
            for qi, tau in enumerate([0.05, 0.25, 0.5, 0.75, 0.95]):
                q = M.logit(r["q"][qi] / 100.0)
                tot += M.pinball_loss(y, q, tau)
                n += 1
        return tot / n if n else float("nan")

    mine = pinball_mean(rows, actual)
    # House rows.
    last_rates = {}
    for i in range(1, 56):
        last_rates[i] = hist[-1]["rates"][i - 1]
    house_rows = []
    for i in range(1, 56):
        qs = [house_q(last_rates[i], 48.5, tau) for tau in [0.05, 0.25, 0.5, 0.75, 0.95]]
        qs = [min(99.99, max(0.01, q)) for q in qs]
        house_rows.append({"id": i, "q": qs})
    house = pinball_mean(house_rows, actual)
    results["walk_forward"]["synth_pinball_mine"] = round(mine, 4)
    results["walk_forward"]["synth_pinball_house"] = round(house, 4)
    results["walk_forward"]["beats_house"] = mine < house

    with open(os.path.join(BASE, "validation", "validation-results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
