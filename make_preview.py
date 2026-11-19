#!/usr/bin/env python3
"""Render forecast-p00-preview.png: horizontal log-scale chart of all 55
trophies (light bar q05-q95, darker bar q25-q75, diamond q50, red tick
last rate). Reads input/p00.in for names/last rates and p00-preview.out
for the forecast. Authored in-session for the v4 rebuild."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "input", "p00.in")) as f:
    case = json.load(f)
with open(os.path.join(BASE, "p00-preview.out")) as f:
    out = json.load(f)

trophies = case["trophies"]
last_rates = [h["rates"][i] for i, h in enumerate([case["history"][-1]] * 0) ] if False else None
# Last rate per trophy = last non-null rate in history.
last = []
for i in range(len(trophies)):
    v = None
    for h in case["history"]:
        r = h["rates"][i]
        if r is not None:
            v = r
    last.append(v if v is not None else 0.01)

rows = {r["id"]: r["q"] for r in out["trophies"]}

fig, ax = plt.subplots(figsize=(12, 16))
y = list(range(len(trophies)))[::-1]
for idx, tr in enumerate(trophies):
    tid = tr["id"]
    q = rows[tid]
    yi = len(trophies) - 1 - idx
    # Light bar q05-q95.
    ax.plot([q[0], q[4]], [yi, yi], color="#c9d6e3", lw=7, solid_capstyle="butt", zorder=1)
    # Darker bar q25-q75.
    ax.plot([q[1], q[3]], [yi, yi], color="#5b7fa6", lw=7, solid_capstyle="butt", zorder=2)
    # Diamond q50.
    ax.plot([q[2]], [yi], marker="D", color="#1f3a5f", ms=6, zorder=3)
    # Red tick: last rate.
    lr = max(last[idx], 0.01)
    ax.plot([lr, lr], [yi - 0.35, yi + 0.35], color="#c0392b", lw=2, zorder=4)

ax.set_xscale("log")
ax.set_xlim(0.05, 120)
ax.set_ylim(-0.6, len(trophies) - 0.4)
ax.set_yticks(y)
ax.set_yticklabels([f"#{tr['id']} {tr['name']}" for tr in trophies], fontsize=8)
ax.set_xlabel("Earn rate % (log scale) — light bar q05–q95, dark bar q25–q75, ◆ q50, red tick = last rate", fontsize=10)
ax.set_title("GTA VI trophy forecast — practice case p00 (target 04:30Z)\nV4 FINAL (rebuild, authored in-session)", fontsize=12)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "forecast-p00-preview.png"), dpi=110)
print("wrote forecast-p00-preview.png")
