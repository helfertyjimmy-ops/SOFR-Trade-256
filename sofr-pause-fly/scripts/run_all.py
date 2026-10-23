"""Reproduce every number and chart used in the pitch deck.

Usage:  python scripts/run_all.py          (writes CSVs and PNGs to outputs/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sofr_fly import bachelier as bl                      # noqa: E402
from sofr_fly.contract import JAN_WEIGHT, MarketState, sfrz6_price   # noqa: E402
from sofr_fly.montecarlo import (BASE_CASE, CENTRAL, HAWKISH, MARKET_LIKE, Scenario,  # noqa: E402
                                 calibrate_basis, simulate, wirp_bounds_p_exactly_one)
from sofr_fly.payoff import Butterfly, Condor, breakeven_probability, pnl_dollars  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RED, GREY, INK = "#E0605B", "#999999", "#1A1A1A"
T_EXPIRY = 58 / 365          # 14-Oct-2026 -> 11-Dec-2026
FLY, CONDOR = Butterfly(), Condor()


def section(title):
    print(f"\n=== {title} ===")


# 1. Market data: parity forward, normal vol, traded fly/condor ------------------------------
section("1. Market check (thinkorswim /SR3Z26, 14-Oct-2026 mids)")
ch = pd.read_csv(ROOT / "data" / "sfrz6_dec_options_2026-10-14.csv")
ch["call_mid"] = (ch.call_bid + ch.call_ask) / 2
ch["put_mid"] = (ch.put_bid + ch.put_ask) / 2
F = bl.parity_forward(ch.strike, ch.call_mid, ch.put_mid)
atm = ch.iloc[(ch.strike - F).abs().argsort().iloc[0]]
vol = bl.straddle_implied_vol(atm.call_mid + atm.put_mid, F, atm.strike, T_EXPIRY)
sd_bp = vol * np.sqrt(T_EXPIRY) * 100
ch["normal_vol_bp"] = [100 * (bl.implied_vol(r.call_mid, F, r.strike, T_EXPIRY, "call") if r.strike >= F
                              else bl.implied_vol(r.put_mid, F, r.strike, T_EXPIRY, "put")) for r in ch.itertuples()]
c = ch.set_index("strike")
fly_calls = (c.call_mid[95.75] - 2 * c.call_mid[95.8125] + c.call_mid[95.875]) * 100
fly_puts = (c.put_mid[95.75] - 2 * c.put_mid[95.8125] + c.put_mid[95.875]) * 100
fly_legged = (c.call_ask[95.75] - 2 * c.call_bid[95.8125] + c.call_ask[95.875]) * 100
condor_calls = (c.call_mid[95.6875] - c.call_mid[95.75] - c.call_mid[95.8125] + c.call_mid[95.875]) * 100
fly_flat = bl.butterfly(F, 95.8125, 0.0625, T_EXPIRY, vol) * 100
print(f"Parity-implied forward: {F:.4f}  | ATM normal vol {vol*100:.1f}bp/yr -> {sd_bp:.1f}bp s.d. to expiry")
print(f"Pause fly 95.75/95.8125/95.875: call mid {fly_calls:.2f}bp, put mid {fly_puts:.2f}bp, legged at bid/ask {fly_legged:.2f}bp, flat-vol {fly_flat:.2f}bp")
print(f"Condor 95.6875-95.875 (call mids): {condor_calls:.2f}bp")
ch.round(4).to_csv(OUT / "market_check.csv", index=False)
COST = round((fly_calls + fly_puts) / 2, 2)

# 2. Contract mechanics: landing-zone map ----------------------------------------------------
section("2. SFRZ6 landing zones on 11-Dec (price by hikes x Jan odds)")
wirp = pd.read_csv(ROOT / "data" / "wirp_2026-10-01.csv")
mean_dec = float(wirp.loc[wirp.meeting == "2026-12-09", "cumulative_hikes"].iloc[0])
p_oct = float(wirp.loc[wirp.meeting == "2026-10-28", "cumulative_hikes"].iloc[0])
jan_mkt = float(wirp.loc[wirp.meeting == "2027-01-27", "cumulative_hikes"].iloc[0]) - mean_dec
basis = calibrate_basis(F, MARKET_LIKE.p_hikes, jan_mkt)
state = MarketState(basis_bp=basis)
print(f"Jan meeting weight in SFRZ6: {JAN_WEIGHT:.3f} (48/91) | WIRP Jan odds {jan_mkt:.3f} | calibrated basis {basis:.2f}bp")
lo, hi = wirp_bounds_p_exactly_one(mean_dec, p_oct)
print(f"WIRP-consistent range for P(exactly one hike by Dec): {lo:.3f} - {hi:.3f}")
grid = pd.DataFrame({f"Jan {int(j*100)}%": [sfrz6_price(k, j, state) for k in range(3)] for j in (0, 0.25, 0.528, 1.0)},
                    index=["0 hikes", "1 hike", "2 hikes"]).round(3)
print(grid.to_string())
grid.to_csv(OUT / "landing_zones.csv")

# 3. Deterministic scenario P&L --------------------------------------------------------------
section(f"3. Scenario P&L at expiry (fly cost {COST}bp, condor cost {condor_calls:.2f}bp)")
rows = [("2 hikes by Dec", 2, 0.5), ("1 hike, Jan kept live", 1, 0.5), ("1 hike, data-dependent", 1, 0.25),
        ("1 hike, clear pause", 1, 0.05), ("0 hikes by Dec", 0, 0.6)]
pl = pd.DataFrame([{"outcome": n, "SFRZ6": round(sfrz6_price(k, j, state), 3), "jan_odds": j,
                    "fly_pnl_$": float(pnl_dollars(FLY.payoff_bp(sfrz6_price(k, j, state)), COST)),
                    "condor_pnl_$": float(pnl_dollars(CONDOR.payoff_bp(sfrz6_price(k, j, state)), condor_calls))}
                   for n, k, j in rows]).round(2)
print(pl.to_string(index=False))
pl.to_csv(OUT / "scenario_pnl.csv", index=False)
be = FLY.breakevens(COST)
print(f"Fly breakevens: {be[0]:.4f} / {be[1]:.4f} | max profit ${pnl_dollars(FLY.max_payoff_bp, COST):.2f} | max loss ${COST*25:.2f}")
print(f"Breakeven probability of landing in zone (avg payoff 4bp, fees 0.25bp): {breakeven_probability(COST, 4.0, 0.25):.0%}")

# 4. Monte Carlo expected payoff ------------------------------------------------------------
section("4. Monte Carlo expected payoff by scenario (bp)")
res = []
for scn in (MARKET_LIKE, BASE_CASE, CENTRAL, HAWKISH):
    x = simulate(scn, F, basis)
    res.append({"scenario": scn.name, "P(0/1/2)": "/".join(f"{p:.0%}" for p in scn.p_hikes),
                "sd_bp": x.std() * 100, "P(95.75-95.875)": np.mean((x > 95.75) & (x < 95.875)),
                "fly_EV_bp": FLY.payoff_bp(x).mean(), "condor_EV_bp": CONDOR.payoff_bp(x).mean(),
                "fly_edge_bp": FLY.payoff_bp(x).mean() - COST})
mc = pd.DataFrame(res).round(3)
print(mc.to_string(index=False))
mc.to_csv(OUT / "montecarlo_summary.csv", index=False)

# EV vs Jan odds grid
section("5. Fly EV vs January odds after one hike")
jgrid = np.round(np.arange(0, 0.61, 0.1), 2)
ev = {}
for p1, label in ((0.80, "P(1 hike) 80%"), (0.65, "P(1 hike) 65%"), (0.50, "P(1 hike) 50%")):
    p2 = round(1 - 0.10 - p1, 2) if p1 < 0.9 else 0.0
    p0 = round(1 - p1 - p2, 2)
    ev[label] = [FLY.payoff_bp(simulate(Scenario("grid", (p0, p1, p2), (0.6, j, 0.4)), F, basis, n=200_000)).mean() for j in jgrid]
evdf = pd.DataFrame(ev, index=[f"{int(j*100)}%" for j in jgrid]).round(2)
print(evdf.to_string())
evdf.to_csv(OUT / "ev_vs_jan_odds.csv")

# 6. Charts --------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
xs = np.linspace(95.65, 95.95, 241)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, pnl_dollars(FLY.payoff_bp(xs), COST), color=RED, lw=2.5, label=f"Pause fly ({COST}bp)")
ax.plot(xs, pnl_dollars(CONDOR.payoff_bp(xs), condor_calls), color=GREY, lw=2, label=f"Condor ({condor_calls:.2f}bp)")
ax.axvspan(95.79, 95.82, color=RED, alpha=0.12, label="Landing zone 95.79-95.82")
ax.axhline(0, color=INK, lw=0.8)
ax.set(title="P&L at expiry per structure", xlabel="SFRZ6 on 11-Dec", ylabel="$ per unit")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "payoff.png", dpi=200)

x_base = simulate(BASE_CASE, F, basis)
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(95.40, 96.10 + 1e-9, 0.0125)
ax.hist(x_base, bins=bins, density=True, color=RED, alpha=0.75, label="Our base case (Monte Carlo)")
ax.plot(bins, norm.pdf(bins, F, sd_bp / 100), color=GREY, lw=2, label=f"Market-implied normal ({sd_bp:.0f}bp s.d.)")
ax.axvspan(95.75, 95.875, color=INK, alpha=0.06, label="Fly strikes 95.75-95.875")
ax.set(title="SFRZ6 on 11-Dec: market-implied vs our base case", xlabel="SFRZ6", ylabel="density")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "distribution.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4))
for (label, vals), col in zip(evdf.items(), (RED, "#E8A09D", GREY)):
    ax.plot(jgrid * 100, vals, lw=2, color=col, label=label)
ax.axhline(COST, color=INK, lw=1.5, ls="--", label=f"Entry cost {COST}bp")
ax.set(title="Fly expected payoff vs January odds priced on 11-Dec", xlabel="Jan hike odds (%)", ylabel="bp")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "ev_vs_jan_odds.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wirp.meeting.str[:7], wirp.cumulative_hikes, color=RED, marker="o", label="Market (WIRP, 1-Oct)")
ax.plot(wirp.meeting.str[:7], [0 if m < "2026-12" else 1 for m in wirp.meeting], color=INK, marker="o", label="Fed SEP median")
ax.set(title="Cumulative hikes: market vs Fed SEP", ylabel="hikes"); ax.tick_params(axis="x", rotation=45)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "market_vs_sep.png", dpi=200)
print(f"\nCharts and tables written to {OUT}")
