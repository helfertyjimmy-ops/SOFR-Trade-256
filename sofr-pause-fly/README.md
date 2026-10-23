# SFRZ6 "Pause Fly": Pricing, Monte Carlo and P&L

Code behind our Fall 2026 Schonfeld Discretionary Macro & Fixed Income trade pitch (Duke):
**buy the SFRZ6 95.75 / 95.8125 / 95.875 call butterfly** (Dec-26 3M SOFR options, expiry 11-Dec-2026).

**Thesis.** The market has the *number* of Fed hikes by December right (about one, in line with the Fed's 4.1% SEP median) but
prices a January hike the Fed's own 2027 projection rules out. If the Fed delivers one hike and signals a pause, January odds
collapse and SFRZ6 lands at roughly 95.79–95.82, where the fly peaks.

> Student research project. Not investment advice. All market data are snapshots (dates in file names); refresh before use.

## What the code does

| Module | Purpose |
|---|---|
| `src/sofr_fly/contract.py` | SFRZ6 mechanics: 91-day reference quarter (16-Dec-26 → 17-Mar-27, ACT/360, daily compounding), FOMC effective dates, January weight (48/91), and the map from *(hikes by Dec, Jan odds, basis)* → SFRZ6 price on 11-Dec |
| `src/sofr_fly/bachelier.py` | Normal (Bachelier) pricing on futures prices: calls/puts, implied vol, straddle vol, parity-implied forward, flat-vol butterfly |
| `src/sofr_fly/montecarlo.py` | Scenario Monte Carlo: hike count ~ {0,1,2}, Jan odds ~ Beta conditional on the count, SOFR-basis noise, small tail; basis calibration to the observed forward; WIRP bounds on P(exactly one hike) |
| `src/sofr_fly/payoff.py` | Fly and condor payoffs, $ P&L ($25/bp/contract), breakevens, breakeven probability |
| `scripts/run_all.py` | Reproduces every number and chart in the deck → `outputs/` |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py      # tables + charts into outputs/
pytest -q                      # 15 unit tests
```
In VS Code: open the folder, select the `.venv` interpreter, then use **Run and Debug → "Run full analysis"** or the Testing panel.

## Key results (14-Oct-2026 chain, 1-Oct-2026 WIRP)

| Item | Value |
|---|---|
| Parity-implied SFRZ6 forward | 95.696 |
| Dec ATM normal vol / s.d. to expiry | 53.4bp/yr / 21.3bp |
| Pause fly mid (calls / puts) | 1.25bp / 1.38bp ($31–34) |
| Flat-vol (bell-curve) fly value | 0.63bp |
| Breakevens at 1.31bp | 95.763 / 95.862 |
| WIRP-consistent P(exactly one hike by Dec) | 48%–81% |

Landing zones (SFRZ6 on 11-Dec):

| | Jan 0% | Jan 25% | Jan 53% | Jan 100% |
|---|---|---|---|---|
| 0 hikes | 96.07 | 96.03 | 96.00 | 95.93 |
| **1 hike** | **95.81** | **95.78** | **95.74** | **95.68** |
| 2 hikes | 95.56 | 95.53 | 95.49 | 95.43 |

Monte Carlo (fly cost 1.31bp):

| Scenario (forecast) | P(0/1/2) | Fly EV | Edge vs cost |
|---|---|---|---|
| Market-like | 11/60/30% | 0.73bp | −0.58bp |
| Base case: one hike, then pause | 10/80/10% | 3.17bp | +1.86bp |
| Central: data-dependent | 10/65/25% | 1.90bp | +0.59bp |
| Hawkish | 15/50/35% | 1.47bp | +0.16bp |

**Reading:** the edge is the January call. Expected payoff clears the entry cost only when January odds priced on 11-Dec fall
below roughly 35–45% (`outputs/ev_vs_jan_odds.csv`).

## Assumptions and limitations
- Hikes are 25bp steps; outcomes outside {0,1,2} sit in a 3% uniform ±50bp tail.
- EFFR 3.88% (WIRP); the SOFR–fed funds/year-end basis (~3.5bp) is calibrated so the market-like scenario reproduces the forward.
- Jan odds conditional on the hike count are **our forecasts** (Beta-distributed). WIRP pins only the mean, not the joint distribution.
- Options data are indicative mids (no package quote); legging costs more. Fees not modeled beyond a 0.25bp assumption.
- Calibration mixes the 14-Oct chain with 1-Oct WIRP; refresh both together before relying on the numbers.

## Data sources
- thinkorswim /SR3Z26 option chain, 14-Oct-2026 (`data/sfrz6_dec_options_2026-10-14.csv`)
- Bloomberg WIRP, pricing date 1-Oct-2026 (`data/wirp_2026-10-01.csv`)
- Federal Reserve Summary of Economic Projections, Sep-2026 (`data/fed_sep_2026-09.csv`)
- CME Three-Month SOFR futures and options specifications

Team: Jimmy Helferty, Jay Patel, Matthew Mashaal, Avi Karamchandani.
