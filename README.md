# Optimal Execution Model: Market Microstructure & Almgren-Chriss

Python framework that compares **TWAP**, **VWAP** and an **Almgren-Chriss** optimal liquidation trajectory for a large equity parent order (default: 100,000 AAPL shares over one trading day, 100 slices), calibrated on real 1-minute intraday data from Polygon.io. It produces a PDF report.

## What it does
- Downloads 1-minute bars from Polygon (with pagination) and caches them locally; keeps the regular session only (09:30-16:00 ET).
- Computes intraday profiles: volume, volatility and a range-based **spread proxy**.
- Estimates a pre-trade impact benchmark with the square-root model: `MI = Y * sigma_daily * sqrt(Q/ADV)`.
- Computes the Almgren-Chriss trajectory `x(t) = X * sinh(k(T-t)) / sinh(kT)` and compares it with TWAP and VWAP.
- Plots the efficient frontier (expected cost vs execution risk for varying lambda).
- Exports a PDF report (`OptimalExecution_Model_PoglianaV6.pdf`; the version generated on real data is included in this repository).

## Run
```bash
pip install -r requirements.txt
```
Create a file named `.env` in the same folder as the script (use `.env.example` as a template) and add your Polygon API key, then:
```bash
python optimal_execution.py
```
Parameters (ticker, dates, order size, slices, impact assumptions) are at the top of the script.

## Assumptions and limitations
- Costs are **model-implied**, to compare schedules on a consistent basis; they are not realised transaction costs (no fills, commissions or fees).
- The impact coefficient `Y = 0.1` is an assumption, not calibrated on fills. The temporary-impact coefficient used by Almgren-Chriss is derived from the same square-root benchmark.
- The spread series is a heuristic range-based proxy, not quoted NBBO bid-ask spreads.
- Lambda is not estimated from a utility function: it is set so that the first slice is 1.8x the TWAP slice.
- Natural extensions: stochastic liquidity, adaptive participation, order-book signals, limit-order execution, real TCA data.

## Author
Gianluca Pogliana
