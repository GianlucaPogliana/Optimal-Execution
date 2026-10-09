"""
Optimal Execution Model — Almgren-Chriss with Market Microstructure Analysis
Combining optimal liquidation trajectory with intraday volume, a range-based
spread proxy and price impact analysis using real market data (Polygon API)

Author:  Gianluca Pogliana
Contact: poglianagianluca@gmail.com | +39 340 327 6133

Description:
    This model implements the Almgren-Chriss (2001) framework for optimal
    execution of large orders, combined with a market microstructure analysis
    of intraday volume profile, spread-proxy dynamics and price impact.
    Real intraday data is downloaded from Polygon API.

Dependencies:
    pip install numpy pandas matplotlib scipy reportlab requests python-dotenv fredapi

API Keys (.env file):
    POLYGON_API_KEY = your_polygon_key
    FRED_API_KEY    = your_fred_key
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.optimize import minimize
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, HRFlowable,
                                 KeepTogether, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from dotenv import load_dotenv
import os, io, time, requests, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Reads POLYGON_API_KEY from a .env file placed next to this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FRED_API_KEY    = os.getenv("FRED_API_KEY")

# Instrument
TICKER     = "AAPL"          # change to any equity ticker on Polygon
MARKET     = "stocks"
START_DATE = "2024-10-14"   # Polygon plan used here gives ~2 years of 1-minute history
END_DATE   = "2026-10-08"

# Order parameters
ORDER_SIZE   = 100000    # shares to liquidate
T_HORIZON    = 1.0          # trading horizon in days (1 = full day)
N_SLICES     = 100           # number of execution slices
VWAP_SMOOTHING_WINDOW = 3
MAX_VWAP_SLICE_MULTIPLE = 2.0

# Model assumptions (documented in the PDF)
IMPACT_COEFF         = 0.1   # Y in MI = Y * sigma_daily * sqrt(Q/ADV). Literature range ~0.1-1.0.
                             # Assumption, NOT calibrated on fills: replace with own TCA data if available.
GAMMA_TO_ETA         = 0.1   # permanent / temporary impact coefficient ratio used in the AC cost function
PERM_TO_TEMP         = 0.5   # permanent / temporary per-share impact in the pre-trade benchmark (Section 4)
FIRST_SLICE_MULTIPLE = 1.8   # lambda is set so that AC trades 1.8x the TWAP slice in the first interval
SPREAD_PROXY_FACTOR  = 0.3   # heuristic scale applied to the 1-min high-low range (see compute_intraday_profile)

OUTPUT_PATH = "OptimalExecution_Model_PoglianaV6.pdf"

DATA_PATH = (
    f"data/{TICKER}_"
    f"{START_DATE}_"
    f"{END_DATE}.csv"
)

# Colors
DARK_C = '#1A3A5C'
MID_C  = '#2E74B5'
GRN_C  = '#1D9E75'
RED_C  = '#E74C3C'
ORG_C  = '#E67E22'
BG_C   = '#F8F9FB'


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD — POLYGON API
# ─────────────────────────────────────────────────────────────────────────────
def fetch_polygon_aggs(ticker: str, start: str, end: str,
                       multiplier: int = 1, timespan: str = "minute",
                       api_key: str = None) -> pd.DataFrame:
    """
    Download aggregated OHLCV bars from Polygon.io, following `next_url`
    pagination (a single request returns at most 50,000 bars, far fewer
    than 1.5 years of 1-minute data).

    Returns:
        DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume, vwap
    """
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
           f"{multiplier}/{timespan}/{start}/{end}")
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

    rows, pages = [], 0
    while url:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:           # rate limit (free tier): wait and retry
            time.sleep(15)
            continue
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("results") or [])
        pages += 1
        url = data.get("next_url")
        params = {"apiKey": api_key}          # next_url does not embed the key

    if not rows:
        print(f"  WARNING: No data returned for {ticker}.")
        return None

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume", "vw": "vwap"})
    df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]].copy()
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    print(f"  Downloaded {len(df)} bars for {ticker} ({start} -> {end}), {pages} page(s)")
    return df


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to New York time and keep the regular session only (09:30-16:00 ET)."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    minute = df.index.hour * 60 + df.index.minute
    return df[(minute >= 570) & (minute < 960)]


def load_or_download_data(ticker, start, end, api_key):

    path = DATA_PATH

    # ============================
    # LOAD EXISTING CACHE
    # ============================

    if os.path.exists(path):

        print(f"Loading cached data: {path}")

        df = pd.read_csv(
            path,
            parse_dates=["timestamp"]
        )

        df = df.set_index("timestamp")

        print(
            f"Loaded {len(df)} bars from cache"
        )

        # Make sure the cache really covers the requested window: a cache written by an
        # older, non-paginated download may contain only the first ~50,000 bars.
        idx = pd.to_datetime(df.index, utc=True)
        covered_end = idx.max()
        requested_end = pd.Timestamp(end, tz="UTC")
        if covered_end >= requested_end - pd.Timedelta(days=10):
            return df
        print(f"  WARNING: cache ends on {covered_end.date()} but {end} was requested "
              f"(likely truncated). Ignoring cache and downloading again...")


    # ============================
    # DOWNLOAD FROM POLYGON
    # ============================

    print("Downloading data from Polygon...")


    df = fetch_polygon_aggs(
        ticker,
        start,
        end,
        multiplier=1,
        timespan="minute",
        api_key=api_key
    )


    if df is not None:

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )


        save_df = df.reset_index()

        save_df.to_csv(
            path,
            index=False
        )


        print(
            f"Saved market data cache: {path}"
        )


    return df

def fetch_polygon_quotes(ticker: str, date: str,
                         api_key: str = None) -> pd.DataFrame:
    """
    Download NBBO quotes (bid/ask) for a single day from Polygon.io.
    NOT used by the main pipeline (a full-sample NBBO download is very heavy);
    kept for single-day spot checks of the range-based spread proxy.

    Returns:
        DataFrame with bid_price, ask_price, spread_bps, timestamp
    """
    url = f"https://api.polygon.io/v3/quotes/{ticker}"
    params = {
        "timestamp.gte": f"{date}T09:30:00Z",
        "timestamp.lte": f"{date}T16:00:00Z",
        "limit":         50000,
        "apiKey":        api_key,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        print(f"  WARNING: No quote data for {ticker} on {date}. Simulating.")
        return None

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
    df = df[["timestamp", "bid_price", "ask_price"]].dropna()
    df["mid_price"]  = (df["bid_price"] + df["ask_price"]) / 2
    df["spread_abs"] = df["ask_price"] - df["bid_price"]
    df["spread_bps"] = df["spread_abs"] / df["mid_price"] * 10000
    df = df.set_index("timestamp").sort_index()
    # Resample to 1-minute to match OHLCV bars
    df = df.resample("1min").mean().dropna()
    print(f"  Downloaded {len(df)} quote snapshots for {ticker} on {date}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MARKET MICROSTRUCTURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def compute_intraday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average intraday profiles for volume, spread and price volatility.

    Variables:
        avg_volume      : mean volume per minute, averaged across all days
                          → identifies liquidity windows for optimal execution
        avg_spread_bps  : mean range-based SPREAD PROXY in bps per minute
                          (NOT a quoted bid-ask spread)
                          → rough indicator of transaction cost at each time of day
        avg_volatility  : mean absolute return per minute
                          → measures timing risk at each time of day
        vwap_dev        : mean deviation of price from intraday VWAP
                          → identifies systematic price patterns

    The U-shaped volume profile (high at open/close, low at midday) is a
    well-documented stylised fact of equity microstructure (Admati & Pfleiderer,
    1988). Execution algorithms exploit this by concentrating trades during
    high-volume periods to minimise market impact.
    """
    if 'spread_bps' not in df.columns:
        # Heuristic proxy from the 1-minute high-low range, scaled by SPREAD_PROXY_FACTOR.
        # It is NOT a quoted bid-ask spread: it also embeds volatility, so it is mechanically
        # wider at the open. Use NBBO quotes (fetch_polygon_quotes) for a true spread.
        df['spread_bps'] = ((df['high'] - df['low']) / df['close'] * 10000
                            * SPREAD_PROXY_FACTOR)

    # Convert Polygon UTC timestamps to New York exchange time
    df.index = df.index.tz_convert("America/New_York")
    print(
        "Market timezone:",
        df.index.tz
    )
    df['minute_of_day'] = (
            df.index.hour * 60
            +
            df.index.minute
    )

    #####################

    # Intraday returns only: avoid the overnight gap between the last bar of a day
    # and the first bar of the next one, which would inflate open-time volatility.
    df['ret'] = df.groupby(df.index.date)['close'].pct_change().abs()

    profile = df.groupby('minute_of_day').agg(
        avg_volume    = ('volume',     'mean'),
        avg_spread_bps= ('spread_bps', 'mean'),
        avg_volatility= ('ret',        'mean'),
    ).reset_index()

    # Filter to market hours (9:30–16:00 = minutes 570–960)
    profile = profile[
        (profile['minute_of_day'] >= 570) &
        (profile['minute_of_day'] < 960)
    ].copy()
    profile['time'] = pd.to_datetime(
        profile['minute_of_day'].apply(
            lambda m: f"{m//60:02d}:{m%60:02d}"), format='%H:%M')
    return profile


def compute_price_impact(df: pd.DataFrame, order_size: float) -> dict:
    """
    Pre-trade benchmark based on the square-root market impact model:

        MI(Q) = Y * sigma_daily * sqrt(Q / ADV)        (fraction of price)

    where Y = IMPACT_COEFF (an assumption, typical range 0.1-1.0 in the literature),
    sigma_daily is the DAILY close-to-close volatility and ADV the average daily
    volume computed over trading days only (weekends/holidays excluded).

    df must be indexed in New York time (see regular_session).
    """
    day = df.index.date
    daily_close = df['close'].groupby(day).last()
    daily_volume = df['volume'].groupby(day).sum()

    adv = daily_volume.mean()
    sigma_daily = daily_close.pct_change().std()
    sigma = sigma_daily * np.sqrt(252)          # annualised, for reporting only
    price = daily_close.iloc[-1]

    impact_fraction = IMPACT_COEFF * sigma_daily * np.sqrt(order_size / adv)
    temp_impact = impact_fraction * price
    perm_impact = PERM_TO_TEMP * temp_impact

    return dict(sigma=sigma, sigma_daily=sigma_daily, eta=IMPACT_COEFF, adv=adv,
                temp_impact=temp_impact, perm_impact=perm_impact, price=price,
                n_days=len(daily_close),
                first_date=daily_close.index[0], last_date=daily_close.index[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 3. ALMGREN-CHRISS OPTIMAL EXECUTION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def almgren_chriss(X: float, T: float, N: int,
                   sigma: float, eta: float, gamma: float,
                   lam: float) -> dict:
    """
    Almgren-Chriss (2001) optimal liquidation trajectory.

    The model solves the problem of liquidating X shares over T days in N slices,
    minimising the expected cost plus a risk penalty:

        min E[Cost] + λ × Var[Cost]

    The optimal trajectory has a closed-form solution:

        x(t) = X × sinh(κ(T−t)) / sinh(κT)

    where:
        κ = √(λσ²/η)   (urgency parameter)

    Variables:
        X     : total shares to liquidate
        T     : time horizon (days)
        N     : number of execution slices
        sigma : DAILY price volatility in USD/share (fractional daily vol x price)
        eta   : temporary market impact coefficient
        gamma : permanent market impact coefficient
        lam   : risk aversion parameter λ
                  λ → 0 : TWAP strategy (linear schedule, ignores risk)
                  λ → ∞ : immediate execution (aggressive, minimises risk)
        kappa : urgency parameter = √(λσ²/η)
        x_t   : shares remaining at each time step (optimal trajectory)
        n_t   : shares traded at each time step
        E_cost: modelled expected execution cost (USD)
        V_cost: variance proxy of execution cost (USD²)
        IS    : implementation shortfall vs arrival price (bps)

    Reference:
        Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio
        transactions. Journal of Risk, 3(2), 5-39.
    """
    tau   = T / N                           # time per slice
    kappa = np.sqrt(lam * sigma**2 / eta)    # urgency parameter



    # Optimal trajectory: shares remaining at each time step
    t_grid = np.linspace(0, T, N + 1)
    if kappa * T < 1e-6:
        # Risk-neutral limit → VWAP (linear schedule)
        x_t = X * (1 - t_grid / T)
    else:
        x_t = X * np.sinh(kappa * (T - t_grid)) / np.sinh(kappa * T)

    # Shares traded at each slice
    n_t = np.diff(x_t)   # negative = selling

    # Expected cost
    E_cost = (
            eta / tau * np.sum(n_t ** 2) +
            gamma / 2 * X ** 2
    )

    # Variance of execution cost
    V_cost = sigma ** 2 * tau * np.sum(x_t[:-1] ** 2)

    # Almgren-Chriss objective function
    objective = E_cost + lam * V_cost

    return dict(
        t_grid=t_grid,
        x_t=x_t,
        n_t=n_t,
        E_cost=E_cost,
        V_cost=V_cost,
        objective=objective,
        kappa=kappa,
        tau=tau
    )

def cap_and_redistribute(weights, cap):
    """
    Cap VWAP slice weights and redistribute excess volume across uncapped slices.

    This prevents the VWAP schedule from becoming unrealistically concentrated
    at the open or close when the historical volume curve is very U-shaped.
    """

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    capped = np.minimum(weights, cap)

    for _ in range(100):
        excess = 1.0 - capped.sum()

        if abs(excess) < 1e-10:
            break

        room = cap - capped
        eligible = room > 1e-10

        if not eligible.any():
            break

        redistribution_base = weights * eligible
        redistribution_base = redistribution_base / redistribution_base.sum()

        add = np.minimum(
            redistribution_base * excess,
            room
        )

        capped += add

    return capped / capped.sum()
def compare_strategies(X: float, T: float, N: int,
                        sigma: float, eta: float, gamma: float,
                        profile: pd.DataFrame,
                        risk_aversion: float) -> dict:
    """
    Compare three execution strategies:
        1. TWAP  : Time-Weighted Average Price (uniform schedule)
        2. VWAP  : Volume-Weighted Average Price (volume-proportional)
        3. AC    : Almgren-Chriss optimal (minimises E[Cost] + λ×Var[Cost])

    Returns costs and trajectories for all three strategies.
    """
    tau    = T / N
    t_grid = np.linspace(0, T, N + 1)
    lam = risk_aversion

    # 1. TWAP — uniform liquidation
    x_twap = X * (1 - t_grid / T)
    n_twap = np.diff(x_twap)
    cost_twap = eta / tau * np.sum(n_twap**2) + gamma / 2 * X**2

    # 2. VWAP — dynamic volume-weighted execution
    #
    # The execution schedule follows the real intraday volume profile.
    # The temporary impact coefficient eta is also dynamic:
    # high-volume slices have lower eta, low-volume slices have higher eta.

    real_volume = profile["avg_volume"].values
    real_spread = profile["avg_spread_bps"].values

    volume_bins = np.array_split(real_volume, N)
    spread_bins = np.array_split(real_spread, N)

    volume_per_slice = np.array([x.sum() for x in volume_bins])
    spread_per_slice = np.array([x.mean() for x in spread_bins])

    # Raw historical VWAP profile
    raw_volume_profile = volume_per_slice / volume_per_slice.sum()

    # Smooth the volume profile to avoid overreacting to opening/closing spikes
    smoothed_volume_profile = (
        pd.Series(raw_volume_profile)
        .rolling(
            window=VWAP_SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
        .values
    )

    smoothed_volume_profile = smoothed_volume_profile / smoothed_volume_profile.sum()

    # Cap the maximum VWAP slice relative to TWAP
    twap_slice_weight = 1.0 / N
    max_vwap_slice_weight = MAX_VWAP_SLICE_MULTIPLE * twap_slice_weight

    volume_profile = cap_and_redistribute(
        smoothed_volume_profile,
        cap=max_vwap_slice_weight
    )

    n_vwap = X * volume_profile

    x_vwap = np.zeros(N + 1)
    x_vwap[0] = X

    for i in range(N):
        x_vwap[i + 1] = x_vwap[i] - n_vwap[i]

    eta_vwap = np.full(N, eta)  # stesso coefficiente per tutti i confronti

    cost_vwap = (
            eta / tau * np.sum(n_vwap ** 2)
            + gamma / 2 * X ** 2
    )


    # 3. Almgren-Chriss optimal
    ac = almgren_chriss(X, T, N, sigma, eta, gamma, lam)
    objective_ac = ac["objective"]
    return dict(
        t_grid=t_grid,
        x_twap=x_twap, cost_twap=cost_twap, n_twap=n_twap,
        x_vwap=x_vwap, cost_vwap=cost_vwap, n_vwap=n_vwap,
        x_ac=ac['x_t'], cost_ac=ac['E_cost'], n_ac=ac['n_t'],
        kappa=ac['kappa'],
        eta_vwap=eta_vwap,
        volume_profile=volume_profile,
        raw_volume_profile=raw_volume_profile,
        max_vwap_slice_weight=max_vwap_slice_weight,
        spread_profile=spread_per_slice,
        risk_ac=ac["V_cost"],
        objective_ac=ac["objective"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_microstructure(profile: pd.DataFrame) -> io.BytesIO:
    """Intraday volume, range-based spread proxy and volatility profiles."""
    fig = plt.figure(figsize=(13, 6.3))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for ax in (ax1, ax2, ax3): ax.set_facecolor(BG_C)

    times = profile['time']

    # Volume profile
    # Volume profile
    # Use a visual cap to prevent the opening auction / first-minute spike
    # from compressing the rest of the intraday volume curve.
    volume_raw = profile['avg_volume'].values

    volume_cap = np.percentile(volume_raw, 98)
    volume_plot = np.minimum(volume_raw, volume_cap)

    ax1.bar(
        times,
        volume_plot,
        color=MID_C,
        alpha=0.75,
        width=0.0005
    )

    ax1.set_ylim(0, volume_cap * 1.15)

    ax1.set_ylabel('Avg Volume\n(shares/min, capped)', fontsize=9)

    ax1.set_title(
        'Market Microstructure Analysis — Intraday Profiles',
        fontsize=11,
        fontweight='bold',
        color=DARK_C,
        pad=8
    )

    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(times.iloc[0], times.iloc[-1])

    # Add a small note to be transparent about the visual cap
    ax1.text(
        0.99,
        0.88,
        'Volume axis capped at 98th percentile for readability',
        fontsize=6.5,
        transform=ax1.transAxes,
        ha='right',
        va='top',
        color='#666666'
    )

    # Spread proxy (range-based)
    ax2.plot(times, profile['avg_spread_bps'], color=ORG_C, lw=1.5)
    ax2.fill_between(times, profile['avg_spread_bps'],
                     profile['avg_spread_bps'].min(),
                     alpha=0.15, color=ORG_C)
    ax2.set_ylabel('Spread proxy (bps)', fontsize=9)
    ax2.grid(True, alpha=0.3); ax2.tick_params(labelsize=8)

    # Volatility
    ax3.plot(times, profile['avg_volatility'] * 10000, color=RED_C, lw=1.5)
    ax3.fill_between(times, profile['avg_volatility'] * 10000, 0,
                     alpha=0.15, color=RED_C)
    ax3.set_ylabel('Price Volatility (bps/min)', fontsize=9)
    ax3.grid(True, alpha=0.3); ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=155, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close()
    return buf


def make_fig_ac_trajectory(strategies: dict, order_size: float) -> io.BytesIO:
    """Almgren-Chriss optimal trajectory vs TWAP vs VWAP."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5))
    fig.patch.set_facecolor('white')
    for ax in (ax1, ax2): ax.set_facecolor(BG_C)

    t = strategies['t_grid']

    # Remaining inventory trajectories
    ax1.plot(t, strategies['x_twap'] / order_size * 100,
             color=ORG_C,  lw=2.0, ls='--', label='TWAP (uniform)')
    ax1.plot(t, strategies['x_vwap'] / order_size * 100,
             color=MID_C,  lw=2.0, ls=':',  label='VWAP (smoothed/capped)')
    ax1.plot(t, strategies['x_ac']   / order_size * 100,
             color=GRN_C,  lw=2.5,           label='Almgren-Chriss (optimal)')
    ax1.fill_between(t, strategies['x_ac'] / order_size * 100, 0,
                     alpha=0.10, color=GRN_C)
    ax1.set_ylabel('Remaining Inventory (%)', fontsize=9)
    ax1.set_title('Top: Remaining Inventory (%)',
                  fontsize=11, fontweight='bold', color=DARK_C, pad=8)
    ax2.set_title('Bottom: Execution Schedule (shares per slice)',
                  fontsize=10, fontweight='bold', color=DARK_C, pad=8)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3); ax1.tick_params(labelsize=8)
    ax1.set_xlabel('Time (days)', fontsize=9)

    # Execution schedule (shares per slice)
    width = (t[1] - t[0]) * 0.25
    ax2.bar(t[:-1] - width, np.abs(strategies['n_twap']) / order_size * 100,
            width=width, color=ORG_C, alpha=0.75, label='TWAP')

    ax2.bar(t[:-1], np.abs(strategies['n_vwap']) / order_size * 100,
            width=width, color=MID_C, alpha=0.75, label='VWAP')

    ax2.bar(t[:-1] + width, np.abs(strategies['n_ac']) / order_size * 100,
            width=width, color=GRN_C, alpha=0.85, label='AC Optimal')
    ax2.set_ylabel('Execution per Slice (%)', fontsize=9)
    ax2.set_xlabel('Time (days)', fontsize=9)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=155, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close()
    return buf


def make_fig_cost_frontier(X: float, T: float, N: int,
                            sigma: float, eta: float, gamma: float,
                            risk_aversion: float) -> io.BytesIO:
    """Efficient frontier: expected cost vs variance for different λ values."""
    lambdas = np.logspace(np.log10(risk_aversion) - 3, np.log10(risk_aversion) + 3, 40)
    E_costs, V_costs = [], []

    for lam in lambdas:
        ac = almgren_chriss(X, T, N, sigma, eta, gamma, lam)
        E_costs.append(ac['E_cost'])
        V_costs.append(np.sqrt(ac['V_cost']))  # std dev

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(BG_C)

    sc = ax.scatter(V_costs, E_costs,
                    c=np.log10(lambdas), cmap='RdYlGn_r',
                    s=40, alpha=0.8, zorder=3)
    ax.plot(V_costs, E_costs, color=MID_C, lw=1.5, alpha=0.5)

    # Mark key strategies
    ac_base = almgren_chriss(X, T, N, sigma, eta, gamma, risk_aversion)
    ax.scatter(np.sqrt(ac_base['V_cost']), ac_base['E_cost'],
               s=150, color=GRN_C, zorder=5, label=f'Selected λ = {risk_aversion:.3g}')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(λ) — Risk Aversion', fontsize=8)

    ax.set_xlabel('Execution Risk — Std Dev of Cost (USD)', fontsize=9)
    ax.set_ylabel('Expected Execution Cost (USD)', fontsize=9)
    ax.set_title(
        f'Efficient Frontier — Expected Cost vs Execution Risk (λ = {risk_aversion:.3g})',
        fontsize=10,
        fontweight='bold',
        color=DARK_C,
        pad=8
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=155, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close()
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD PDF — v3: compact layout + result interpretation
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(buf_micro, buf_traj, buf_frontier,
              strategies, impact, profile, eta, gamma, sigma,
              risk_aversion, first_slice_pct, output_path):

    DARK = colors.HexColor('#1A3A5C')
    MID  = colors.HexColor('#2E74B5')
    LBG  = colors.HexColor('#F0F4F8')
    GR   = colors.HexColor('#1D9E75')
    RD   = colors.HexColor('#E74C3C')
    GY   = colors.HexColor('#444444')

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    T   = ps('T',  fontName='Helvetica-Bold',    fontSize=15, leading=19, textColor=DARK, spaceAfter=3)
    Su  = ps('Su', fontName='Helvetica',         fontSize=9,  leading=12, textColor=colors.HexColor('#666'), spaceAfter=4)
    Se  = ps('Se', fontName='Helvetica-Bold',    fontSize=11, textColor=DARK, spaceBefore=14, spaceAfter=5)
    Se2 = ps('S2', fontName='Helvetica-Bold',    fontSize=10, textColor=MID,  spaceBefore=10, spaceAfter=4)
    Bo  = ps('Bo', fontName='Helvetica',         fontSize=9,  textColor=GY,   spaceAfter=5, leading=14, alignment=4)
    Bul = ps('Bu', fontName='Helvetica',         fontSize=9,  textColor=GY,   spaceAfter=3, leading=14, leftIndent=14)
    Co  = ps('Co', fontName='Courier',           fontSize=8,  textColor=DARK, backColor=LBG, spaceAfter=2, leading=12, leftIndent=10)
    Ca  = ps('Ca', fontName='Helvetica-Oblique', fontSize=8,  textColor=colors.HexColor('#888'), alignment=TA_CENTER, spaceAfter=8)
    Fo  = ps('Fo', fontName='Helvetica',         fontSize=7.5,textColor=colors.HexColor('#999'), alignment=TA_CENTER)

    Cell = ps(
        'Cell',
        fontName='Helvetica',
        fontSize=7.4,
        leading=9,
        textColor=GY
    )

    CellB = ps(
        'CellB',
        fontName='Helvetica-Bold',
        fontSize=7.4,
        leading=9,
        textColor=DARK
    )

    CellH = ps(
        'CellH',
        fontName='Helvetica-Bold',
        fontSize=7.4,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    def C(text, style=Cell):
        return Paragraph(str(text), style)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Optimal Execution Model", T))
    story.append(Paragraph(
        "Market Microstructure Analysis & Optimal Liquidation Trajectory  ·  "
        "Gianluca Pogliana - +39 3403276133 - poglianagianluca@gmail.com", Su))
    story.append(HRFlowable(width="100%", thickness=2, color=MID, spaceAfter=8))

    # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", Se))

    order_value = ORDER_SIZE * impact['price']
    ac_vs_twap = (strategies['cost_ac'] / strategies['cost_twap'] - 1) * 100

    story.append(Paragraph(
        f"This project develops a quantitative execution framework for a large "
        f"<b>{TICKER}</b> liquidation order. The objective is to compare standard benchmark "
        f"execution schedules, such as TWAP and VWAP, with an <b>Almgren-Chriss optimal "
        f"trajectory</b> calibrated on real intraday market data. The model is designed to "
        f"answer a practical execution-desk question: how should a large parent order be "
        f"split over the trading day in order to balance expected transaction costs against "
        f"the risk of adverse price movements?",
        Bo
    ))

    summary_data = [
        [C("Project Snapshot", CellH), ""],

        [C("Instrument", CellB),
         C(f"{TICKER} equity | 1-minute bars, regular session (09:30-16:00 ET) | "
           f"{impact['first_date']} – {impact['last_date']} ({impact['n_days']} trading days)")],

        [C("Parent Order", CellB),
         C(f"{ORDER_SIZE:,} shares | Notional ~USD {order_value:,.0f}")],

        [C("Execution Setup", CellB),
         C(f"{T_HORIZON:.1f} trading day | {N_SLICES} execution slices | TWAP, VWAP and Almgren-Chriss comparison")],


    ]

    summary_table = Table(
        summary_data,
        colWidths=[4.0 * cm, 12.5 * cm]
    )

    summary_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (-1, 0)),
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#E8F0F8')),
        ('ROWBACKGROUNDS', (1, 1), (1, -1), [colors.white, LBG]),

        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(Spacer(1, 6))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # ── 2. MARKET MICROSTRUCTURE ──────────────────────────────────────────────

    # ── 2. MARKET MICROSTRUCTURE ──────────────────────────────────────────────
    story.append(Paragraph("2. Market Microstructure Analysis", Se))

    # Main chart first, so it fits on page 1
    story.append(Paragraph("2.1 Intraday Profiles — AAPL (Polygon API)", Se2))

    story.append(Image(
        buf_micro,
        width=16.0 * cm,
        height=8 * cm
    ))


    # Microstructure statistics
    peak_vol_time = profile.loc[profile['avg_volume'].idxmax(), 'time'].strftime('%H:%M')
    min_spread_time = profile.loc[profile['avg_spread_bps'].idxmin(), 'time'].strftime('%H:%M')
    avg_spread = profile['avg_spread_bps'].mean()
    max_spread = profile['avg_spread_bps'].max()

    ms_data = [
        [C("Metric", CellH), C("Value", CellH), C("Metric", CellH), C("Value", CellH)],

        [C("Peak Volume Time", CellB),
         C(peak_vol_time),
         C("Avg Spread Proxy", CellB),
         C(f"{avg_spread:.1f} bps")],

        [C("Min Spread Time", CellB),
         C(min_spread_time),
         C("Max Spread Proxy", CellB),
         C(f"{max_spread:.1f} bps")],

        [C("ADV", CellB),
         C(f"{impact['adv']:,.0f} shares"),
         C("Last Price", CellB),
         C(f"USD {impact['price']:.2f}")],
    ]

    mst = Table(ms_data, colWidths=[4.2 * cm, 3.3 * cm, 4.2 * cm, 4.8 * cm])
    mst.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LBG]),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCC')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))

    story.append(mst)
    story.append(Spacer(1, 6))

    # Interpretation after the chart
    story.append(Paragraph("2.2 Interpretation & Execution Relevance", Se2))

    story.append(Paragraph(
        f"Peak volume occurs at <b>{peak_vol_time}</b> (market open), confirming the U-shaped "
        f"intraday volume pattern. The range-based spread proxy (not a quoted bid-ask spread) reaches a maximum of "
        f"<b>{max_spread:.1f} bps</b> at the open and compresses to a daily minimum at "
        f"<b>{min_spread_time}</b>, with an intraday average of <b>{avg_spread:.1f} bps</b>. "
        f"These findings suggest that the most favourable execution window for minimising direct "
        f"transaction costs lies between <b>10:30 and 14:00</b>, when spreads are lower and "
        f"intraday volatility is more stable. "
        f"Volume is highest at open and close and lowest at midday. "
        f"High-volume windows offer more liquidity per unit of time, but they coincide with wider "
        f"spreads and higher volatility, so the trade-off between impact, spread and timing risk is "
        f"what the model in Section 3 quantifies. "
        f"The spread is widest at open, when uncertainty and adverse selection "
        f"risk are highest, and tightens during the day as liquidity improves. "
        f"Intraday volatility is highest at open and close, increasing timing "
        f"risk for patient execution strategies.",
        Bo
    ))

    # ── 3. ALMGREN-CHRISS MODEL ───────────────────────────────────────────────

    story.append(Paragraph("3. Almgren-Chriss Optimal Execution Model", Se))
    story.append(Paragraph("3.1 Problem Statement & Closed-Form Solution", Se2))
    story.append(Paragraph(
        "Liquidating a large order exposes the trader to a fundamental trade-off: "
        "<b>slow execution</b> reduces market impact but increases timing risk (price drift); "
        "<b>fast execution</b> reduces timing risk but amplifies impact costs. "
        "Almgren & Chriss (2001) solve this formally by minimising:", Bo))
    story.append(Paragraph("<b>min  E[Cost] + λ × Var[Cost]</b>", Se2))
    story.append(Paragraph(
        "The closed-form optimal trajectory is <b>x(t) = X × sinh(κ(T−t)) / sinh(κT)</b> "
        "where <b>κ = √(λσ²/η)</b> is the urgency parameter. "
        f"For λ = {risk_aversion:.3g} (set so that the first slice is {FIRST_SLICE_MULTIPLE:.1f}× the TWAP slice), κ = {strategies['kappa']:.4f}: "
        f"the strategy is moderately front-loaded, executing {first_slice_pct:.1f}% in the first "
        f"slice vs {100/N_SLICES:.1f}% for a uniform TWAP.", Bo))

    # Variable table — compact
    vars_data = [
        ["Var", "Value", "Description"],
        ["X",   f"{ORDER_SIZE:,} shares",      "Total order size"],
        ["T",   f"{T_HORIZON} day",             "Execution horizon"],
        ["N",   f"{N_SLICES} slices",           "Execution intervals"],
        ["λ",   f"{risk_aversion:.3g}",         f"Risk aversion (set so first slice = {FIRST_SLICE_MULTIPLE:.1f}x TWAP slice)"],
        ["κ",   f"{strategies['kappa']:.4f}",   "Urgency parameter √(λσ²/η)"],
        ["σ",   f"{sigma*100:.3f}%/day",        "Daily price volatility"],
        ["η",   f"{eta:.2e} USD/share²",        "Temporary impact coefficient"],
        ["γ",   f"{gamma:.2e} USD/share²",      "Permanent impact coefficient"],
        ["ADV", f"{impact['adv']:,.0f}",        "Average daily volume (shares)"],
    ]
    vt = Table(vars_data, colWidths=[1.5*cm, 4*cm, 11*cm])
    vt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), DARK),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (1,1), (-1,-1),'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 8.8),
        ('TEXTCOLOR',     (0,1), (0,-1), DARK),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, LBG]),
        ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor('#CCC')),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(vt); story.append(Spacer(1, 6))

    story.append(Paragraph("3.2 Execution Trajectories", Se2))
    story.append(Image(buf_traj, width=16.5*cm, height=7.5*cm))
    story.append(Paragraph(
        "Fig. 2 — Top: remaining inventory (%) for TWAP, VWAP and AC Optimal over the "
        f"trading horizon. Bottom: shares executed per slice (%). "
        f"AC front-loads execution vs TWAP (first slice: {first_slice_pct:.1f}% vs "
        f"{100/N_SLICES:.1f}%) to reduce timing risk; VWAP concentrates at open/close "
        "following the volume profile.", Ca))

    # Trajectory interpretation
    twap_cost = strategies['cost_twap']
    vwap_cost = strategies['cost_vwap']
    ac_cost   = strategies['cost_ac']
    story.append(Paragraph(
        f"The AC Optimal strategy has a <b>modelled expected execution cost</b> of "
        f"<b>USD {ac_cost:,.0f}</b> "
        f"({(ac_cost / twap_cost - 1) * 100:+.1f}% vs TWAP), achieving a lower-risk trajectory "
        f"by front-loading execution while containing market impact. "
        f"VWAP has a higher modelled expected cost (<b>USD {vwap_cost:,.0f}</b>, "
        f"{(vwap_cost / twap_cost - 1) * 100:+.1f}% vs TWAP) because its smoothed volume-based schedule "
        f"still allocates larger slices around the open and close. Since the model penalises trading "
        f"intensity through a squared trade-size term, more concentrated execution increases expected "
        f"temporary impact relative to a uniform TWAP schedule. In practice, VWAP remains widely used "
        f"because it minimises <i>deviation from the market average price</i>, not necessarily because "
        f"it minimises model-implied implementation shortfall.",
        Bo
    ))
    story.append(Paragraph(
        "The cost figures reported below are <b>model-implied expected execution costs</b>, "
        "not realised trading costs. They are derived from the Almgren-Chriss impact function "
        "and should be interpreted as an implementation-shortfall proxy under the model assumptions. "
        "They do not include actual broker commissions, exchange fees, realised bid-ask spread paid, "
        "taxes or slippage from real fills. Their main purpose is to compare execution trajectories "
        "on a consistent basis.",
        Bo
    ))

    # Strategy comparison table
    strat_data = [
        [
            C("Strategy", CellH),
            C("Modelled Cost", CellH),
            C("vs TWAP", CellH),
            C("Execution Logic", CellH),
            C("Best Use Case", CellH)
        ],

        [
            C("TWAP", CellB),
            C(f"USD {twap_cost:,.0f}<br/>({twap_cost / order_value * 10000:.1f} bps)"),
            C("—"),
            C("Uniform execution across all slices"),
            C("Stable, liquid and low-volatility markets")
        ],

        [
            C("VWAP", CellB),
            C(f"USD {vwap_cost:,.0f}<br/>({vwap_cost / order_value * 10000:.1f} bps)"),
            C(f"{(vwap_cost / twap_cost - 1) * 100:+.1f}%"),
            C("Execution follows the observed volume profile"),
            C("Benchmark-sensitive execution versus market VWAP")
        ],

        [
            C("AC Optimal", CellB),
            C(f"USD {ac_cost:,.0f}<br/>({ac_cost / order_value * 10000:.1f} bps)", CellB),
            C(f"{(ac_cost / twap_cost - 1) * 100:+.1f}%", CellB),
            C("Risk-adjusted schedule minimising E[Cost] + λVar[Cost]", CellB),
            C("Institutional block execution with explicit risk-cost trade-off", CellB)
        ],
    ]

    sc = Table(
        strat_data,
        colWidths=[2.1 * cm, 2.8 * cm, 2.0 * cm, 4.7 * cm, 4.9 * cm]
    )
    sc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LBG]),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#E8F5EE')),
    ]))
    story.append(sc)
    story.append(Spacer(1, 8))


    story.append(Paragraph(
        "The strategy comparison highlights the central trade-off faced by an execution desk: "
        "minimising expected transaction costs versus reducing exposure to adverse price movements "
        "during the execution horizon.",
        Bo
    ))

    story.append(Paragraph(
        f"<b>TWAP</b> achieves the lowest modelled expected cost "
        f"(<b>USD {twap_cost:,.0f}</b>) because it distributes the parent order uniformly across "
        "the trading day. However, its simplicity comes at the cost of higher timing risk: a large "
        "portion of the order remains exposed to market movements until late in the execution window.",
        Bo
    ))

    story.append(Paragraph(
        f"<b>VWAP</b> follows the historical intraday volume profile and is useful when the trader is "
        "benchmarked against market VWAP. In this case, however, VWAP produces a higher modelled cost "
        f"(<b>USD {vwap_cost:,.0f}</b>, {(vwap_cost / twap_cost - 1) * 100:+.1f}% vs TWAP) because its "
        "schedule trades larger slices around the open and close, where AAPL volume is concentrated; with a constant "
        "temporary-impact coefficient, the squared trade-size term makes uneven schedules more expensive than a uniform one.",
        Bo
    ))

    story.append(Paragraph(
        f"<b>Almgren-Chriss</b> provides the most balanced institutional execution trajectory. "
        f"The calibrated schedule executes <b>{first_slice_pct:.1f}%</b> in the first slice versus "
        f"<b>{100 / N_SLICES:.1f}%</b> for TWAP, reducing inventory exposure while avoiding the excessive "
        "market impact of immediate execution. The resulting cost is higher than TWAP but reflects a "
        "more explicit risk-adjusted execution objective.",
        Bo
    ))

    story.append(Paragraph("3.3 Efficient Frontier", Se2))
    story.append(Image(buf_frontier, width=14.5*cm, height=5.8*cm))
    story.append(Paragraph(
        f"Fig. 3 — Efficient frontier: expected execution cost versus execution risk "
        f"for varying λ. Moving left along the frontier corresponds to more aggressive "
        f"execution: risk decreases, but expected cost increases due to stronger market impact. "
        f"Moving right corresponds to more patient execution: expected cost decreases, but "
        f"timing risk increases. Green dot = selected λ = {risk_aversion:.3g}.",
        Ca
    ))
    story.append(Paragraph(
        f"The frontier illustrates the irreducible trade-off between "
        f"execution cost and execution risk. At low λ the strategy is patient (TWAP-like), "
        f"accepting high timing risk to minimise market impact. At high λ it becomes aggressive, "
        f"front-loading execution to eliminate price risk at the cost of higher impact. "
        f"The selected λ = {risk_aversion:.3g} positions the strategy at the "
        f"patient end of the frontier, consistent with a mildly front-loaded schedule.", Bo))

    story.append(Paragraph("3.4 Strategy Selection Matrix", Se2))

    selection_data = [
        [
            C("Strategy", CellH),
            C("Strengths", CellH),
            C("Weaknesses", CellH),
            C("Typical Use", CellH)
        ],

        [
            C("TWAP", CellB),
            C("Simple, transparent and operationally easy to implement"),
            C("Ignores liquidity profile and leaves high residual inventory risk"),
            C("Passive execution in stable and liquid markets")
        ],

        [
            C("VWAP", CellB),
            C("Tracks the market volume curve and aligns with a common benchmark"),
            C("Can trade too aggressively during high-spread or high-volatility windows"),
            C("Benchmark-driven execution for asset managers")
        ],

        [
            C("Almgren-Chriss", CellB),
            C("Explicitly balances expected market impact and timing risk"),
            C("Requires calibration of impact and risk-aversion parameters"),
            C("Institutional block execution")
        ],
    ]

    smt = Table(
        selection_data,
        colWidths=[2.4 * cm, 4.7 * cm, 5.2 * cm, 4.2 * cm]
    )
    smt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LBG]),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#E8F5EE')),
    ]))
    story.append(smt)
    story.append(Spacer(1, 8))

    # ── 4. PRICE IMPACT ──────────────────────────────────────────────────────
    story.append(Paragraph("4. Price Impact Estimation", Se))
    total_impact = (impact['temp_impact'] + impact['perm_impact']) * ORDER_SIZE
    impact_bps   = total_impact / order_value * 10000
    story.append(Paragraph(
        f"Price impact is estimated using the <b>square-root market impact model</b> "
        f"(Almgren et al., 2005): MI = Y × σ_daily × √(Q/ADV), with Y = {IMPACT_COEFF} "
        f"(an assumption, not calibrated on fills). "
        f"This provides an indicative pre-trade benchmark for assessing the size of the order "
        f"relative to market liquidity. For an order of <b>{ORDER_SIZE:,} shares</b> against an "
        f"ADV of <b>{impact['adv']:,.0f} shares</b>, the participation rate is "
        f"<b>{ORDER_SIZE / impact['adv'] * 100:.1f}%</b> of daily volume. "
        f"The estimated total impact benchmark is <b>USD {total_impact:,.0f}</b> "
        f"(<b>{impact_bps:.0f} bps</b> of notional), providing a reference point for evaluating "
        f"the scale of the execution problem.",
        Bo
    ))

    impact_data = [
        ["Parameter", "Value", "Formula / Note"],
        ["Daily Volatility (σ)",      f"{sigma*100:.2f}%",              "Std(daily returns)"],
        ["Annualised Volatility",      f"{impact['sigma']*100:.1f}% p.a.", "σ_daily × √252"],
        ["Avg Daily Volume (ADV)",     f"{impact['adv']:,.0f} shares",  "Mean of daily volume"],
        ["Participation Rate",         f"{ORDER_SIZE/impact['adv']*100:.1f}%", "Q / ADV"],
        ["Last Price",                 f"USD {impact['price']:.2f}",    "Most recent close"],
        ["Temp. Market Impact",        f"USD {impact['temp_impact']:.4f}/share", "Y × σ_daily × √(Q/ADV) × P"],
        ["Perm. Market Impact",        f"USD {impact['perm_impact']:.4f}/share", f"{PERM_TO_TEMP*100:.0f}% of temporary impact"],
        ["Total Impact Cost",          f"USD {total_impact:,.0f}",      f"({impact_bps:.0f} bps of notional)"],
    ]
    it = Table(impact_data, colWidths=[5*cm, 4.5*cm, 7*cm])
    it.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), DARK),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (1,1), (-1,-1),'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 8.8),
        ('TEXTCOLOR',     (0,1), (0,-1), DARK),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, LBG]),
        ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor('#CCC')),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(it); story.append(Spacer(1, 8))

    # ── 5. CORE CODE ─────────────────────────────────────────────────────────

    # ── 5. CONCLUSIONS ───────────────────────────────────────────────────────────
    story.append(Paragraph(
        "<b>Data and assumptions.</b> Volume and prices are 1-minute aggregate bars from Polygon "
        "(regular session only). The spread series is a heuristic range-based proxy, not NBBO quotes. "
        f"The impact coefficient Y = {IMPACT_COEFF} is an assumption; the AC temporary-impact coefficient "
        "η is derived from the same square-root benchmark (η = Y·σ·P·√(Q/ADV)/Q), so Sections 3 and 4 are "
        "consistent. λ is not estimated from a utility function: it is set to obtain a chosen first-slice "
        "size. ADV comes from Polygon aggregate bars and may differ from the consolidated volume shown by other "
        "providers, so absolute figures are indicative while the comparison between schedules is unaffected. "
        "All costs are model-implied and are meant to compare schedules, not to forecast realised costs.",
        Bo))
    story.append(Paragraph("5. Conclusions", Se))

    story.append(Paragraph(
        "This project demonstrates how optimal execution theory can be combined with real market "
        "microstructure information to design institutional execution strategies. By integrating "
        "intraday volume, a range-based spread proxy, realised volatility and market impact estimation, "
        "the framework provides a practical tool for comparing benchmark execution algorithms against "
        "a risk-adjusted Almgren-Chriss trajectory.",
        Bo
    ))

    story.append(Paragraph(
        f"For the analysed AAPL order, the calibrated Almgren-Chriss strategy deliberately accepts a "
        f"moderate increase in expected cost versus TWAP ({ac_vs_twap:+.1f}%) in exchange for a more "
        "controlled inventory profile and lower exposure to adverse price movements. This is consistent "
        "with the way an execution desk evaluates large orders: not only by expected cost, but by the "
        "balance between market impact, timing risk and benchmark sensitivity.",
        Bo
    ))

    story.append(Paragraph(
        "Although simplified relative to production execution systems, the model captures the fundamental "
        "economic trade-off underlying modern electronic execution. Natural extensions include stochastic "
        "liquidity, adaptive participation rates, order-book imbalance signals, limit-order execution and "
        "reinforcement-learning based execution policies.",
        Bo
    ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Gianluca Pogliana - +39 3403276133 - poglianagianluca@gmail.com", Fo))

    doc.build(story)
    print(f"PDF saved: {output_path}")

def calibrate_risk_aversion(X, T, N, sigma, eta, target_first_slice_pct=7.0):
    """
    Calibrate lambda so that the AC strategy executes a realistic percentage
    of the order in the first slice.

    Example:
    target_first_slice_pct = 18 means AC sells 18% in the first slice.
    TWAP would sell 10% with N=10.
    """

    target = target_first_slice_pct / 100

    lambdas = np.logspace(-14, 2, 1000)

    best_lambda = None
    best_error = np.inf
    best_first_slice = None
    best_kappa = None

    for lam in lambdas:
        ac = almgren_chriss(
            X=X,
            T=T,
            N=N,
            sigma=sigma,
            eta=eta,
            gamma=0.0,
            lam=lam
        )

        first_slice = abs(ac["n_t"][0]) / X
        error = abs(first_slice - target)

        if error < best_error:
            best_error = error
            best_lambda = lam
            best_first_slice = first_slice
            best_kappa = ac["kappa"]

    return best_lambda, best_first_slice * 100, best_kappa
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(df=None):
    print("=" * 60)
    print("Optimal Execution Model — Almgren-Chriss + Microstructure")
    print("=" * 60)

    # 1. Market data (Polygon, cached locally)
    if df is None:
        print(f"\n[1/6] Loading intraday data for {TICKER}...")
        if not POLYGON_API_KEY:
            raise SystemExit(
                "POLYGON_API_KEY not found. Create a file named .env in the same folder as this "
                "script (" + os.path.dirname(os.path.abspath(__file__)) + ") containing the line:\n"
                "POLYGON_API_KEY=your_key")
        df = load_or_download_data(TICKER, START_DATE, END_DATE, POLYGON_API_KEY)
        if df is None:
            raise SystemExit("No market data available: check API key / ticker / dates. "
                             "The model does not fall back to simulated data.")
    df = regular_session(df)
    print(f"  {len(df)} regular-session bars, "
          f"{df.index[0].date()} -> {df.index[-1].date()}")

    # 2. Microstructure profiles
    print("[2/6] Computing intraday microstructure profiles...")
    profile = compute_intraday_profile(df)

    # 3. Price impact benchmark and AC parameters
    print("[3/6] Estimating price impact parameters...")
    impact = compute_price_impact(df, ORDER_SIZE)
    ADV, price = impact['adv'], impact['price']
    sigma = impact['sigma_daily']            # daily vol, fractional
    sigma_px = sigma * price                 # daily vol, USD/share (AC works in price units)
    participation = ORDER_SIZE / ADV

    # eta [USD/share^2]: temporary impact per share from the square-root benchmark,
    # spread over the order size so that cost = eta/tau * sum(n^2).
    eta = IMPACT_COEFF * sigma * price * np.sqrt(participation) / ORDER_SIZE
    gamma = GAMMA_TO_ETA * eta

    risk_aversion, first_slice_pct, kappa = calibrate_risk_aversion(
        X=ORDER_SIZE, T=T_HORIZON, N=N_SLICES, sigma=sigma_px, eta=eta,
        target_first_slice_pct=FIRST_SLICE_MULTIPLE * 100.0 / N_SLICES)

    print(f"  ADV:           {ADV:,.0f} shares ({impact['n_days']} trading days)")
    print(f"  Price:         USD {price:.2f}   daily sigma: {sigma*100:.3f}%")
    print(f"  Participation: {participation*100:.3f}% of ADV")
    print(f"  eta / gamma:   {eta:.3e} / {gamma:.3e} USD/share^2")
    print(f"  lambda:        {risk_aversion:.4g}  kappa: {kappa:.4f}  first slice: {first_slice_pct:.2f}%")

    # 4. Strategies
    print("[4/6] Computing optimal execution trajectories...")
    strategies = compare_strategies(ORDER_SIZE, T_HORIZON, N_SLICES, sigma_px,
                                    eta, gamma, profile, risk_aversion)

    order_value = ORDER_SIZE * price
    print("\nModelled execution cost (model-implied, not realised)")
    for name, key in [("TWAP", "cost_twap"), ("VWAP", "cost_vwap"), ("AC", "cost_ac")]:
        c = strategies[key]
        print(f"  {name:5s} USD {c:,.0f}  ({c / order_value * 1e4:.2f} bps of notional)")

    # 5. Charts
    print("[5/6] Generating charts...")
    buf_micro = make_fig_microstructure(profile)
    buf_traj = make_fig_ac_trajectory(strategies, ORDER_SIZE)
    buf_frontier = make_fig_cost_frontier(ORDER_SIZE, T_HORIZON, N_SLICES,
                                          sigma_px, eta, gamma, risk_aversion)

    # 6. PDF
    print("[6/6] Building PDF...")
    build_pdf(buf_micro, buf_traj, buf_frontier, strategies, impact, profile,
              eta, gamma, sigma, risk_aversion, first_slice_pct, OUTPUT_PATH)


if __name__ == '__main__':
    main()
