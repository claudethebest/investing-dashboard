"""
Stock/ETF data engine for the index-investing dashboard.

Goal (per the user): NOT life-changing returns — just "don't get poorer."
So the whole thing is built around the boring truth that ~90% of pros can't
beat a broad index fund. We pull real data, compute long-term metrics, and
backtest simple "lazy" index portfolios — always shown next to plain S&P 500
and an inflation line, so the honest benchmark is always visible.

Data: Yahoo Finance chart API (free, no key). Uses ADJUSTED close, so dividends
are included (total return) — important for bonds and dividend stocks.

Run: python stock_data.py   -> writes stocks.json + chart_stocks.png
"""

import json
import datetime as dt
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UA = {"User-Agent": "Mozilla/5.0"}

# universe — ETFs are the core (index tracking); a few stocks for context
ETFS = {
    "SPY": "S&P 500 (US large-cap)",
    "QQQ": "Nasdaq-100 (US tech)",
    "VTI": "Total US market",
    "VXUS": "International ex-US",
    "BND": "US bonds",
    "VT": "Total world",
}
STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "KO": "Coca-Cola", "TSLA": "Tesla",
}
# simple lazy portfolios (weights sum to 1)
PORTFOLIOS = {
    "S&P 500 only": {"SPY": 1.0},
    "Classic 60/40": {"VTI": 0.60, "BND": 0.40},
    "3-Fund (global)": {"VTI": 0.50, "VXUS": 0.30, "BND": 0.20},
}
INFLATION = 0.03   # ~3%/yr — the "don't get poorer" hurdle


def fetch_yahoo(ticker, rng="10y"):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={rng}&interval=1d"
    r = requests.get(u, headers=UA, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    ind = res["indicators"]
    adj = (ind.get("adjclose", [{}])[0].get("adjclose")) or ind["quote"][0]["close"]
    s = pd.Series(
        {dt.datetime.utcfromtimestamp(t).date(): c
         for t, c in zip(ts, adj) if c is not None})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def metrics(s):
    v = s.values.astype(float)
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (v[-1] / v[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(v)
    max_dd = float(((v - peak) / peak).min() * 100)
    vol = float(np.std(np.diff(np.log(v))) * np.sqrt(252) * 100)
    return {
        "years": round(years, 1),
        "total_ret_pct": round((v[-1] / v[0] - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd, 1),
        "vol_pct": round(vol, 1),
        "grow_10k": round(10000 * v[-1] / v[0]),
        "real_cagr_pct": round((cagr - INFLATION) * 100, 1),   # after inflation
    }


def backtest_portfolio(prices, weights):
    """Daily-rebalanced constant-weight portfolio (a standard simplification;
    monthly rebalancing differs only marginally). Returns an equity Series."""
    cols = list(weights)
    df = prices[cols].dropna()
    rets = df.pct_change().fillna(0)
    w = np.array([weights[c] for c in cols])
    port_ret = (rets.values * w).sum(axis=1)
    equity = 10000 * np.cumprod(1 + port_ret)
    return pd.Series(equity, index=df.index)


def downsample(s, n=400):
    if len(s) <= n:
        return s
    idx = np.linspace(0, len(s) - 1, n).astype(int)
    return s.iloc[idx]


def main():
    tickers = {**ETFS, **STOCKS}
    print("fetching", len(tickers), "tickers from Yahoo ...")
    prices = {}
    for t in tickers:
        try:
            prices[t] = fetch_yahoo(t)
            print(f"  {t:5} {len(prices[t])} days  ({prices[t].index[0].date()} -> {prices[t].index[-1].date()})")
        except Exception as e:
            print(f"  {t:5} FAIL {e}")
    price_df = pd.DataFrame(prices)

    # per-asset metrics
    assets = []
    print("\n--- per-asset (10y, total return incl. dividends) ---")
    print(f"{'ticker':6}{'CAGR':>7}{'real':>7}{'maxDD':>8}{'vol':>7}{'$10k->':>10}")
    for t, name in tickers.items():
        if t not in prices:
            continue
        m = metrics(prices[t])
        kind = "ETF" if t in ETFS else "stock"
        assets.append({"ticker": t, "name": name, "kind": kind, **m})
        print(f"{t:6}{m['cagr_pct']:>6.1f}%{m['real_cagr_pct']:>6.1f}%"
              f"{m['max_dd_pct']:>7.1f}%{m['vol_pct']:>6.1f}%{m['grow_10k']:>10,}")

    # lazy portfolios
    portfolios, curves = [], {}
    print("\n--- lazy portfolios ($10k, 10y) ---")
    for name, w in PORTFOLIOS.items():
        eq = backtest_portfolio(price_df, w)
        m = metrics(eq)
        portfolios.append({"name": name, "weights": w, **m})
        curves[name] = eq
        print(f"  {name:18} CAGR {m['cagr_pct']:>5.1f}%  maxDD {m['max_dd_pct']:>6.1f}%  $10k->{m['grow_10k']:>9,}")

    # chart: portfolio growth vs inflation
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"S&P 500 only": "#d62728", "Classic 60/40": "#2ca02c", "3-Fund (global)": "#4c9be8"}
    base = None
    for name, eq in curves.items():
        d = downsample(eq)
        ax.plot(d.index, d.values, lw=1.8, color=colors.get(name), label=f"{name} ({metrics(eq)['cagr_pct']:+.1f}%/yr)")
        base = d.index if base is None else base
    infl = 10000 * (1 + INFLATION) ** ((base - base[0]).days / 365.25)
    ax.plot(base, infl, color="#888", lw=1.3, ls="--", label="inflation (~3%/yr) — the 'don't get poorer' line")
    ax.set_title("Lazy index portfolios — $10k over 10 years", fontweight="bold")
    ax.set_ylabel("USD"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig("chart_stocks.png", dpi=110)

    # single-stock risk overlay: $10k growth of a few individual names vs the index
    RISK_SET = ["SPY", "AAPL", "MSFT", "KO", "TSLA"]
    asset_curves = {}
    for t in RISK_SET:
        if t in prices:
            norm = 10000 * prices[t] / prices[t].iloc[0]
            asset_curves[t] = [round(x) for x in downsample(norm).values]
    asset_curve_dates = [d.strftime("%Y-%m-%d") for d in downsample(prices["SPY"]).index]

    # live-ish quotes (latest close)
    quotes = {t: round(float(prices[t].iloc[-1]), 2) for t in prices}

    out = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "inflation_pct": INFLATION * 100,
        "assets": assets, "portfolios": portfolios,
        "curve_dates": [d.strftime("%Y-%m-%d") for d in downsample(curves["S&P 500 only"]).index],
        "curves": {name: [round(x) for x in downsample(eq).values] for name, eq in curves.items()},
        "inflation_curve": [round(x) for x in infl],
        "asset_curves": asset_curves,
        "asset_curve_dates": asset_curve_dates,
        "quotes": quotes,
    }
    with open("stocks.json", "w") as f:
        json.dump(out, f)
    print("\nwrote stocks.json + chart_stocks.png")


if __name__ == "__main__":
    main()
