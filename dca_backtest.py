"""
DCA (dollar-cost averaging) backtest — buy a fixed amount of BTC every day.

The simplest possible strategy: every day, spend X on BTC, regardless of price.
No timing, no bot, no API trading needed (Binance TH "Auto-Invest" / recurring
buy does this for you). It keeps buying through crashes (when you'd be scared)
and through rallies (when you'd be greedy) — the behavior that actually works.

Compares three things on the same money:
  - DCA           : spend X/day, accumulate
  - Lump sum      : invest the WHOLE budget on day one (more time in market)
  - Cash (no buy) : keep the THB — the do-nothing baseline (value == invested)

Models the real Binance TH fee (0.25% per buy). Run: python dca_backtest.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grid_backtest as gb


def dip_multipliers(df, ma_window=30):
    """Buy more when price is below its trailing MA (no lookahead):
       within 5% of MA -> 1x, 5-15% below -> 2x, >15% below -> 3x."""
    price = df["close"].values
    ma = df["close"].rolling(ma_window).mean().values
    mult = np.ones(len(price))
    for i in range(len(price)):
        if np.isnan(ma[i]):
            continue
        gap = price[i] / ma[i] - 1
        if gap < -0.15:
            mult[i] = 3.0
        elif gap < -0.05:
            mult[i] = 2.0
    return mult


def run_dca(df, daily, fee, dip=False):
    price = df["close"].values
    n = len(price)
    mult = dip_multipliers(df) if dip else np.ones(n)
    spend = daily * mult                            # THB spent each day
    btc_bought = (spend * (1 - fee)) / price        # BTC acquired each day
    btc_cum = np.cumsum(btc_bought)
    invested = np.cumsum(spend)                     # cumulative THB put in
    value = btc_cum * price                         # mark-to-market

    # lump sum: whole budget deployed on day 0
    total = daily * n
    lump_btc = total * (1 - fee) / price[0]
    lump_value = lump_btc * price

    avg_cost = invested[-1] / btc_cum[-1]           # average price paid
    peak = np.maximum.accumulate(value)
    max_dd = float(((value - peak) / peak).min() * 100)

    return {
        "days": n, "invested": invested[-1], "btc": btc_cum[-1],
        "value": value[-1], "profit": value[-1] - invested[-1],
        "roi_pct": (value[-1] / invested[-1] - 1) * 100,
        "avg_cost": avg_cost, "final_price": price[-1],
        "lump_value": lump_value[-1],
        "lump_roi_pct": (lump_value[-1] / total - 1) * 100,
        "max_dd": max_dd,
        "t": df["dt"].values, "value_curve": value,
        "invested_curve": invested, "lump_curve": lump_value,
    }


def report(name, sym, ccy, df, daily, fee):
    r = run_dca(df, daily, fee)
    d = run_dca(df, daily, fee, dip=True)
    print(f"\n{'='*60}\n{name}  —  {daily:g} {ccy}/day into {sym}\n{'='*60}")
    print(f"  period          : {r['days']} days (~{r['days']/365:.2f} yr)")
    print(f"  [flat DCA]   invested {r['invested']:,.0f}  avg cost {r['avg_cost']:,.0f}  "
          f"value {r['value']:,.0f}  ROI {r['roi_pct']:+.1f}%")
    print(f"  [dip DCA]    invested {d['invested']:,.0f}  avg cost {d['avg_cost']:,.0f}  "
          f"value {d['value']:,.0f}  ROI {d['roi_pct']:+.1f}%")
    print(f"  final price  : {r['final_price']:,.0f}")
    print(f"  lump sum     : ROI {r['lump_roi_pct']:+.1f}%  (all on day 1)")
    print(f"  dip vs flat  : {d['roi_pct']-r['roi_pct']:+.1f} ROI points")
    return r, d


def plot(rows, out="chart_dca.png"):
    fig, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 5), squeeze=False)
    for ax, (title, ccy, r, d) in zip(axes[0], rows):
        t = r["t"]
        ax.plot(t, r["invested_curve"], color="#8b949e", lw=1.4, label="cash invested (do nothing)")
        ax.plot(t, r["lump_curve"], color="#4c9be8", lw=1.3, ls="--", label=f"lump sum ({r['lump_roi_pct']:+.0f}%)")
        ax.fill_between(t, r["invested_curve"], r["value_curve"],
                        where=(r["value_curve"] >= r["invested_curve"]), color="#2ca02c", alpha=0.15)
        ax.fill_between(t, r["invested_curve"], r["value_curve"],
                        where=(r["value_curve"] < r["invested_curve"]), color="#d62728", alpha=0.15)
        ax.plot(t, d["value_curve"], color="#e8a33d", lw=1.6, label=f"DCA + dip-buying ({d['roi_pct']:+.0f}%)")
        ax.plot(t, r["value_curve"], color="#d62728", lw=2, label=f"flat DCA ({r['roi_pct']:+.0f}%)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ccy); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\nchart -> {out}")


if __name__ == "__main__":
    START = "1 Jan 2024"
    rows = []

    df_th = gb.fetch_klines("https://api.binance.th", "/api/v1/klines",
                            "BTCTHB", "1d", gb.to_ms(START))
    df_th, _ = gb.clean_klines(df_th)
    r_th, d_th = report("BINANCE THAILAND", "BTCTHB", "THB", df_th, 108, 0.0025)
    rows.append(("Binance TH — 108 THB/day DCA", "THB", r_th, d_th))

    df_us = gb.fetch_klines("https://api.binance.com", "/api/v3/klines",
                            "BTCUSDT", "1d", gb.to_ms(START))
    df_us, _ = gb.clean_klines(df_us)
    r_us, d_us = report("GLOBAL BINANCE", "BTCUSDT", "USDT", df_us, 3, 0.0010)
    rows.append(("Global — $3/day DCA", "USDT", r_us, d_us))

    plot(rows)
