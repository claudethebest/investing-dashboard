"""
Grid / infinity-grid backtest for "buy low, wait, sell high" on spot BTC,
with order-book slippage modeling and charts.

Markets (same strategy, run side by side):
  - BTCUSDT on global Binance   (api.binance.com, /api/v3/klines, /api/v3/depth)
  - BTCTHB  on Binance Thailand (api.binance.th,  /api/v1/klines, /api/v1/depth)

All data comes from PUBLIC endpoints — no API key, no auth.

Three things this script does:
  1. Pull paginated historical klines (1000/call, page forward by startTime).
  2. Measure REAL per-fill slippage by walking the live order book (/depth),
     so the thin-book cost the price-only backtest ignores is priced in.
  3. Simulate the grid, track an equity curve, and render charts to PNG.

HONEST LIMITATIONS:
  - The sim steps CLOSE-TO-CLOSE: intra-bar round trips are missed, so fills
    are UNDERSTATED. Finer candles (15m/5m) capture more. Grids feed on wiggle.
  - Slippage is measured from ONE current order-book snapshot and applied as a
    constant. Real books vary with time/volatility; treat it as a calibrated
    estimate, not a guarantee. Still far better than assuming zero cost.

Usage:
  pip install requests pandas numpy matplotlib
  python grid_backtest.py
Charts are written next to this file as PNGs.
"""

import time
import datetime as dt
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # headless: render straight to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------------------------------
# 1. DATA: paginated public klines
# --------------------------------------------------------------------------
def fetch_klines(base_url, path, symbol, interval, start_ms, end_ms=None):
    out, url = [], base_url + path
    while True:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": start_ms, "limit": 1000}
        if end_ms:
            params["endTime"] = end_ms
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        start_ms = batch[-1][0] + 1           # next page = 1ms after last open
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tbqv", "ignore"]
    df = pd.DataFrame(out, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["dt", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def to_ms(date_str):
    return int(dt.datetime.strptime(date_str, "%d %b %Y").timestamp() * 1000)


def clean_klines(df, dev=0.25, window=7):
    """
    Drop spurious price prints. New/thin exchanges (e.g. Binance TH at launch)
    emit bad ticks — a close that deviates from its local median by more than
    `dev` is almost certainly a glitch, not a real move. Leaving them in lets a
    spike-then-revert book a PHANTOM grid round-trip (fake profit), so we remove
    rows where close OR high OR low blows past the rolling median.
    """
    med = df["close"].rolling(window, center=True, min_periods=3).median()
    bad = ((df["close"] / med - 1).abs() > dev) | \
          ((df["high"] / med - 1).abs() > dev) | \
          ((df["low"] / med - 1).abs() > dev)
    removed = int(bad.sum())
    return df[~bad].reset_index(drop=True), removed


# --------------------------------------------------------------------------
# 2. SLIPPAGE: walk the live order book to price a real fill
# --------------------------------------------------------------------------
def measure_slippage(base_url, path, symbol, order_value_quote, limit=1000):
    """
    Pull the current order book and compute the average cost, as a fraction of
    mid price, to execute an order worth `order_value_quote` on each side.

    Returns (slip_fraction, spread_fraction). slip_fraction is half-spread plus
    market impact for that order size — the per-fill cost a taker really pays.
    """
    r = requests.get(base_url + path, params={"symbol": symbol, "limit": limit}, timeout=20)
    r.raise_for_status()
    book = r.json()
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    if not bids or not asks:
        return 0.0, 0.0
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread = (best_ask - best_bid) / mid

    def walk(levels, value):                  # VWAP to fill `value` quote, vs mid
        filled_q = filled_v = 0.0
        for price, qty in levels:
            lvl_value = price * qty
            take_v = min(lvl_value, value - filled_v)
            take_q = take_v / price
            filled_v += take_v
            filled_q += take_q
            if filled_v >= value:
                break
        if filled_q == 0:
            return 0.0
        vwap = filled_v / filled_q
        return vwap

    buy_vwap = walk(asks, order_value_quote)            # buying lifts the ask
    sell_vwap = walk(list(bids), order_value_quote)     # selling hits the bid
    buy_cost = (buy_vwap / mid - 1) if buy_vwap else 0.0
    sell_cost = (1 - sell_vwap / mid) if sell_vwap else 0.0
    slip = max(0.0, (buy_cost + sell_cost) / 2)
    return slip, spread


# --------------------------------------------------------------------------
# 3. STRATEGY: geometric grid with fee + slippage, tracking an equity curve
# --------------------------------------------------------------------------
def build_levels(lower, upper, ratio):
    levels = [lower]
    while levels[-1] < upper:
        levels.append(levels[-1] * (1 + ratio))
    return np.array(levels)


def choose_spacing(df, fee, slippage=0.0, vol_mult=3.0, margin=1.5):
    """
    Pick grid spacing from two principles instead of a hardcoded guess:

      1. FEE FLOOR — a round-trip pays fee+slippage on BOTH legs, so spacing
         must clear 2*(fee+slippage) with a safety margin or every trade is a
         structural loser. On Binance TH (0.25% fee) this floor is ~0.75%.
      2. VOLATILITY SCALING — each rung should span a few bars' worth of typical
         range (vol_mult * ATR%), so rungs widen in choppy/volatile markets and
         tighten when calm, instead of churning on noise.

    Effective spacing = max(fee_floor, vol_scaled). Returns (ratio, info).
    """
    floor = 2 * (fee + slippage) * margin
    tr = (df["high"] - df["low"]) / df["close"]          # per-bar true range %
    atr_pct = float(tr.tail(min(len(df), 2000)).mean())  # recent average
    vol_scaled = vol_mult * atr_pct
    ratio = max(floor, vol_scaled)
    info = {"fee_floor": floor, "atr_pct": atr_pct, "vol_scaled": vol_scaled,
            "ratio": ratio, "bound_by": "fee-floor" if floor >= vol_scaled else "volatility"}
    return ratio, info


def run_grid(df, lower, upper, ratio, capital, fee, slippage=0.0,
             regime_filter=False, ma_window=200, core_fraction=0.0):
    """
    Event-driven grid on CLOSE prices. Each adjacent level pair is a "cell"
    (buy@L, sell@L_next). Cells above the start price are pre-bought to seed
    sells; cells below wait to buy dips. Fee AND slippage are charged per fill.

    core_fraction: 0..1 of capital bought once at the start and HELD forever
    (never sold). This is the "core + satellite" refinement — the core captures
    melt-up upside the grid would otherwise sell away; only the remaining
    (1-core_fraction) runs the grid to harvest chop. core_fraction=0 is the
    pure grid; core_fraction=1 is just buy & hold.

    Returns a results dict including an 'equity' array (mark-to-market per bar).
    """
    levels = build_levels(lower, upper, ratio)
    buy_p, sell_p = levels[:-1], levels[1:]
    n = len(buy_p)
    price = df["close"].values
    price0 = price[0]

    grid_capital = capital * (1 - core_fraction)
    quote_per_cell = grid_capital / n

    # core tranche: bought once at the start, held to the end
    core_base = (capital * core_fraction) * (1 - fee) / (price0 * (1 + slippage)) \
        if core_fraction > 0 else 0.0

    holding = np.zeros(n, dtype=bool)
    qty = np.zeros(n)
    cash = grid_capital
    total_base = 0.0

    for i in range(n):                        # seed inventory above start price
        if buy_p[i] >= price0:
            base = quote_per_cell * (1 - fee) / (price0 * (1 + slippage))
            qty[i] = base
            total_base += base
            holding[i] = True
            cash -= quote_per_cell

    ma = df["close"].rolling(ma_window).mean().values if regime_filter else None

    equity = np.empty(len(price))
    equity[0] = cash + (total_base + core_base) * price0
    trades = 0
    prev = price0
    for k in range(1, len(price)):
        p = price[k]
        if p > prev:                          # rally -> sell held cells
            for i in np.where(holding & (sell_p <= p))[0]:
                cash += qty[i] * sell_p[i] * (1 - slippage) * (1 - fee)
                total_base -= qty[i]
                qty[i] = 0.0
                holding[i] = False
                trades += 1
        elif p < prev:                        # dip -> buy unheld cells
            paused = regime_filter and not np.isnan(ma[k]) and p < ma[k]
            if not paused:
                for i in np.where((~holding) & (buy_p >= p))[0]:
                    base = quote_per_cell * (1 - fee) / (buy_p[i] * (1 + slippage))
                    qty[i] = base
                    total_base += base
                    holding[i] = True
                    cash -= quote_per_cell
                    trades += 1
        equity[k] = cash + (total_base + core_base) * p   # core held throughout
        prev = p

    last = price[-1]
    final_value = cash + (total_base + core_base) * last
    bh_curve = capital / price0 * price        # buy & hold equity curve
    return {
        "levels": n + 1, "trades": trades,
        "start_price": price0, "end_price": last,
        "cash_left": cash, "base_held": total_base + core_base,
        "core_fraction": core_fraction,
        "final_value": final_value,
        "grid_return_pct": (final_value / capital - 1) * 100,
        "buyhold_return_pct": (bh_curve[-1] / capital - 1) * 100,
        "equity": equity, "bh_curve": bh_curve,
    }


# --------------------------------------------------------------------------
# 4. CHARTS
# --------------------------------------------------------------------------
def plot_market(name, symbol, df, lower, upper, res_noslip, res_slip, slip, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    t = df["dt"]

    # --- price + grid band ---
    ax1.plot(t, df["close"], color="#222", lw=0.8, label="BTC price")
    ax1.axhspan(lower, upper, color="#4c9be8", alpha=0.10, label="grid range")
    ax1.axhline(lower, color="#4c9be8", lw=0.8, ls="--")
    ax1.axhline(upper, color="#4c9be8", lw=0.8, ls="--")
    ax1.set_title(f"{name}  —  {symbol}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.2)

    # --- equity curves: HODL vs grid(no slip) vs grid(with slip) ---
    cap = res_slip["equity"][0]
    ax2.plot(t, res_slip["bh_curve"], color="#888", lw=1.3, label=f"buy & hold ({res_slip['buyhold_return_pct']:+.1f}%)")
    ax2.plot(t, res_noslip["equity"], color="#2ca02c", lw=1.3,
             label=f"grid, no slippage ({res_noslip['grid_return_pct']:+.1f}%)")
    ax2.plot(t, res_slip["equity"], color="#d62728", lw=1.5,
             label=f"grid, {slip*100:.2f}% slip/fill ({res_slip['grid_return_pct']:+.1f}%)")
    ax2.axhline(cap, color="#000", lw=0.6, ls=":")
    ax2.set_ylabel("portfolio value")
    ax2.set_title(f"equity curve   (slippage drag: {res_noslip['grid_return_pct']-res_slip['grid_return_pct']:.1f} pts, "
                  f"{res_slip['trades']} fills)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def plot_summary(rows, out_png):
    """Grouped bar chart: final return per market for HODL / grid / grid+slip."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, [r["hodl"] for r in rows], w, label="buy & hold", color="#888")
    ax.bar(x,     [r["grid"] for r in rows], w, label="grid (no slippage)", color="#2ca02c")
    ax.bar(x + w, [r["grid_slip"] for r in rows], w, label="grid (with slippage)", color="#d62728")
    for i, r in enumerate(rows):
        for off, key in ((-w, "hodl"), (0, "grid"), (w, "grid_slip")):
            ax.text(i + off, r[key], f"{r[key]:+.0f}", ha="center",
                    va="bottom" if r[key] >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#000", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("return %")
    ax.set_title("Grid vs buy & hold — slippage priced in", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# 5. RUN
# --------------------------------------------------------------------------
def run_market(name, base_url, kpath, dpath, symbol, interval, start_str,
               capital, fee, ratio):
    print(f"\n{'='*64}\n{name}  ({symbol} @ {interval})\n{'='*64}")
    df = fetch_klines(base_url, kpath, symbol, interval, to_ms(start_str))
    if df.empty:
        print("  no data — check symbol/endpoint."); return None
    df, removed = clean_klines(df)
    if removed:
        print(f"  despiked {removed} bad tick(s) from raw data")
    lower = df["close"].min() * 0.98
    upper = df["close"].max() * 1.02
    quote_per_cell = capital / max(1, (np.log(upper / lower) / np.log(1 + ratio)))

    slip, spread = measure_slippage(base_url, dpath, symbol, quote_per_cell)
    print(f"  candles {len(df)}  |  {df['dt'].iloc[0].date()} -> {df['dt'].iloc[-1].date()}")
    print(f"  grid {lower:,.0f} -> {upper:,.0f}  spacing {ratio*100:.1f}%")
    print(f"  capital {capital:,.0f}  |  per-grid order ~{quote_per_cell:,.0f}")
    print(f"  live book: touch spread {spread*100:.4f}%")
    # depth probe: how slippage grows with order size (maps real book depth)
    print("  order-size -> taker slippage:")
    for mult, tag in [(1, "1x grid"), (10, "10x"), (100, "100x"), (1000, "1000x")]:
        s, _ = measure_slippage(base_url, dpath, symbol, quote_per_cell * mult)
        print(f"      {quote_per_cell*mult:>14,.0f}  ({tag:>7})  ->  {s*100:.4f}%")
    print(f"  modeled slippage at grid size: {slip*100:.4f}%/fill")

    res_noslip = run_grid(df, lower, upper, ratio, capital, fee, slippage=0.0)
    res_slip   = run_grid(df, lower, upper, ratio, capital, fee, slippage=slip)

    print(f"  fills {res_slip['trades']}")
    print(f"  buy & hold            : {res_slip['buyhold_return_pct']:+.1f}%")
    print(f"  grid, no slippage     : {res_noslip['grid_return_pct']:+.1f}%")
    print(f"  grid, with slippage   : {res_slip['grid_return_pct']:+.1f}%   "
          f"(drag {res_noslip['grid_return_pct']-res_slip['grid_return_pct']:.1f} pts)")
    print(f"  edge vs HODL (net)    : {res_slip['grid_return_pct']-res_slip['buyhold_return_pct']:+.1f}%")

    png = plot_market(name, symbol, df, lower, upper, res_noslip, res_slip, slip,
                      f"chart_{symbol}_{interval}.png")
    print(f"  chart -> {png}")
    return {"label": f"{symbol}\n{interval}",
            "hodl": res_slip["buyhold_return_pct"],
            "grid": res_noslip["grid_return_pct"],
            "grid_slip": res_slip["grid_return_pct"]}


if __name__ == "__main__":
    # ----------------------------- CONFIG -----------------------------
    INTERVAL   = "15m"         # 15m captures intra-bar wiggle the grid feeds on
    START      = "1 Jan 2024"
    # Capital is in each market's QUOTE currency. Keep them ~equal in USD so the
    # slippage comparison is fair: ~$10k each (1 USD ~ 35 THB).
    CAP_USDT   = 10_000.0
    CAP_THB    = 350_000.0
    # Per-market fees (real schedules):
    #   global BTCUSDT  = 0.10% taker (digital/digital)
    #   Binance TH BTCTHB = 0.25% FLAT, no maker/taker split, incl VAT
    #   (source: binance.th spot-trading fee FAQ). 2.5x the global rate.
    FEE_USDT   = 0.0010
    FEE_THB    = 0.0025
    GRID_RATIO = 0.01          # 1% spacing
    # ------------------------------------------------------------------

    rows = []
    r = run_market("GLOBAL BINANCE", "https://api.binance.com",
                   "/api/v3/klines", "/api/v3/depth",
                   "BTCUSDT", INTERVAL, START, CAP_USDT, FEE_USDT, GRID_RATIO)
    if r: rows.append(r)

    r = run_market("BINANCE THAILAND", "https://api.binance.th",
                   "/api/v1/klines", "/api/v1/depth",
                   "BTCTHB", INTERVAL, START, CAP_THB, FEE_THB, GRID_RATIO)
    if r: rows.append(r)

    if rows:
        png = plot_summary(rows, "chart_summary.png")
        print(f"\nsummary chart -> {png}")
