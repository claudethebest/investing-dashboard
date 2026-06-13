"""
Run the grid backtest for both markets and dump everything the dashboard needs
into results.json (downsampled series + headline numbers + depth probe).

Reuses the core functions from grid_backtest.py so the math stays in one place.
Run this whenever you want to refresh the backtest:  python build_data.py
"""

import json
import datetime as dt
import numpy as np
import grid_backtest as gb
import dca_backtest as dca


def build_dca_plan(start, daily, fee):
    """Compute the live DCA-plan tracker: how much BTC you'd have accumulated and
    your average cost, buying `daily` THB of BTCTHB every day since `start`.
    The dashboard marks this to the LIVE price each tick."""
    df = gb.fetch_klines("https://api.binance.th", "/api/v1/klines",
                         "BTCTHB", "1d", gb.to_ms(start))
    df, _ = gb.clean_klines(df)
    r = dca.run_dca(df, daily, fee)
    return {
        "daily": daily, "ccy": "THB", "freq": "daily", "start": start,
        "days": r["days"], "invested": round(r["invested"]),
        "btc": r["btc"], "avg_cost": round(r["avg_cost"]),
        "asof_price": round(r["final_price"]), "asof_roi": round(r["roi_pct"], 1),
    }


def downsample(arr, n=800):
    """Thin a long series to ~n points for the browser (keeps first & last)."""
    arr = np.asarray(arr)
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


CORE_FRACTION = 0.60   # 60% held as core, 40% runs the grid


def build_market(name, base, kpath, dpath, symbol, quote, interval,
                 start, capital, fee, ratio):
    df = gb.fetch_klines(base, kpath, symbol, interval, gb.to_ms(start))
    df, removed = gb.clean_klines(df)
    lower = df["close"].min() * 0.98
    upper = df["close"].max() * 1.02
    n_cells = max(1, np.log(upper / lower) / np.log(1 + ratio))
    quote_per_cell = capital / n_cells

    slip, spread = gb.measure_slippage(base, dpath, symbol, quote_per_cell)

    # FEE-FLOOR SAFETY RAIL: never grid tighter than fees can clear.
    req_ratio = ratio
    spacing_ratio, sp_info = gb.choose_spacing(df, fee, slip, vol_mult=3.0, margin=1.5)
    eff_ratio = max(req_ratio, sp_info["fee_floor"])     # floor the configured ratio
    floored = eff_ratio > req_ratio
    ratio = eff_ratio
    # rebuild cell count/order size at the effective ratio
    n_cells = max(1, np.log(upper / lower) / np.log(1 + ratio))
    quote_per_cell = capital / n_cells
    probe = []
    for mult in (1, 10, 100, 1000):
        s, _ = gb.measure_slippage(base, dpath, symbol, quote_per_cell * mult)
        probe.append({"order": round(quote_per_cell * mult, 2), "slip_pct": round(s * 100, 4)})

    res_s   = gb.run_grid(df, lower, upper, ratio, capital, fee, slippage=slip)
    res_cs  = gb.run_grid(df, lower, upper, ratio, capital, fee, slippage=slip,
                          core_fraction=CORE_FRACTION)

    dates = [d.strftime("%Y-%m-%d") for d in df["dt"]]
    return {
        "name": name, "symbol": symbol, "quote": quote, "interval": interval,
        "capital": capital, "fee_pct": round(fee * 100, 3),
        "spread_pct": round(spread * 100, 4), "slip_pct": round(slip * 100, 4),
        "removed_ticks": removed, "fills": res_s["trades"],
        "range": [round(lower, 2), round(upper, 2)],
        "cells": res_s["levels"] - 1,
        "core_fraction": CORE_FRACTION,
        "req_spacing_pct": round(req_ratio * 100, 3),
        "fee_floor_pct": round(sp_info["fee_floor"] * 100, 3),
        "eff_spacing_pct": round(ratio * 100, 3),
        "spacing_floored": floored,
        "hodl_pct": round(res_s["buyhold_return_pct"], 1),
        "grid_pct": round(res_s["grid_return_pct"], 1),
        "coresat_pct": round(res_cs["grid_return_pct"], 1),
        "edge_pct": round(res_s["grid_return_pct"] - res_s["buyhold_return_pct"], 1),
        "coresat_edge_pct": round(res_cs["grid_return_pct"] - res_s["buyhold_return_pct"], 1),
        "depth_probe": probe,
        "dates": downsample(dates),
        "price": [round(x, 2) for x in downsample(df["close"].values)],
        "grid_equity": [round(x, 2) for x in downsample(res_s["equity"])],
        "coresat_equity": [round(x, 2) for x in downsample(res_cs["equity"])],
        "hodl_equity": [round(x, 2) for x in downsample(res_s["bh_curve"])],
    }


def main():
    INTERVAL, START, RATIO = "15m", "1 Jan 2024", 0.01
    markets = []
    print("building BTCUSDT ...")
    markets.append(build_market(
        "Global Binance", "https://api.binance.com",
        "/api/v3/klines", "/api/v3/depth", "BTCUSDT", "USDT",
        INTERVAL, START, 10_000.0, 0.0010, RATIO))
    print("building BTCTHB ...")
    markets.append(build_market(
        "Binance Thailand", "https://api.binance.th",
        "/api/v1/klines", "/api/v1/depth", "BTCTHB", "THB",
        INTERVAL, START, 350_000.0, 0.0025, RATIO))

    print("building DCA plan ...")
    dca_plan = build_dca_plan(START, 108, 0.0025)

    # baked live-price snapshot — fallback for the static site when the browser
    # can't reach Binance directly (the page tries live first, then uses these)
    quotes = {}
    for sym, url in [("BTCUSDT", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"),
                     ("BTCTHB", "https://api.binance.th/api/v1/ticker/price?symbol=BTCTHB")]:
        try:
            quotes[sym] = float(__import__("requests").get(url, timeout=8).json()["price"])
        except Exception:
            pass

    out = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "interval": INTERVAL, "start": START, "grid_ratio_pct": RATIO * 100,
        "markets": markets,
        "dca_plan": dca_plan,
        "quotes": quotes,
        "findings": [
            "Slippage is negligible at retail size on both pairs — fees dominate, not the order book.",
            "Binance TH charges 0.25% flat per fill (no maker/taker) — 2.5x global's 0.10%. Tight grids bleed on TH.",
            "Early-2024 BTCTHB data had bad ticks that booked phantom profit; despiking cut THB from +87.5% to +63.7%.",
            "After cleaning, both pairs show ~+18.5% edge over buy & hold — the strategy edge is consistent, not THB magic.",
            "The grid's win is DEFENSIVE: HODL led massively at the Oct-2025 peak, then gave it back. Grid only wins measured post-drawdown.",
            "Net edge ~18% over 2.5 years (~7%/yr) — real but modest, and nowhere near 1-3% daily.",
            "Core+satellite is a RISK DIAL, not a free win: at the Oct-2025 peak HODL +194% / core+sat +150% / grid +84%; after the drawdown grid +69% / core+sat +57% / HODL +50%. core_fraction slides you between max-upside and max-defense.",
            "FEE FLOOR safety rail: on TH a 0.30% grid does 28k fills and hands ~all its edge to fees (+18% -> +0.7%). Spacing is now floored at ~2x round-trip fees (~0.75% on TH) so a too-tight grid can't bleed. High-fee markets structurally want WIDER grids (1.5% beat 1% on TH).",
            "WALK-FORWARD VERDICT (the honest test): out-of-sample the grid LOST to buy & hold — BTCUSDT -19% vs +23%, BTCTHB -27% vs +6%. Optimizing params did worse than a fixed 1% grid = overfit. The in-sample +18% edge was hindsight (early cheap-BTC accumulation + lucky endpoint), not a repeatable edge.",
            "THE WHEEL (options income): selling puts+calls yields ~35%/yr GROSS premium and lower drawdown (-38% vs -48%), but net total return +21% lagged holding's +49% in this bull run. Real cashcow yield is ~8%/yr net, and it caps upside. It wins only in flat/sideways markets.",
            "OVERALL: across grid, core+satellite, and the Wheel, NOTHING beat simply holding spot BTC over this trending period. Every strategy that caps upside loses to holding a rising asset. These are defensive/sideways tools, not bull-market tools.",
        ],
    }
    with open("results.json", "w") as f:
        json.dump(out, f)
    print("wrote results.json")


if __name__ == "__main__":
    main()
