"""
Walk-forward validation for the grid strategy.

Why: every number so far was measured on the SAME window we eyeballed — that's
in-sample, and in-sample results lie. Walk-forward fixes this:

  train window  ->  pick the best params (spacing, core_fraction) on THIS data only
  test window   ->  apply those params to the NEXT, UNSEEN slice; record the result
  roll forward  ->  repeat; the grid RANGE is also derived from train only (no peek)

Concatenating every test slice gives an out-of-sample (OOS) equity curve — what
you'd actually have earned choosing params with no knowledge of the future.

The honest question it answers: does tuning the grid add real edge, or does a
dumb fixed 1% grid do just as well OOS? If optimization only wins in-sample,
that's overfitting, and we should know before risking money.

Run:  python walkforward.py        (writes chart_walkforward.png + walkforward.json)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grid_backtest as gb

BARS_PER_DAY = 96            # 15m candles
TRAIN_DAYS = 180
TEST_DAYS = 60
SPACINGS = [0.005, 0.0075, 0.01, 0.015, 0.02]
CORES = [0.0, 0.3, 0.6]


def grid_range(window):
    """Derive the grid bounds from a price window (train only — no lookahead)."""
    return window["close"].min() * 0.98, window["close"].max() * 1.02


def eval_params(window, lower, upper, ratio, core, capital, fee, slip, floor):
    eff = max(ratio, floor)                      # fee-floor safety rail
    r = gb.run_grid(window, lower, upper, eff, capital, fee,
                    slippage=slip, core_fraction=core)
    return r["grid_return_pct"], r["buyhold_return_pct"], eff


def walk_forward(symbol, base, kpath, dpath, capital, fee):
    df = gb.fetch_klines(base, kpath, symbol, "15m", gb.to_ms("1 Jan 2024"))
    df, _ = gb.clean_klines(df)
    n_cells_guess = max(1, np.log(3) / np.log(1.01))
    slip, _ = gb.measure_slippage(base, dpath, symbol, capital / n_cells_guess)
    floor = 2 * (fee + slip) * 1.5

    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    folds = []
    i = 0
    while i + train_bars + test_bars <= len(df):
        train = df.iloc[i:i + train_bars].reset_index(drop=True)
        test = df.iloc[i + train_bars:i + train_bars + test_bars].reset_index(drop=True)
        lo, hi = grid_range(train)               # RANGE FROM TRAIN ONLY

        # --- optimize on train: pick params with best in-sample return ---
        best = None
        for ratio in SPACINGS:
            for core in CORES:
                tr_ret, _, eff = eval_params(train, lo, hi, ratio, core,
                                             capital, fee, slip, floor)
                if best is None or tr_ret > best["train_ret"]:
                    best = {"ratio": ratio, "core": core, "eff": eff, "train_ret": tr_ret}

        # --- apply chosen params to the UNSEEN test slice ---
        opt_ret, hodl_ret, _ = eval_params(test, lo, hi, best["ratio"], best["core"],
                                           capital, fee, slip, floor)
        # baseline: a dumb fixed 1% grid, no core, same fee floor
        base_ret, _, _ = eval_params(test, lo, hi, 0.01, 0.0,
                                     capital, fee, slip, floor)

        folds.append({
            "train_start": str(train["dt"].iloc[0].date()),
            "test_start": str(test["dt"].iloc[0].date()),
            "test_end": str(test["dt"].iloc[-1].date()),
            "chosen_spacing_pct": round(best["eff"] * 100, 3),
            "chosen_core": best["core"],
            "train_ret": round(best["train_ret"], 1),
            "opt_test_ret": round(opt_ret, 1),
            "fixed_test_ret": round(base_ret, 1),
            "hodl_test_ret": round(hodl_ret, 1),
        })
        i += test_bars                           # non-overlapping test windows

    return folds, floor, slip


def compound(rets):
    """Chain per-window % returns into a cumulative equity multiple (start=1)."""
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r / 100))
    return eq


def summarize(symbol, folds):
    opt = compound([f["opt_test_ret"] for f in folds])
    fix = compound([f["fixed_test_ret"] for f in folds])
    hod = compound([f["hodl_test_ret"] for f in folds])
    print(f"\n{'='*78}\n{symbol}  —  {len(folds)} out-of-sample folds "
          f"({TRAIN_DAYS}d train / {TEST_DAYS}d test)\n{'='*78}")
    print(f"{'test window':<24}{'params':<16}{'train':>8}{'OPT oos':>9}{'fixed':>8}{'hodl':>8}")
    for f in folds:
        p = f"{f['chosen_spacing_pct']}% c{f['chosen_core']}"
        print(f"{f['test_start']}->{f['test_end']:<12}{p:<16}"
              f"{f['train_ret']:>7.1f}%{f['opt_test_ret']:>8.1f}%"
              f"{f['fixed_test_ret']:>7.1f}%{f['hodl_test_ret']:>7.1f}%")
    print("-" * 78)
    print(f"{'COMPOUNDED OOS':<40}{'':>8}{(opt[-1]-1)*100:>8.1f}%"
          f"{(fix[-1]-1)*100:>7.1f}%{(hod[-1]-1)*100:>7.1f}%")
    return {"opt": opt, "fix": fix, "hod": hod}


def plot(results, out="chart_walkforward.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5), squeeze=False)
    for ax, (symbol, d) in zip(axes[0], results.items()):
        x = range(len(d["curves"]["opt"]))
        ax.plot(x, d["curves"]["hod"], color="#8b949e", lw=1.4, marker="o", ms=3, label="buy & hold")
        ax.plot(x, d["curves"]["fix"], color="#2ca02c", lw=1.4, marker="o", ms=3, label="fixed 1% grid")
        ax.plot(x, d["curves"]["opt"], color="#d62728", lw=1.8, marker="o", ms=3, label="optimized (OOS)")
        ax.axhline(1.0, color="#000", lw=0.6, ls=":")
        ax.set_title(f"{symbol} — walk-forward OOS equity", fontweight="bold")
        ax.set_xlabel("fold (sequential 60d test windows)")
        ax.set_ylabel("cumulative equity (x)")
        ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    markets = [
        ("BTCUSDT", "https://api.binance.com", "/api/v3/klines", "/api/v3/depth", 10_000.0, 0.0010),
        ("BTCTHB",  "https://api.binance.th",  "/api/v1/klines",  "/api/v1/depth",  350_000.0, 0.0025),
    ]
    results, dump = {}, {}
    for sym, base, kp, dp, cap, fee in markets:
        print(f"running walk-forward for {sym} ...")
        folds, floor, slip = walk_forward(sym, base, kp, dp, cap, fee)
        curves = summarize(sym, folds)
        results[sym] = {"curves": curves}
        dump[sym] = {"folds": folds, "fee_floor_pct": round(floor * 100, 3),
                     "compounded": {k: round((v[-1] - 1) * 100, 1) for k, v in curves.items()}}
    png = plot(results)
    with open("walkforward.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nchart -> {png}\njson  -> walkforward.json")


if __name__ == "__main__":
    main()
