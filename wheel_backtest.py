"""
"The Wheel" options-income backtest on BTC — the cashcow candidate.

Strategy (premium SELLING, the income side of options):
  CASH state : sell a cash-secured PUT ~OTM_PUT below spot, collect premium.
               at expiry  -> spot above strike: keep premium, sell another put.
                          -> spot below strike: ASSIGNED, buy BTC at strike -> COIN.
  COIN state : sell a covered CALL ~OTM_CALL above spot, collect premium.
               at expiry  -> spot below strike: keep premium + BTC, sell another.
                          -> spot above strike: BTC CALLED AWAY at strike -> CASH.
  Repeat. You collect premium at every step — that's the income engine.

HONEST LIMITATION (read this):
  Free historical BTC *option* prices aren't available, so premiums here are
  MODELED with Black-Scholes, not real fills. The critical input is implied
  volatility (IV) — the price you actually get paid. Real BTC IV usually runs
  ABOVE realized volatility (the "variance risk premium" — that gap is literally
  the seller's edge). We model IV = IV_PREMIUM * trailing realized vol. If your
  real IV is higher, income is higher; if lower, lower. Treat the level as an
  estimate; treat the SHAPE (income smooth, upside capped) as the real lesson.
  European-style settlement at expiry (no early assignment, no intra-period path).

Run:  python wheel_backtest.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import grid_backtest as gb


# ---- Black-Scholes (r = 0, crypto convention) -----------------------------
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * _ncdf(d2)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


# ---- realized volatility (annualized), trailing, no lookahead -------------
def trailing_vol(close, window):
    logret = np.diff(np.log(close))
    vol = np.full(len(close), np.nan)
    for i in range(window, len(close)):
        vol[i] = np.std(logret[i - window:i]) * math.sqrt(365)
    return vol


def run_wheel(df, capital, otm_put, otm_call, expiry_days, iv_premium,
              vol_window, fee):
    close = df["close"].values
    vol = trailing_vol(close, vol_window)
    T = expiry_days / 365.0

    cash = capital
    btc = 0.0
    state = "CASH"
    prem_total = 0.0
    puts = puts_assigned = calls = calls_away = 0

    equity_t, equity_v = [], []
    i = vol_window + 1                       # warmup for trailing vol
    while i + expiry_days < len(close):
        S = close[i]
        sigma = vol[i]
        if np.isnan(sigma) or sigma <= 0:
            i += expiry_days; continue
        iv = sigma * iv_premium              # modeled implied vol
        Sx = close[i + expiry_days]          # spot at expiry

        if state == "CASH":
            K = S * (1 - otm_put)
            q = cash / K                     # cash-secured: lock cash to buy q at K
            prem = q * bs_price(S, K, T, iv, "put") * (1 - fee)
            cash += prem; prem_total += prem; puts += 1
            if Sx <= K:                      # assigned -> buy BTC at strike
                btc = q; cash -= q * K; state = "COIN"; puts_assigned += 1
        else:  # COIN
            K = S * (1 + otm_call)
            prem = btc * bs_price(S, K, T, iv, "call") * (1 - fee)
            cash += prem; prem_total += prem; calls += 1
            if Sx >= K:                      # called away -> sell BTC at strike
                cash += btc * K; btc = 0.0; state = "CASH"; calls_away += 1

        equity_t.append(df["dt"].iloc[i + expiry_days])
        equity_v.append(cash + btc * Sx)
        i += expiry_days

    last = close[-1]
    final = cash + btc * last
    start_price = close[vol_window + 1]
    hodl = capital / start_price * last
    years = (df["dt"].iloc[-1] - df["dt"].iloc[vol_window + 1]).days / 365.0
    return {
        "final": final, "ret_pct": (final / capital - 1) * 100,
        "hodl_pct": (hodl / capital - 1) * 100,
        "prem_total": prem_total, "prem_yield_pct": prem_total / capital / years * 100,
        "puts": puts, "puts_assigned": puts_assigned,
        "calls": calls, "calls_away": calls_away, "years": years,
        "eq_t": equity_t, "eq_v": equity_v,
        "hodl_curve": [capital / start_price * close[j] for j in
                       range(vol_window + 1 + expiry_days, len(close), expiry_days)],
    }


def max_dd(curve):
    curve = np.asarray(curve, float)
    peak = np.maximum.accumulate(curve)
    return float(((curve - peak) / peak).min() * 100)


def main():
    CAP = 10_000.0
    OTM_PUT = 0.10
    OTM_CALL = 0.10
    EXPIRY_DAYS = 7
    IV_PREMIUM = 1.3        # IV = 1.3x realized vol (variance risk premium)
    VOL_WINDOW = 30
    FEE = 0.0005

    df = gb.fetch_klines("https://api.binance.com", "/api/v3/klines",
                         "BTCUSDT", "1d", gb.to_ms("1 Jan 2024"))
    df, _ = gb.clean_klines(df)

    print(f"{'='*66}\nTHE WHEEL on BTCUSDT  (1d, {EXPIRY_DAYS}d expiries, "
          f"{int(OTM_PUT*100)}% OTM, IV={IV_PREMIUM}x realized)\n{'='*66}")
    r = run_wheel(df, CAP, OTM_PUT, OTM_CALL, EXPIRY_DAYS, IV_PREMIUM, VOL_WINDOW, FEE)
    print(f"  period         : {r['years']:.2f} years")
    print(f"  puts sold      : {r['puts']}   (assigned {r['puts_assigned']})")
    print(f"  calls sold     : {r['calls']}   (called away {r['calls_away']})")
    print(f"  premium income : {r['prem_total']:,.0f}  =>  {r['prem_yield_pct']:.1f}%/yr gross yield")
    print(f"  wheel total ret: {r['ret_pct']:+.1f}%")
    print(f"  buy & hold     : {r['hodl_pct']:+.1f}%")
    print(f"  edge vs HODL   : {r['ret_pct']-r['hodl_pct']:+.1f}%")
    print(f"  wheel max DD   : {max_dd(r['eq_v']):.1f}%   "
          f"HODL max DD: {max_dd(r['hodl_curve']):.1f}%")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r["eq_t"], r["hodl_curve"], color="#8b949e", lw=1.4, label=f"buy & hold ({r['hodl_pct']:+.0f}%)")
    ax.plot(r["eq_t"], r["eq_v"], color="#d62728", lw=1.8, label=f"the wheel ({r['ret_pct']:+.0f}%)")
    ax.axhline(CAP, color="#000", lw=0.6, ls=":")
    ax.set_title("The Wheel (options income) vs buy & hold — BTC", fontweight="bold")
    ax.set_ylabel("portfolio value"); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig("chart_wheel.png", dpi=110)
    print("\n  chart -> chart_wheel.png")


if __name__ == "__main__":
    main()
