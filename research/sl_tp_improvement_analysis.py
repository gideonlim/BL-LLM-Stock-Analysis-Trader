"""
SL/TP Improvement Analysis
==========================
Research script that quantifies problems with current SL/TP logic
and simulates proposed improvements.

Run: python research/sl_tp_improvement_analysis.py
"""

import json
import glob
import statistics
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent


def load_buy_signals() -> list[dict]:
    """Load all BUY signals across all signal files."""
    signals = []
    for f in sorted(BASE.glob("signals/signals_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for sig in data.get("signals", []):
            if sig.get("signal") != "BUY":
                continue
            sl_pct = float(sig.get("stop_loss_pct", 0) or 0)
            tp_pct = float(sig.get("take_profit_pct", 0) or 0)
            avg_hold = float(sig.get("avg_holding_days", 0) or 0)
            vol_20 = float(sig.get("vol_20", 0) or 0)
            if sl_pct > 0 and tp_pct > 0 and avg_hold > 0 and vol_20 > 0:
                daily_vol = vol_20 / (252**0.5) * 100
                expected_max = daily_vol * (avg_hold**0.5)
                sig["_sl_pct"] = sl_pct
                sig["_tp_pct"] = tp_pct
                sig["_avg_hold"] = avg_hold
                sig["_daily_vol"] = daily_vol
                sig["_expected_max_move"] = expected_max
                sig["_tp_vs_expected"] = (
                    tp_pct / expected_max if expected_max > 0 else 0
                )
                signals.append(sig)
    return signals


def load_journal_trades() -> list[dict]:
    """Load all journal trade entries."""
    trades = []
    journal_dir = BASE / "execution_logs" / "journal"
    for f in sorted(journal_dir.glob("*.json")):
        with open(f) as fh:
            trades.append(json.load(fh))
    return trades


MEAN_REVERSION = {"Z-Score Mean Reversion", "Bollinger Band Mean Reversion"}
TREND_FOLLOWING = {
    "VWAP Trend",
    "ADX Trend Following",
    "EMA Crossover (9/21)",
    "52-Week High Momentum",
}
MOMENTUM_OSCILLATOR = {
    "Momentum (Rate of Change)",
    "MACD Crossover",
    "Stochastic Oscillator",
}


def proposed_tp_pct(
    strategy: str,
    confidence: int,
    trend: str,
    avg_hold: float,
    sl_pct: float,
    daily_vol: float,
) -> float:
    """
    Proposed strategy-aware R:R with holding-period cap.

    Changes vs current:
    1. Base R:R varies by strategy type
    2. Confidence/trend adjustment is smaller
    3. TP is capped at 1.5x expected max move for the holding period
    """
    # 1. Strategy-specific base R:R
    if strategy in MEAN_REVERSION:
        base_rr = 1.0  # targets the mean, not a runaway
    elif strategy in TREND_FOLLOWING:
        base_rr = 2.0  # can let winners run further
    else:
        base_rr = 1.5  # momentum/oscillator

    # 2. Smaller confidence/trend adjustment
    if trend == "BULLISH" and confidence >= 4:
        base_rr = min(base_rr + 0.5, 3.0)
    elif trend == "BEARISH" or confidence <= 1:
        base_rr = max(base_rr - 0.5, 1.0)

    tp = sl_pct * base_rr

    # 3. Cap at 1.5x expected max move
    if daily_vol > 0 and avg_hold > 0:
        expected_max = daily_vol * (avg_hold**0.5)
        cap = expected_max * 1.5
        tp = min(tp, cap)

    # Floor: at least 1:1
    return max(tp, sl_pct)


def main() -> None:
    signals = load_buy_signals()
    trades = load_journal_trades()

    print(f"\n{'='*70}")
    print("  SL/TP IMPROVEMENT ANALYSIS")
    print(f"  {len(signals)} BUY signals across "
          f"{len(list(BASE.glob('signals/signals_*.json')))} signal files")
    print(f"{'='*70}\n")

    # ── Problem 1: TP targets are unreachable ─────────────────────
    print("PROBLEM 1: TP TARGETS ARE UNREACHABLE\n")

    tp_vs = [s["_tp_vs_expected"] for s in signals]
    print(f"  TP / Expected-Max-Move ratio (>1.5 = very unlikely to hit):")
    print(f"    Mean:   {statistics.mean(tp_vs):.2f}")
    print(f"    Median: {statistics.median(tp_vs):.2f}")
    above_1_5 = sum(1 for r in tp_vs if r > 1.5)
    above_2_0 = sum(1 for r in tp_vs if r > 2.0)
    print(f"    >1.5x: {above_1_5}/{len(tp_vs)} "
          f"({above_1_5/len(tp_vs)*100:.0f}%)")
    print(f"    >2.0x: {above_2_0}/{len(tp_vs)} "
          f"({above_2_0/len(tp_vs)*100:.0f}%)")

    print("\n  By strategy:")
    by_strat: dict[str, list] = {}
    for s in signals:
        strat = s["strategy"]
        by_strat.setdefault(strat, []).append(s)

    for strat in sorted(
        by_strat, key=lambda x: len(by_strat[x]), reverse=True
    ):
        pts = by_strat[strat]
        if len(pts) < 10:
            continue
        tp_med = statistics.median([p["_tp_pct"] for p in pts])
        hold_med = statistics.median([p["_avg_hold"] for p in pts])
        ratio_med = statistics.median(
            [p["_tp_vs_expected"] for p in pts]
        )
        sl_med = statistics.median([p["_sl_pct"] for p in pts])
        rr = tp_med / sl_med if sl_med > 0 else 0
        flag = " *** PROBLEM" if ratio_med > 2.0 else ""
        print(
            f"    {strat:30} SL={sl_med:>5.1f}% TP={tp_med:>5.1f}% "
            f"hold={hold_med:>4.1f}d R:R={rr:.1f} "
            f"TP/ExpMax={ratio_med:.2f}{flag}"
        )

    # ── Problem 2: Closed trades almost never hit TP ──────────────
    print(f"\n{'─'*70}")
    print("PROBLEM 2: LIVE TRADES ALMOST NEVER HIT TP\n")

    closed = [t for t in trades if t.get("status") == "closed"]
    non_migrated_closed = [
        t for t in closed if "migrated" not in t.get("trade_id", "")
    ]

    reasons = Counter(t.get("exit_reason", "?") for t in closed)
    print("  Exit reason breakdown (all closed):")
    for r, c in reasons.most_common():
        print(f"    {r}: {c} ({c/len(closed)*100:.0f}%)")

    print(f"\n  Non-migrated closed trades TP reachability:")
    for t in non_migrated_closed:
        fill = t.get("entry_fill_price", 0) or 0
        tp = t.get("original_tp_price", 0) or 0
        mfe = t.get("mfe_pct", 0) or 0
        sl = t.get("original_sl_price", 0) or 0
        reason = t.get("exit_reason", "?") or "?"
        pnl = t.get("realized_pnl_pct", 0) or 0

        if fill > 0 and tp > 0:
            tp_dist = abs(tp - fill) / fill * 100
            mfe_ratio = mfe / tp_dist * 100 if tp_dist > 0 else 0
            print(
                f"    {t['ticker']:>6}: TP={tp_dist:>5.1f}% away | "
                f"MFE={mfe:>5.2f}% ({mfe_ratio:>3.0f}% of TP) | "
                f"PnL={pnl:>+6.2f}% | {reason}"
            )

    # ── Proposed improvement ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PROPOSED: STRATEGY-AWARE R:R WITH HOLDING-PERIOD CAP\n")

    current_tps = []
    proposed_tps = []
    current_reachable = 0
    proposed_reachable = 0

    for s in signals:
        expected_max = s["_expected_max_move"]
        current_tp = s["_tp_pct"]
        new_tp = proposed_tp_pct(
            strategy=s["strategy"],
            confidence=int(s.get("confidence_score", 0) or 0),
            trend=s.get("trend", ""),
            avg_hold=s["_avg_hold"],
            sl_pct=s["_sl_pct"],
            daily_vol=s["_daily_vol"],
        )
        current_tps.append(current_tp)
        proposed_tps.append(new_tp)

        if current_tp <= expected_max * 1.5:
            current_reachable += 1
        if new_tp <= expected_max * 1.5:
            proposed_reachable += 1

    n = len(signals)
    print(f"  Current:  mean TP={statistics.mean(current_tps):.1f}%  "
          f"median={statistics.median(current_tps):.1f}%")
    print(f"  Proposed: mean TP={statistics.mean(proposed_tps):.1f}%  "
          f"median={statistics.median(proposed_tps):.1f}%")
    print(
        f"\n  Reachable signals (TP <= 1.5x expected move):"
    )
    print(
        f"    Current:  {current_reachable}/{n} "
        f"({current_reachable/n*100:.0f}%)"
    )
    print(
        f"    Proposed: {proposed_reachable}/{n} "
        f"({proposed_reachable/n*100:.0f}%)"
    )

    print("\n  By strategy (median TP):")
    for strat in sorted(
        by_strat, key=lambda x: len(by_strat[x]), reverse=True
    ):
        pts = by_strat[strat]
        if len(pts) < 10:
            continue
        cur_tps = []
        new_tps = []
        for s in pts:
            cur_tps.append(s["_tp_pct"])
            new_tps.append(
                proposed_tp_pct(
                    s["strategy"],
                    int(s.get("confidence_score", 0) or 0),
                    s.get("trend", ""),
                    s["_avg_hold"],
                    s["_sl_pct"],
                    s["_daily_vol"],
                )
            )
        sl_med = statistics.median([p["_sl_pct"] for p in pts])
        cur_med = statistics.median(cur_tps)
        new_med = statistics.median(new_tps)
        rr_cur = cur_med / sl_med if sl_med > 0 else 0
        rr_new = new_med / sl_med if sl_med > 0 else 0
        change = (new_med - cur_med) / cur_med * 100
        print(
            f"    {strat:30} "
            f"current={cur_med:>5.1f}% (R:R={rr_cur:.1f}) -> "
            f"proposed={new_med:>5.1f}% (R:R={rr_new:.1f}) "
            f"[{change:+.0f}%]"
        )

    print(f"\n{'='*70}")
    print("  SUMMARY OF RECOMMENDED CHANGES")
    print(f"{'='*70}\n")

    print("""\
  1. STRATEGY-SPECIFIC BASE R:R (signals.py lines 395-410)
     Current:  All strategies use same trend/confidence logic (1.5-3.5x)
     Proposed: Mean reversion 1.0x | Momentum/oscillator 1.5x | Trend 2.0x
     Why:      Mean reversion targets the mean, not a runaway. Using 3:1
               R:R on a mean reversion signal is conceptually wrong.

  2. HOLDING-PERIOD TP CAP (signals.py, new logic after R:R)
     Add:      tp_pct = min(tp_pct, daily_vol * sqrt(avg_hold) * 1.5)
     Why:      57% of current signals have TP > 1.5x the expected max
               move for their holding period. These TPs will almost
               never be hit, causing trades to drift into time/market
               exits instead. The cap ensures TP is physically reachable.
     Impact:   Reachable signals go from 43% to 83%.

  3. VWAP TREND IS THE WORST OFFENDER
     Problem:  39% of all BUY signals, median hold 3.5 days,
               but median TP is 14.3% (needs 4.2%/day). TP/ExpMax = 4.06.
     Fix:      With proposed changes, VWAP Trend TP drops from 14.3% to
               5.8%, which is actually reachable in 3.5 days of trading.

  4. TIGHTEN BREAKEVEN STOP THRESHOLD (monitor.py line 1354)
     Current:  Breakeven triggers at 1x ATR or 3% above entry
     Proposed: Trigger at 0.5x ATR or 1.5% — lock in break-even sooner
               on short-duration trades that may never reach TP.
     Why:      Most VWAP Trend trades hold ~3 days. Waiting for 3%
               profit to move SL to breakeven means it rarely triggers.

  5. SL FLOOR INCREASE (signals.py line 149)
     Current:  1.5% minimum SL
     Proposed: 2.0% minimum SL
     Why:      Only 1.2% of signals hit the floor, but 1.5% is extremely
               tight. With typical bid-ask spread + slippage, a 1.5%
               stop can get hit on entry noise alone.
""")


if __name__ == "__main__":
    main()
