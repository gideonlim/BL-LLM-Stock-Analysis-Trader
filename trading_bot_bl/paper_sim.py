"""Paper trading simulator — forward-tests signals without a broker.

Runs daily after signal generation.  Maintains a virtual portfolio
in a JSON state file, enters positions from BUY signals at the
current market price (yfinance), and exits when stop-loss / take-
profit is hit or holding period expires.

No broker connection required — all prices come from yfinance.

Usage:
    python -m trading_bot_bl.paper_sim --config configs/lse.json
    python -m trading_bot_bl.paper_sim --config configs/tse.json
    python -m trading_bot_bl.paper_sim --config configs/lse.json --reset
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_STARTING_EQUITY = 250_000.0  # 50% of $500k paper account
DEFAULT_MAX_POSITIONS = 6
DEFAULT_MIN_COMPOSITE_SCORE = 15.0


# ── Data models ──────────────────────────────────────────────────


@dataclass
class SimPosition:
    ticker: str
    entry_date: str
    entry_price: float
    qty: float
    notional: float
    stop_loss: float
    take_profit: float
    strategy: str
    composite_score: float
    holding_days: int = 0


@dataclass
class SimClosedTrade:
    ticker: str
    strategy: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "stop_loss" | "take_profit" | "max_hold" | "exit_signal"


@dataclass
class SimSnapshot:
    date: str
    equity: float
    cash: float
    num_positions: int
    total_pnl: float
    total_pnl_pct: float


@dataclass
class SimState:
    """Full paper simulation state — persisted as JSON."""

    starting_equity: float = DEFAULT_STARTING_EQUITY
    cash: float = DEFAULT_STARTING_EQUITY
    positions: list[SimPosition] = field(default_factory=list)
    closed_trades: list[SimClosedTrade] = field(default_factory=list)
    equity_history: list[SimSnapshot] = field(default_factory=list)
    last_run_date: str = ""

    def to_dict(self) -> dict:
        return {
            "starting_equity": self.starting_equity,
            "cash": self.cash,
            "positions": [asdict(p) for p in self.positions],
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "equity_history": [asdict(s) for s in self.equity_history],
            "last_run_date": self.last_run_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SimState:
        state = cls(
            starting_equity=d.get(
                "starting_equity", DEFAULT_STARTING_EQUITY
            ),
            cash=d.get("cash", DEFAULT_STARTING_EQUITY),
            last_run_date=d.get("last_run_date", ""),
        )
        for p in d.get("positions", []):
            state.positions.append(SimPosition(**p))
        for t in d.get("closed_trades", []):
            state.closed_trades.append(SimClosedTrade(**t))
        for s in d.get("equity_history", []):
            state.equity_history.append(SimSnapshot(**s))
        return state


# ── Price fetching ───────────────────────────────────────────────


def _fetch_latest_prices(
    tickers: list[str],
) -> dict[str, float]:
    """Fetch latest close prices for tickers via yfinance."""
    import yfinance as yf

    if not tickers:
        return {}
    prices: dict[str, float] = {}
    try:
        data = yf.download(
            tickers,
            period="5d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if data.empty:
            return prices
        close = data["Close"]
        for t in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker: Close is a plain Series
                    series = close.dropna()
                else:
                    series = close[t].dropna()
                if not series.empty:
                    prices[t] = float(series.iloc[-1])
            except (KeyError, IndexError, TypeError):
                log.warning(f"  No price data for {t}")
    except Exception as e:
        log.warning(f"  yfinance batch download failed: {e}")
    return prices


# ── Core simulation logic ────────────────────────────────────────


def _close_position(
    state: SimState,
    pos: SimPosition,
    exit_price: float,
    exit_reason: str,
    today: str,
) -> None:
    """Move a position to closed_trades and return cash."""
    pnl = (exit_price - pos.entry_price) * pos.qty
    pnl_pct = (
        (exit_price / pos.entry_price - 1) * 100
        if pos.entry_price != 0
        else 0.0
    )
    state.closed_trades.append(
        SimClosedTrade(
            ticker=pos.ticker,
            strategy=pos.strategy,
            entry_date=pos.entry_date,
            exit_date=today,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
        )
    )
    state.cash += pos.notional + pnl
    log.info(
        f"  CLOSED {pos.ticker}: {exit_reason} @ {exit_price:.2f} "
        f"(entry {pos.entry_price:.2f}, P&L {pnl:+.2f} / "
        f"{pnl_pct:+.1f}%)"
    )


def run_paper_sim(
    signals_dir: Path,
    state_path: Path,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    min_composite: float = DEFAULT_MIN_COMPOSITE_SCORE,
    max_hold_days: int = 10,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
) -> SimState:
    """Run one day of paper simulation.

    1. Load state from disk (or initialise fresh).
    2. Fetch latest prices for open positions.
    3. Check stop-loss / take-profit / max-hold exits.
    4. Load today's signals and enter new BUY positions.
    5. Record equity snapshot.
    6. Save state.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Load or initialise state ─────────────────────────────────
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state = SimState.from_dict(json.load(f))
        log.info(f"Loaded sim state: {len(state.positions)} positions, "
                 f"cash={state.cash:,.0f}")
    else:
        state = SimState(
            starting_equity=starting_equity,
            cash=starting_equity,
        )
        log.info(f"Initialised fresh sim: equity={starting_equity:,.0f}")

    # Guard against running twice on the same day
    if state.last_run_date == today:
        log.info(f"Already ran today ({today}) — skipping.")
        return state

    # ── Fetch prices for open positions ──────────────────────────
    held_tickers = [p.ticker for p in state.positions]
    prices = _fetch_latest_prices(held_tickers) if held_tickers else {}

    # ── Check exits on existing positions ────────────────────────
    to_close: list[tuple[SimPosition, float, str]] = []
    for pos in state.positions:
        pos.holding_days += 1
        price = prices.get(pos.ticker)
        if price is None:
            log.warning(f"  No price for {pos.ticker} — keeping position")
            continue

        if price <= pos.stop_loss:
            to_close.append((pos, pos.stop_loss, "stop_loss"))
        elif price >= pos.take_profit:
            to_close.append((pos, pos.take_profit, "take_profit"))
        elif pos.holding_days >= max_hold_days:
            to_close.append((pos, price, "max_hold"))

    for pos, exit_price, reason in to_close:
        _close_position(state, pos, exit_price, reason, today)
        state.positions.remove(pos)

    # ── Load today's signals ─────────────────────────────────────
    from trading_bot_bl.executor import load_latest_signals

    signals = load_latest_signals(signals_dir)
    buys = [
        s
        for s in signals
        if s.signal_raw == 1
        and s.composite_score >= min_composite
    ]
    exits = [s for s in signals if s.signal_raw == -1]

    # ── Process EXIT signals ─────────────────────────────────────
    held_map = {p.ticker: p for p in state.positions}
    for sig in exits:
        if sig.ticker in held_map:
            pos = held_map[sig.ticker]
            price = prices.get(sig.ticker, sig.current_price)
            _close_position(state, pos, price, "exit_signal", today)
            state.positions.remove(pos)

    # ── Enter new BUY positions ──────────────────────────────────
    held_tickers_now = {p.ticker for p in state.positions}
    slots = max_positions - len(state.positions)

    if slots > 0 and buys:
        # Sort by composite score descending, take top N
        buys.sort(key=lambda s: -s.composite_score)
        candidates = [
            s for s in buys if s.ticker not in held_tickers_now
        ][:slots]

        # Equal-weight sizing across available slots
        if candidates:
            # Fetch entry prices for new candidates
            new_tickers = [s.ticker for s in candidates]
            new_prices = _fetch_latest_prices(new_tickers)

            per_position = state.cash / max(
                slots, len(candidates)
            )
            for sig in candidates:
                entry_price = new_prices.get(
                    sig.ticker, sig.current_price
                )
                if entry_price <= 0:
                    continue
                notional = min(per_position, state.cash * 0.95)
                if notional < 100:
                    log.info(f"  Insufficient cash for {sig.ticker}")
                    continue
                qty = notional / entry_price

                state.positions.append(
                    SimPosition(
                        ticker=sig.ticker,
                        entry_date=today,
                        entry_price=round(entry_price, 4),
                        qty=round(qty, 4),
                        notional=round(notional, 2),
                        stop_loss=round(sig.stop_loss_price, 4),
                        take_profit=round(sig.take_profit_price, 4),
                        strategy=sig.strategy,
                        composite_score=sig.composite_score,
                    )
                )
                state.cash -= notional
                log.info(
                    f"  ENTERED {sig.ticker}: {sig.strategy} "
                    f"@ {entry_price:.2f}, "
                    f"SL={sig.stop_loss_price:.2f}, "
                    f"TP={sig.take_profit_price:.2f}, "
                    f"size={notional:,.0f}"
                )

    # ── Equity snapshot ──────────────────────────────────────────
    # Refresh prices for all current positions (including new ones)
    all_tickers = [p.ticker for p in state.positions]
    if all_tickers:
        all_prices = _fetch_latest_prices(all_tickers)
    else:
        all_prices = {}

    market_value = sum(
        all_prices.get(p.ticker, p.entry_price) * p.qty
        for p in state.positions
    )
    equity = state.cash + market_value
    total_pnl = equity - state.starting_equity
    total_pnl_pct = (
        (equity / state.starting_equity - 1) * 100
        if state.starting_equity != 0
        else 0.0
    )

    state.equity_history.append(
        SimSnapshot(
            date=today,
            equity=round(equity, 2),
            cash=round(state.cash, 2),
            num_positions=len(state.positions),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
        )
    )
    state.last_run_date = today

    # ── Save state ───────────────────────────────────────────────
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    log.info(f"Saved state to {state_path}")

    # ── Summary ──────────────────────────────────────────────────
    log.info(f"{'=' * 50}")
    log.info(f"  Paper Sim Summary — {today}")
    log.info(f"  Equity:    {equity:>12,.2f}")
    log.info(f"  Cash:      {state.cash:>12,.2f}")
    log.info(f"  Positions: {len(state.positions)}")
    log.info(f"  Total P&L: {total_pnl:>+12,.2f} ({total_pnl_pct:+.2f}%)")
    log.info(f"  Closed trades: {len(state.closed_trades)}")
    log.info(f"{'=' * 50}")

    for p in state.positions:
        cur = all_prices.get(p.ticker, p.entry_price)
        unrealised = (cur - p.entry_price) * p.qty
        pct = (cur / p.entry_price - 1) * 100 if p.entry_price else 0
        log.info(
            f"    {p.ticker:>8}  {p.strategy:<25} "
            f"entry={p.entry_price:.2f}  now={cur:.2f}  "
            f"P&L={unrealised:+,.0f} ({pct:+.1f}%)  "
            f"day {p.holding_days}"
        )

    return state


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trading simulator (no broker needed)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to market config (e.g. configs/lse.json)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset simulation state (start fresh)",
    )
    args = parser.parse_args()

    # Load config to get market-scoped paths
    from trading_bot_bl.config import TradingConfig

    config = TradingConfig.from_file(Path(args.config))
    market = config.get_market()

    signals_dir = config.path_for("signals")
    state_dir = config.path_for("execution_logs")
    state_path = state_dir / "paper_sim_state.json"

    if args.reset and state_path.exists():
        state_path.unlink()
        log.info("Reset: deleted existing sim state.")

    # Read risk limits from config
    max_positions = config.risk.max_positions
    min_composite = config.risk.min_composite_score
    max_hold = config.risk.max_hold_days

    # Starting equity = total equity * allocation fraction
    ibkr_alloc = config.ibkr.max_equity_allocation
    starting_eq = DEFAULT_STARTING_EQUITY * ibkr_alloc / 0.5
    # The default 250k assumes 50% allocation of 500k.
    # If allocation is 0.5 (50%), starting_eq = 250k.

    log.info(f"Market: {market.market_id} ({market.currency})")
    log.info(f"Signals: {signals_dir}")
    log.info(f"State: {state_path}")

    run_paper_sim(
        signals_dir=signals_dir,
        state_path=state_path,
        max_positions=max_positions,
        min_composite=min_composite,
        max_hold_days=max_hold,
        starting_equity=starting_eq,
    )


if __name__ == "__main__":
    main()
