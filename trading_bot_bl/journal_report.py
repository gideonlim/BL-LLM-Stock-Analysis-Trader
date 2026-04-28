"""Generate PDF performance reports with charts from journal data.

Uses matplotlib for chart generation and reportlab for PDF assembly.
All chart images are rendered in-memory (no temp files needed).
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from dataclasses import fields as dc_fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from trading_bot_bl.journal_analytics import (
    JournalMetrics,
    breakdown_by_trade_type,
    compute_journal_metrics,
)
from trading_bot_bl.models import EquitySnapshot, JournalEntry

log = logging.getLogger(__name__)


# ── Benchmark helpers ────────────────────────────────────────────


@dataclasses.dataclass
class BenchmarkStats:
    """Side-by-side performance stats for bot vs benchmark."""

    # Bot
    bot_cumulative_return: float = 0.0
    bot_annualized_return: float = 0.0
    bot_sharpe: float = 0.0
    bot_sortino: float = 0.0
    bot_max_drawdown: float = 0.0
    bot_volatility: float = 0.0
    # Benchmark
    bench_cumulative_return: float = 0.0
    bench_annualized_return: float = 0.0
    bench_sharpe: float = 0.0
    bench_sortino: float = 0.0
    bench_max_drawdown: float = 0.0
    bench_volatility: float = 0.0
    # Relative
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    correlation: float = 0.0


def _fetch_spy_prices(
    start_date: datetime,
    end_date: datetime,
) -> Optional[list[tuple[datetime, float]]]:
    """Fetch SPY daily close prices for the given date range.

    Returns list of (date, price) sorted by date, or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — cannot fetch SPY benchmark")
        return None

    try:
        # Pad start by a few days to ensure we get data on/before start_date
        padded_start = start_date - timedelta(days=5)
        data = yf.download(
            "SPY",
            start=padded_start.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            log.warning("SPY download returned empty data")
            return None

        prices: list[tuple[datetime, float]] = []
        for idx, row in data.iterrows():
            ts = idx.to_pydatetime()  # type: ignore[union-attr]
            close = float(row["Close"].iloc[0])
            prices.append((ts, close))
        prices.sort(key=lambda x: x[0])
        return prices
    except Exception as e:
        log.warning(f"Could not fetch SPY benchmark: {e}")
        return None


def _compute_benchmark_stats(
    bot_dates: list[datetime],
    bot_equity: list[float],
    spy_prices: list[tuple[datetime, float]],
    bot_sharpe_override: Optional[float] = None,
    bot_sortino_override: Optional[float] = None,
) -> Optional[BenchmarkStats]:
    """Compute side-by-side bot vs SPY statistics.

    Aligns bot equity to SPY by calendar date so both series
    cover the same time window.  Bot snapshots on non-trading
    days are carried forward to the next SPY trading day.
    """
    if len(bot_dates) < 3 or len(spy_prices) < 3:
        return None

    # Index bot equity by date (last snapshot per day wins).
    # Filter to weekdays only — weekend snapshots contain stale or
    # after-hours equity values that don't correspond to any SPY
    # trading day, introducing noise into the correlation.
    bot_by_date: dict[str, float] = {}
    for dt, eq in zip(bot_dates, bot_equity):
        if dt.weekday() < 5:  # Mon-Fri only
            bot_by_date[dt.strftime("%Y-%m-%d")] = eq

    # Index SPY prices by date
    spy_by_date: dict[str, float] = {}
    for dt, price in spy_prices:
        spy_by_date[dt.strftime("%Y-%m-%d")] = price

    # Align bot equity to SPY trading days.  Prefer exact date
    # matches; carry forward only when the bot missed a weekday
    # (e.g. didn't run on a trading day).
    spy_dates_sorted = sorted(spy_by_date.keys())
    bot_dates_sorted = sorted(bot_by_date.keys())

    aligned_bot: list[float] = []
    aligned_spy: list[float] = []
    last_bot_eq: Optional[float] = None

    all_dates = sorted(set(bot_dates_sorted + spy_dates_sorted))
    for d in all_dates:
        if d in bot_by_date:
            last_bot_eq = bot_by_date[d]
        if d in spy_by_date and last_bot_eq is not None:
            aligned_bot.append(last_bot_eq)
            aligned_spy.append(spy_by_date[d])

    if len(aligned_bot) < 3:
        return None

    # Compute daily returns from the aligned equity/price series
    bot_returns: list[float] = []
    spy_returns: list[float] = []
    for i in range(1, len(aligned_bot)):
        if aligned_bot[i - 1] != 0:
            bot_returns.append(
                aligned_bot[i] / aligned_bot[i - 1] - 1
            )
        else:
            bot_returns.append(0.0)
        if aligned_spy[i - 1] != 0:
            spy_returns.append(
                aligned_spy[i] / aligned_spy[i - 1] - 1
            )
        else:
            spy_returns.append(0.0)

    if len(bot_returns) < 2:
        return None

    bot_r = np.array(bot_returns)
    spy_r = np.array(spy_returns)

    trading_days = 252

    def _annualized_return(returns: np.ndarray) -> float:
        cum = np.prod(1 + returns) - 1
        n_days = len(returns)
        if n_days == 0:
            return 0.0
        return float((1 + cum) ** (trading_days / n_days) - 1)

    def _cumulative_return(returns: np.ndarray) -> float:
        return float(np.prod(1 + returns) - 1)

    def _sharpe(returns: np.ndarray) -> float:
        std = returns.std(ddof=1) if len(returns) > 1 else 0.0
        if std == 0:
            return 0.0
        return float(returns.mean() / std * np.sqrt(trading_days))

    def _sortino(returns: np.ndarray) -> float:
        # TDD per Sortino & Price (1994): all observations,
        # positives clamped to zero.
        downside_diff = np.minimum(returns, 0.0)
        tdd = float(np.sqrt(np.mean(downside_diff ** 2)))
        if tdd == 0:
            return 0.0
        return float(returns.mean() / tdd * np.sqrt(trading_days))

    def _max_dd(returns: np.ndarray) -> float:
        cum = np.cumprod(1 + returns)
        hwm = np.maximum.accumulate(cum)
        dd = (cum - hwm) / hwm
        return float(dd.min()) if len(dd) > 0 else 0.0

    def _volatility(returns: np.ndarray) -> float:
        std = returns.std(ddof=1) if len(returns) > 1 else 0.0
        return float(std * np.sqrt(trading_days))

    # Beta & alpha (CAPM)
    cov = np.cov(bot_r, spy_r)
    spy_var = cov[1, 1]
    beta = float(cov[0, 1] / spy_var) if spy_var != 0 else 0.0
    bot_ann = _annualized_return(bot_r)
    spy_ann = _annualized_return(spy_r)
    alpha = bot_ann - beta * spy_ann  # simplified (risk-free ≈ 0)

    # Information ratio
    excess = bot_r - spy_r
    ir = (
        float(excess.mean() / excess.std() * np.sqrt(trading_days))
        if excess.std() != 0
        else 0.0
    )

    corr_matrix = np.corrcoef(bot_r, spy_r)
    corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0

    return BenchmarkStats(
        bot_cumulative_return=_cumulative_return(bot_r),
        bot_annualized_return=bot_ann,
        bot_sharpe=(bot_sharpe_override
                    if bot_sharpe_override is not None
                    else _sharpe(bot_r)),
        bot_sortino=(bot_sortino_override
                     if bot_sortino_override is not None
                     else _sortino(bot_r)),
        bot_max_drawdown=_max_dd(bot_r),
        bot_volatility=_volatility(bot_r),
        bench_cumulative_return=_cumulative_return(spy_r),
        bench_annualized_return=spy_ann,
        bench_sharpe=_sharpe(spy_r),
        bench_sortino=_sortino(spy_r),
        bench_max_drawdown=_max_dd(spy_r),
        bench_volatility=_volatility(spy_r),
        alpha=alpha,
        beta=beta,
        information_ratio=ir,
        correlation=corr,
    )


# ── Data loading ─────────────────────────────────────────────────


def _load_journal_data(
    log_dir: Path,
) -> tuple[list[JournalEntry], list[EquitySnapshot], int, int]:
    """Load trades and equity snapshots from disk.

    Returns (closed_trades, snapshots, open_count, pending_count).
    """
    journal_dir = log_dir / "journal"
    equity_file = log_dir / "equity_curve.jsonl"
    valid_fields = {f.name for f in dc_fields(JournalEntry)}

    trades: list[JournalEntry] = []
    open_count = 0
    pending_count = 0

    if journal_dir.exists():
        for f in sorted(journal_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                entry = JournalEntry(
                    **{k: v for k, v in data.items() if k in valid_fields}
                )
                if entry.status == "closed":
                    trades.append(entry)
                elif entry.status == "open":
                    open_count += 1
                elif entry.status == "pending":
                    pending_count += 1
            except Exception:
                pass

    snapshots: list[EquitySnapshot] = []
    if equity_file.exists():
        for line in equity_file.read_text().strip().splitlines():
            if line.strip():
                try:
                    snapshots.append(EquitySnapshot(**json.loads(line)))
                except Exception:
                    pass

    return trades, snapshots, open_count, pending_count


# ── Chart generation ─────────────────────────────────────────────


def _fig_to_image(fig) -> io.BytesIO:
    """Render a matplotlib figure to an in-memory PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


def _parse_snapshots(
    snapshots: list[EquitySnapshot],
) -> list[tuple[datetime, EquitySnapshot]]:
    """Parse, sort, and deduplicate equity snapshots by timestamp."""
    parsed: list[tuple[datetime, EquitySnapshot]] = []
    for s in snapshots:
        try:
            parsed.append((datetime.fromisoformat(s.timestamp), s))
        except Exception:
            continue  # drop unparseable entries

    if len(parsed) < 2:
        return []

    parsed.sort(key=lambda x: x[0])
    seen: dict[str, int] = {}
    for i, (ts, _snap) in enumerate(parsed):
        seen[ts.isoformat()] = i
    unique_indices = sorted(seen.values())
    return [parsed[i] for i in unique_indices]


def _chart_equity_curve(
    snapshots: list[EquitySnapshot],
    spy_prices: Optional[list[tuple[datetime, float]]] = None,
) -> Optional[io.BytesIO]:
    """Equity over time with drawdown shading and optional SPY overlay.

    When *spy_prices* is provided both series are normalized to 100.
    Data is aggregated to one point per calendar day (last snapshot)
    and restricted to trading days where SPY data exists, so the
    chart stays clean on weekends / holidays.
    """
    parsed = _parse_snapshots(snapshots)
    if len(parsed) < 2:
        return None

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import date as date_type

    # ── Aggregate bot snapshots to one per calendar day (last wins) ──
    daily_bot: dict[date_type, EquitySnapshot] = {}
    for ts, snap in parsed:
        daily_bot[ts.date()] = snap  # last snapshot of day wins

    # ── Build SPY lookup by date ─────────────────────────────
    spy_by_date: dict[date_type, float] = {}
    if spy_prices and len(spy_prices) >= 2:
        for ts, price in spy_prices:
            spy_by_date[ts.date()] = price

    # ── Align series to common dates ─────────────────────────
    # When SPY data is available, restrict to trading days only
    # (dates that exist in both bot AND spy data).
    # When SPY is absent, use all bot dates.
    if spy_by_date:
        common_dates = sorted(
            d for d in daily_bot if d in spy_by_date
        )
        # If the latest bot snapshot falls on a non-trading day
        # (weekend / holiday), carry its equity back to the last
        # trading date so the chart reflects the most recent value.
        all_bot_dates = sorted(daily_bot.keys())
        if all_bot_dates and common_dates:
            latest_bot = all_bot_dates[-1]
            last_common = common_dates[-1]
            if latest_bot > last_common and last_common in spy_by_date:
                # Overwrite the last trading day's snapshot with the
                # newer (more accurate) weekend/holiday reading.
                daily_bot[last_common] = daily_bot[latest_bot]
    else:
        common_dates = sorted(daily_bot.keys())

    if len(common_dates) < 2:
        # Not enough aligned data — fall back to all bot dates
        common_dates = sorted(daily_bot.keys())

    plot_dates = [datetime.combine(d, datetime.min.time()) for d in common_dates]
    plot_equity = [daily_bot[d].equity for d in common_dates]
    plot_dd = [daily_bot[d].drawdown_pct for d in common_dates]

    # ── Normalize to growth-of-$100 ──────────────────────────
    base_eq = plot_equity[0] if plot_equity[0] != 0 else 1.0
    norm_equity = [e / base_eq * 100 for e in plot_equity]

    norm_spy: Optional[list[float]] = None
    if spy_by_date and len(common_dates) >= 2:
        first_spy = spy_by_date.get(common_dates[0])
        if first_spy and first_spy != 0:
            norm_spy = [
                spy_by_date[d] / first_spy * 100
                for d in common_dates
                if d in spy_by_date
            ]
            # Guard: lengths must match after filtering
            if len(norm_spy) != len(common_dates):
                norm_spy = None

    has_spy = norm_spy is not None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.5), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # ── Top panel: normalized performance ─────────────────────
    ax1.plot(
        plot_dates, norm_equity,
        color="#2563eb", linewidth=1.5, marker="o", markersize=3,
        label="Bot",
    )
    if has_spy:
        ax1.plot(
            plot_dates, norm_spy,
            color="#f59e0b", linewidth=1.3, marker="s", markersize=3,
            linestyle="--", label="S&P 500 (SPY)", alpha=0.85,
        )
    ax1.axhline(y=100, color="#94a3b8", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("Growth of $100")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    title = "Performance vs S&P 500" if has_spy else "Equity Curve"
    ax1.set_title(title, fontsize=12, fontweight="bold")

    # ── Bottom panel: drawdown ────────────────────────────────
    ax2.fill_between(plot_dates, plot_dd, 0, alpha=0.4, color="#ef4444")
    ax2.plot(plot_dates, plot_dd, color="#ef4444", linewidth=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_pnl_distribution(trades: list[JournalEntry]) -> Optional[io.BytesIO]:
    """Histogram of trade P&L and R-multiples."""
    if len(trades) < 3:
        return None

    import matplotlib.pyplot as plt

    sorted_trades = sorted(
        trades, key=lambda t: t.closed_at or t.exit_date or "",
    )
    pnl_vals = [t.realized_pnl or 0.0 for t in sorted_trades]
    r_vals = [
        t.r_multiple for t in sorted_trades
        if t.r_multiple is not None
        and t.initial_risk_dollars
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # P&L histogram
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in pnl_vals]
    ax1.bar(range(len(pnl_vals)), pnl_vals, color=colors, alpha=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title("P&L per Trade", fontsize=11, fontweight="bold")
    ax1.set_ylabel("P&L ($)")
    ax1.set_xlabel("Trade #")
    ax1.grid(True, alpha=0.3, axis="y")

    # R-multiple distribution
    if r_vals:
        r_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in r_vals]
        ax2.bar(range(len(r_vals)), r_vals, color=r_colors, alpha=0.8)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.axhline(y=1, color="#94a3b8", linewidth=0.8, linestyle="--",
                     label="1R")
        ax2.axhline(y=-1, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax2.set_title("R-Multiple per Trade", fontsize=11, fontweight="bold")
        ax2.set_ylabel("R-Multiple")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_win_loss(metrics: JournalMetrics) -> Optional[io.BytesIO]:
    """Win/loss pie + strategy breakdown bar chart."""
    import matplotlib.pyplot as plt

    o = metrics.overall
    if o.total_trades == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Win/Loss pie
    sizes = [o.wins, o.losses]
    labels = [f"Wins ({o.wins})", f"Losses ({o.losses})"]
    colors_pie = ["#22c55e", "#ef4444"]
    if o.breakeven > 0:
        sizes.append(o.breakeven)
        labels.append(f"Breakeven ({o.breakeven})")
        colors_pie.append("#94a3b8")

    ax1.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Win Rate", fontsize=11, fontweight="bold")

    # Strategy breakdown
    strats = list(metrics.by_strategy.values())
    if strats:
        names = [s.strategy for s in strats]
        pnls = [s.total_pnl for s in strats]
        bar_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in pnls]
        ax2.barh(names, pnls, color=bar_colors, alpha=0.8)
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.set_title("P&L by Strategy", fontsize=11, fontweight="bold")
        ax2.set_xlabel("P&L ($)")
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.text(0.5, 0.5, "No strategy data", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_cumulative_pnl(trades: list[JournalEntry]) -> Optional[io.BytesIO]:
    """Cumulative P&L curve over trade sequence (chronological)."""
    if len(trades) < 2:
        return None

    import matplotlib.pyplot as plt

    # Sort by close date for a meaningful chronological curve
    sorted_trades = sorted(
        trades,
        key=lambda t: t.closed_at or t.exit_date or "",
    )

    cum_pnl = []
    running = 0.0
    for t in sorted_trades:
        running += t.realized_pnl or 0.0
        cum_pnl.append(running)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(range(1, len(cum_pnl) + 1), cum_pnl, color="#2563eb",
            linewidth=1.5, marker="o", markersize=3)
    ax.fill_between(
        range(1, len(cum_pnl) + 1), cum_pnl, 0,
        alpha=0.1, color="#2563eb",
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Cumulative P&L", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Trade #")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_monthly_returns_heatmap(
    snapshots: list[EquitySnapshot],
) -> Optional[io.BytesIO]:
    """Calendar-style heatmap of monthly mark-to-market returns.

    Uses equity snapshots to compute month-over-month percentage
    returns (last equity of month vs last equity of prior month).
    This is the institutional standard: it includes both realized
    and unrealized P&L, giving the true account return each month.
    Monthly returns are only computed between consecutive calendar
    months to avoid misattributing multi-month gaps.

    YTD = first-to-last equity of the year.
    """
    from datetime import date as date_type

    parsed = _parse_snapshots(snapshots)
    if len(parsed) < 2:
        return None

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Aggregate to one equity per calendar day (last snapshot wins)
    daily: dict[date_type, float] = {}
    for ts, snap in parsed:
        daily[ts.date()] = snap.equity

    sorted_dates = sorted(daily.keys())
    if len(sorted_dates) < 2:
        return None

    # Last equity per month
    monthly_equity: dict[tuple[int, int], float] = {}
    for d in sorted_dates:
        monthly_equity[(d.year, d.month)] = daily[d]

    sorted_months = sorted(monthly_equity.keys())
    if len(sorted_months) < 2:
        return None

    # Compute month-over-month returns — only when months are
    # truly consecutive.  If a calendar month has no snapshot the
    # chain breaks (NaN) rather than attributing a multi-month
    # move to a single cell.
    def _months_consecutive(
        a: tuple[int, int], b: tuple[int, int],
    ) -> bool:
        if a[1] == 12:
            return b == (a[0] + 1, 1)
        return b == (a[0], a[1] + 1)

    # First equity snapshot of each year — used as the baseline
    # for the first month that has data (partial-month return).
    yearly_first_eq: dict[int, float] = {}
    if parsed:
        for ts, snap in parsed:
            if ts.year not in yearly_first_eq:
                yearly_first_eq[ts.year] = snap.equity

    # First month of each year's data gets a partial-month return
    # (first snapshot of year → end of that month) so the cells
    # compound to the YTD.
    first_month_per_year: dict[int, tuple[int, int]] = {}
    for ym in sorted_months:
        if ym[0] not in first_month_per_year:
            first_month_per_year[ym[0]] = ym

    monthly_returns: dict[tuple[int, int], float] = {}
    for yr, first_ym in first_month_per_year.items():
        start_eq = yearly_first_eq.get(yr)
        end_eq = monthly_equity.get(first_ym)
        if start_eq and end_eq and start_eq > 0:
            ret = (end_eq - start_eq) / start_eq * 100
            # Only include if nonzero or if it's the only month
            monthly_returns[first_ym] = ret

    for i in range(1, len(sorted_months)):
        prev_key = sorted_months[i - 1]
        curr_key = sorted_months[i]
        if not _months_consecutive(prev_key, curr_key):
            # Gap — the month after the gap becomes a new "first
            # month" for its stretch; use the first snapshot of
            # that month as the baseline.
            first_eq_month: float | None = None
            for ts, snap in parsed:
                if (ts.year, ts.month) == curr_key:
                    first_eq_month = snap.equity
                    break
            end_eq = monthly_equity[curr_key]
            if first_eq_month and first_eq_month > 0:
                monthly_returns[curr_key] = (
                    (end_eq - first_eq_month) / first_eq_month * 100
                )
            continue
        prev_eq = monthly_equity[prev_key]
        curr_eq = monthly_equity[curr_key]
        if prev_eq and prev_eq > 0:
            monthly_returns[curr_key] = (
                (curr_eq - prev_eq) / prev_eq * 100
            )

    if not monthly_returns:
        return None

    # Build year × month grid
    years = sorted({ym[0] for ym in monthly_returns})
    n_rows = len(years)
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    n_cols = 13  # 12 months + YTD

    grid = np.full((n_rows, 12), np.nan)
    for (yr, mo), ret in monthly_returns.items():
        if yr in years:
            row = years.index(yr)
            grid[row, mo - 1] = ret

    # Compute YTD per year from equity snapshots (first → last).
    # Consistent with monthly cells which are also equity-based.
    yearly_first_eq: dict[int, float] = {}
    yearly_last_eq: dict[int, float] = {}
    if parsed:
        for ts, snap in parsed:
            yr = ts.year
            if yr not in yearly_first_eq:
                yearly_first_eq[yr] = snap.equity
            yearly_last_eq[yr] = snap.equity

    ytd = []
    for row_i in range(n_rows):
        yr = years[row_i]
        first_eq = yearly_first_eq.get(yr)
        last_eq = yearly_last_eq.get(yr)
        if first_eq and last_eq and first_eq > 0:
            ytd.append((last_eq - first_eq) / first_eq * 100)
        else:
            ytd.append(np.nan)

    # Colour scale — symmetric around zero
    all_vals = [v for v in grid.flat if not np.isnan(v)]
    all_vals.extend(v for v in ytd if not np.isnan(v))
    abs_max = max((abs(v) for v in all_vals), default=1.0)
    abs_max = max(abs_max, 1.0)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ret", ["#ef4444", "#fef9c3", "#22c55e"],
    )

    def _val_to_color(val: float) -> str:
        norm = (val + abs_max) / (2 * abs_max)
        norm = max(0.0, min(1.0, norm))
        rgba = cmap(norm)
        return rgba

    # ── Render with matplotlib table ──────────────────────────
    # Using ax.table gives consistent cell sizes regardless of
    # figure dimensions — no data-coordinate stretching issues.
    col_labels = month_labels + ["YTD"]
    row_labels = [str(y) for y in years]

    # Build cell text and colours
    cell_text: list[list[str]] = []
    cell_colours: list[list] = []
    for row_i in range(n_rows):
        row_txt: list[str] = []
        row_clr: list = []
        for col_j in range(n_cols):
            val = grid[row_i, col_j] if col_j < 12 else ytd[row_i]
            if np.isnan(val):
                row_txt.append("")
                row_clr.append("white")
            else:
                row_txt.append(f"{val:+.1f}%")
                row_clr.append(_val_to_color(val))
            # end col
        cell_text.append(row_txt)
        cell_colours.append(row_clr)

    # Height must match the PDF insertion ratio (width * 0.35)
    # to avoid aspect-ratio distortion.  10 × 3.5 = 0.35 ratio.
    fig_h = 3.5
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")
    ax.set_title(
        "Monthly Returns (%)", fontsize=12, fontweight="bold",
        pad=8,
    )

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)  # widen row height slightly

    # Style cells
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            # Column header row
            cell.set_text_props(fontweight="bold", color="#334155")
            cell.set_facecolor("#f8fafc")
        elif col_idx == -1:
            # Row label (year)
            cell.set_text_props(fontweight="bold", color="#334155")
            cell.set_facecolor("#f8fafc")
        else:
            # Data cell — pick text colour based on background
            data_row = row_idx - 1  # row_idx 0 is header
            data_col = col_idx
            val = (
                grid[data_row, data_col]
                if data_col < 12
                else ytd[data_row]
            )
            if not np.isnan(val):
                text_color = (
                    "white" if abs(val) > abs_max * 0.55 else "black"
                )
                fw = "bold" if data_col == 12 else "normal"
                cell.set_text_props(
                    color=text_color, fontweight=fw,
                )

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_rolling_sharpe(
    snapshots: list[EquitySnapshot],
    window: int = 20,
) -> Optional[io.BytesIO]:
    """Rolling annualised Sharpe ratio over a sliding window of daily returns.

    Uses a *window*-day rolling window (default 20 trading days ≈ 1 month).
    Requires at least *window* + 5 daily observations to be meaningful.
    """
    from datetime import date as date_type

    parsed = _parse_snapshots(snapshots)
    if len(parsed) < 2:
        return None

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Aggregate to one equity per calendar day
    daily: dict[date_type, float] = {}
    for ts, snap in parsed:
        daily[ts.date()] = snap.equity

    sorted_dates = sorted(daily.keys())
    if len(sorted_dates) < window + 5:
        return None

    equities = [daily[d] for d in sorted_dates]

    # Daily-equivalent returns — scale by elapsed calendar days so
    # multi-day gaps (weekends, missed runs) don't distort the
    # annualisation.  A 3-day gap return is converted to a per-day
    # return via geometric decomposition: r_daily = (1+r)^(1/n) - 1.
    returns = []
    return_dates = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        if prev <= 0:
            returns.append(0.0)
            return_dates.append(sorted_dates[i])
            continue
        raw_ret = (equities[i] - prev) / prev
        elapsed = (sorted_dates[i] - sorted_dates[i - 1]).days
        if elapsed > 1 and abs(raw_ret) > 1e-12:
            # Geometric per-day return
            sign = 1 if (1 + raw_ret) > 0 else -1
            daily_ret = sign * (abs(1 + raw_ret) ** (1.0 / elapsed) - 1)
        else:
            daily_ret = raw_ret
        returns.append(daily_ret)
        return_dates.append(sorted_dates[i])

    if len(returns) < window:
        return None

    # Rolling Sharpe (annualised)
    roll_sharpe = []
    roll_dates = []
    sqrt_252 = np.sqrt(252)
    for i in range(window - 1, len(returns)):
        w = returns[i - window + 1: i + 1]
        mean_r = sum(w) / window
        var_r = sum((r - mean_r) ** 2 for r in w) / (window - 1)
        std_r = var_r ** 0.5
        if std_r > 1e-10:
            sr = (mean_r / std_r) * sqrt_252
        else:
            sr = 0.0
        roll_sharpe.append(sr)
        roll_dates.append(
            datetime.combine(return_dates[i], datetime.min.time())
        )

    if len(roll_sharpe) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(roll_dates, roll_sharpe, color="#2563eb", linewidth=1.3)
    ax.fill_between(
        roll_dates, roll_sharpe, 0,
        where=[s >= 0 for s in roll_sharpe],
        alpha=0.15, color="#22c55e", interpolate=True,
    )
    ax.fill_between(
        roll_dates, roll_sharpe, 0,
        where=[s < 0 for s in roll_sharpe],
        alpha=0.15, color="#ef4444", interpolate=True,
    )

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1.0, color="#22c55e", linewidth=0.7, linestyle="--",
               alpha=0.6, label="Sharpe = 1.0")
    ax.axhline(y=-1.0, color="#ef4444", linewidth=0.7, linestyle="--",
               alpha=0.6, label="Sharpe = -1.0")

    ax.set_ylabel("Annualised Sharpe")
    ax.set_title(
        f"Rolling Sharpe Ratio ({window}-day window)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _trade_color(t: JournalEntry) -> str:
    """Return dot color: green for wins, red for losses, gray for breakeven."""
    pnl = t.realized_pnl or 0.0
    if pnl > 0:
        return "#22c55e"
    elif pnl < 0:
        return "#ef4444"
    return "#94a3b8"  # breakeven


def _chart_mfe_mae_scatter(trades: list[JournalEntry]) -> Optional[io.BytesIO]:
    """Scatter plot of MAE vs MFE per trade, colored by win/loss/breakeven."""
    closed = [
        t for t in trades
        if t.status == "closed"
        and t.mfe_pct is not None
        and t.mae_pct is not None
    ]
    if len(closed) < 3:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mfe_vals = [t.mfe_pct for t in closed]
    mae_vals = [t.mae_pct for t in closed]
    colors = [_trade_color(t) for t in closed]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(mae_vals, mfe_vals, c=colors, alpha=0.7, edgecolors="white",
               linewidths=0.5, s=50)

    # Diagonal reference: MFE = MAE means breakeven excursion
    max_val = max(max(mfe_vals, default=1), max(mae_vals, default=1)) * 1.1
    ax.plot([0, max_val], [0, max_val], color="#94a3b8", linewidth=0.8,
            linestyle="--", label="MFE = MAE")

    ax.set_xlabel("MAE (%) — Max Adverse Excursion", fontsize=10)
    ax.set_ylabel("MFE (%) — Max Favorable Excursion", fontsize=10)
    ax.set_title("Trade Excursion: MFE vs MAE", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#22c55e",
               markersize=8, label="Win"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444",
               markersize=8, label="Loss"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8",
               markersize=8, label="Breakeven"),
        Line2D([0], [0], color="#94a3b8", linestyle="--", label="MFE = MAE"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_exit_reason_breakdown(
    metrics: JournalMetrics,
) -> Optional[io.BytesIO]:
    """Horizontal bar chart of exit reason distribution."""
    dist = metrics.holding.exit_reason_distribution
    if not dist:
        return None

    import matplotlib.pyplot as plt

    # Sort by count descending
    sorted_reasons = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in sorted_reasons]
    counts = [r[1] for r in sorted_reasons]
    total = sum(counts) or 1

    # Height must match the PDF insertion ratio (width * 0.35)
    # to avoid aspect-ratio distortion.  10 × 3.5 = 0.35 ratio.
    fig, ax = plt.subplots(figsize=(10, 3.5))

    bar_colors = []
    color_map = {
        "take_profit": "#22c55e",
        "stop_loss": "#ef4444",
        "trailing_stop": "#f59e0b",
        "time_exit": "#94a3b8",
        "signal_reversal": "#2563eb",
    }
    for label in labels:
        matched = False
        for key, color in color_map.items():
            if key in label.lower().replace(" ", "_"):
                bar_colors.append(color)
                matched = True
                break
        if not matched:
            bar_colors.append("#6366f1")

    bars = ax.barh(labels, counts, color=bar_colors, alpha=0.85)

    # Add percentage labels
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.0f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of Trades")
    ax.set_title("Exit Reason Breakdown", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


def _chart_holding_vs_return(trades: list[JournalEntry]) -> Optional[io.BytesIO]:
    """Scatter plot of holding period (days) vs realized return (%).

    Includes same-day exits (holding_days == 0) so the chart reflects
    full closed-trade behavior including fast stop-outs.
    """
    closed = [
        t for t in trades
        if t.status == "closed"
        and t.holding_days is not None
        and t.holding_days >= 0
        and t.realized_pnl_pct is not None
    ]
    if len(closed) < 3:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    days = [t.holding_days for t in closed]
    returns = [t.realized_pnl_pct for t in closed]
    colors = [_trade_color(t) for t in closed]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(days, returns, c=colors, alpha=0.7, edgecolors="white",
               linewidths=0.5, s=50)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Trend line — only when there's meaningful variance in hold duration.
    # polyfit is ill-conditioned when all x-values are identical.
    has_trend = False
    z = None
    if len(closed) >= 5 and len(set(days)) >= 2:
        z = np.polyfit(days, returns, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(days), max(days), 100)
        ax.plot(x_line, p(x_line), color="#2563eb", linewidth=1.2,
                linestyle="--", alpha=0.7)
        has_trend = True

    ax.set_xlabel("Holding Period (days)", fontsize=10)
    ax.set_ylabel("Return (%)", fontsize=10)
    ax.set_title("Holding Period vs Return", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#22c55e",
               markersize=8, label="Win"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444",
               markersize=8, label="Loss"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8",
               markersize=8, label="Breakeven"),
    ]
    if has_trend and z is not None:
        legend_elements.append(
            Line2D([0], [0], color="#2563eb", linestyle="--",
                   label=f"Trend (slope={z[0]:.3f})"),
        )
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    fig.tight_layout()
    buf = _fig_to_image(fig)
    plt.close(fig)
    return buf


# ── PDF assembly ─────────────────────────────────────────────────


def generate_pdf_report(
    log_dir: Path,
    output_path: Path,
) -> Path:
    """Build a full PDF performance report with charts.

    Args:
        log_dir: Directory containing journal/ and equity_curve.jsonl.
        output_path: Where to write the PDF.

    Returns:
        The output path.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Table,
        TableStyle,
        PageBreak,
        KeepTogether,
        CondPageBreak,
    )

    trades, snapshots, open_count, pending_count = _load_journal_data(log_dir)
    metrics = compute_journal_metrics(trades, snapshots) if trades else None

    # ── Fetch SPY benchmark data ─────────────────────────────
    parsed_snaps = _parse_snapshots(snapshots)
    spy_prices: Optional[list[tuple[datetime, float]]] = None
    bench_stats: Optional[BenchmarkStats] = None
    if len(parsed_snaps) >= 2:
        start_dt = parsed_snaps[0][0]
        end_dt = parsed_snaps[-1][0]
        spy_prices = _fetch_spy_prices(start_dt, end_dt)
        if spy_prices:
            bot_dates = [ts for ts, _ in parsed_snaps]
            bot_equity = [s.equity for _, s in parsed_snaps]
            # Pass the already-computed bot Sharpe/Sortino from
            # journal_analytics so the Benchmark Comparison table
            # shows the same values as the Key Metrics table.
            _ra = metrics.risk_adjusted if metrics else None
            bench_stats = _compute_benchmark_stats(
                bot_dates, bot_equity, spy_prices,
                bot_sharpe_override=_ra.sharpe_ratio if _ra else None,
                bot_sortino_override=_ra.sortino_ratio if _ra else None,
            )

    # ── Styles ──────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=22, spaceAfter=6,
        textColor=HexColor("#1e293b"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=14, spaceBefore=14, spaceAfter=6,
        textColor=HexColor("#1e40af"),
    ))
    styles.add(ParagraphStyle(
        "MetricLabel", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#64748b"),
    ))
    styles.add(ParagraphStyle(
        "MetricValue", parent=styles["Normal"],
        fontSize=11, textColor=HexColor("#1e293b"),
    ))
    styles.add(ParagraphStyle(
        "SubText", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#94a3b8"),
    ))
    styles.add(ParagraphStyle(
        "CellLabel", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#64748b"),
        leading=11,
    ))
    styles.add(ParagraphStyle(
        "CellValue", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#1e293b"),
        fontName="Helvetica-Bold",
        leading=12,
    ))
    styles.add(ParagraphStyle(
        "CellSmall", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#1e293b"),
        leading=10,
    ))

    # Helpers: wrap text in Paragraph so long strings wrap inside cells
    def _lbl(text: str) -> Paragraph:
        """Label cell (grey, wraps)."""
        return Paragraph(text, styles["CellLabel"])

    def _val(text: str) -> Paragraph:
        """Value cell (bold, wraps)."""
        return Paragraph(text, styles["CellValue"])

    def _cell(text: str) -> Paragraph:
        """Generic small cell (wraps)."""
        return Paragraph(text, styles["CellSmall"])

    story = []
    width = letter[0] - 2 * inch  # usable width

    # ── Title page header ───────────────────────────────────────
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph("Trading Bot Performance Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Generated {now_str}", styles["SubText"]))
    story.append(Spacer(1, 12))

    # ── Summary box ─────────────────────────────────────────────
    story.append(Paragraph("Overview", styles["SectionHead"]))

    summary_data = [
        [_lbl("Closed Trades"), _val(str(len(trades))),
         _lbl("Open"), _val(str(open_count)),
         _lbl("Pending"), _val(str(pending_count))],
    ]
    if snapshots:
        first, last = snapshots[0], snapshots[-1]
        equity_change = last.equity - first.equity
        eq_color = "#22c55e" if equity_change > 0 else (
            "#ef4444" if equity_change < 0 else "#1e293b"
        )
        summary_data.append([
            _lbl("Starting Equity"), _val(f"${first.equity:,.2f}"),
            _lbl("Current Equity"), _val(f"${last.equity:,.2f}"),
            _lbl("HWM"), _val(f"${last.high_water_mark:,.2f}"),
        ])
        summary_data.append([
            _lbl("Net P&amp;L"),
            Paragraph(
                f'<font color="{eq_color}">'
                f"${equity_change:+,.2f}</font>",
                styles["CellValue"],
            ),
            _lbl("Drawdown"), _val(f"{last.drawdown_pct:.2f}%"),
            _lbl("Period"),
            _val(f"{first.timestamp[:10]} to {last.timestamp[:10]}"),
        ])

    if summary_data:
        # Label cols narrower, value cols wider; period value gets extra room
        t = Table(summary_data, colWidths=[
            width * 0.12, width * 0.21,
            width * 0.12, width * 0.19,
            width * 0.12, width * 0.18,
        ])
        t.setStyle(TableStyle([
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f8fafc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    if not trades:
        story.append(Paragraph(
            "No closed trades yet. Run the bot and let some trades "
            "complete before generating a report.",
            styles["Normal"],
        ))
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        doc.build(story)
        return output_path

    o = metrics.overall

    # ── Key metrics table ───────────────────────────────────────
    story.append(Paragraph("Key Metrics", styles["SectionHead"]))

    def _fmt_pnl(v):
        color = GREEN if v > 0 else (RED if v < 0 else NEUTRAL)
        return f'<font color="{color}">${v:+,.2f}</font>'

    def _fmt_pf(v: float) -> str:
        """Format profit factor, handling infinity."""
        if v == float("inf") or v > 99:
            return "∞"  # ∞ symbol
        return f"{v:.2f}"

    def _fmt_pct(v):
        color = GREEN if v > 0 else (RED if v < 0 else NEUTRAL)
        return f'<font color="{color}">{v:+.1f}%</font>'

    # ── Conditional-color value helpers ──────────────────────────
    # _cv  = color a formatted string green/red by sign of `v`
    # _cvt = threshold variant: green if above threshold, red below
    GREEN, RED, NEUTRAL = "#22c55e", "#ef4444", "#1e293b"

    def _cv(text: str, v: float, style_name: str = "CellValue") -> Paragraph:
        """Wrap *text* in green (v>0), red (v<0), or neutral (v==0)."""
        color = GREEN if v > 0 else (RED if v < 0 else NEUTRAL)
        return Paragraph(f'<font color="{color}">{text}</font>',
                         styles[style_name])

    def _cvt(text: str, v: float, good_above: float,
             style_name: str = "CellValue") -> Paragraph:
        """Green if v >= good_above, red if below."""
        color = GREEN if v >= good_above else RED
        return Paragraph(f'<font color="{color}">{text}</font>',
                         styles[style_name])

    metrics_data = [
        [_lbl("Win Rate"), _val(f"{o.win_rate:.1%}"),
         _lbl("Profit Factor"),
         _cvt(_fmt_pf(o.profit_factor), o.profit_factor, 1.0)],
        [_lbl("Closed P&amp;L"),
         Paragraph(_fmt_pnl(o.total_pnl), styles["CellValue"]),
         _lbl("Expectancy"),
         _cv(f"${o.expectancy:+.2f}/trade", o.expectancy)],
        [_lbl("Avg Win"), _val(f"${o.avg_win:+.2f}"),
         _lbl("Avg Loss"), _val(f"${o.avg_loss:+.2f}")],
        [_lbl("Largest Win"), _val(f"${o.largest_win:+.2f}"),
         _lbl("Largest Loss"), _val(f"${o.largest_loss:+.2f}")],
        [_lbl("Win/Loss Ratio"), _val(f"{o.win_loss_ratio:.2f}"),
         _lbl("Expectancy (R)"),
         _cv(f"{o.expectancy_r:+.2f}R", o.expectancy_r)],
    ]

    ra = metrics.risk_adjusted
    if ra.sharpe_ratio != 0:
        metrics_data.append([
            _lbl("Sharpe"), _val(f"{ra.sharpe_ratio:.2f}"),
            _lbl("Calmar"), _val(f"{ra.calmar_ratio:.2f}"),
        ])
        metrics_data.append([
            _lbl("PSR"), _val(f"{ra.probabilistic_sharpe:.1%}"),
            _lbl("Sortino"), _val(f"{ra.sortino_ratio:.2f}"),
        ])

    t = Table(metrics_data, colWidths=[width * 0.20, width * 0.30,
                                       width * 0.20, width * 0.30])
    t.setStyle(TableStyle([
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # ── Charts ──────────────────────────────────────────────────
    # KeepTogether prevents orphaned section headers: if the
    # header + chart don't fit on the current page, both move
    # together to the next page.

    # Equity curve (with SPY overlay when available)
    eq_buf = _chart_equity_curve(snapshots, spy_prices=spy_prices)
    if eq_buf:
        eq_title = (
            "Performance vs S&amp;P 500"
            if spy_prices
            else "Equity Curve &amp; Drawdown"
        )
        story.append(KeepTogether([
            Paragraph(eq_title, styles["SectionHead"]),
            Image(eq_buf, width=width, height=width * 0.55),
            Spacer(1, 12),
        ]))

    # Benchmark comparison table
    if bench_stats:
        story.append(
            Paragraph("Benchmark Comparison (SPY)", styles["SectionHead"])
        )

        def _color_val(val: float, fmt: str = "{:+.2f}") -> Paragraph:
            """Color a value green/red/neutral based on sign."""
            color = GREEN if val > 0 else (RED if val < 0 else NEUTRAL)
            text = fmt.format(val)
            return Paragraph(
                f'<font color="{color}">{text}</font>',
                styles["CellValue"],
            )

        bench_header = [_lbl(""), _lbl("Bot"), _lbl("S&amp;P 500")]
        bs = bench_stats
        bench_data = [
            bench_header,
            [
                _lbl("Cumulative Return"),
                _color_val(bs.bot_cumulative_return * 100, "{:+.2f}%"),
                _color_val(bs.bench_cumulative_return * 100, "{:+.2f}%"),
            ],
            [
                _lbl("Annualized Return"),
                _color_val(bs.bot_annualized_return * 100, "{:+.2f}%"),
                _color_val(bs.bench_annualized_return * 100, "{:+.2f}%"),
            ],
            [
                _lbl("Sharpe Ratio"),
                _val(f"{bs.bot_sharpe:.2f}"),
                _val(f"{bs.bench_sharpe:.2f}"),
            ],
            [
                _lbl("Sortino Ratio"),
                _val(f"{bs.bot_sortino:.2f}"),
                _val(f"{bs.bench_sortino:.2f}"),
            ],
            [
                _lbl("Max Drawdown"),
                _color_val(bs.bot_max_drawdown * 100, "{:.2f}%"),
                _color_val(bs.bench_max_drawdown * 100, "{:.2f}%"),
            ],
            [
                _lbl("Volatility (ann.)"),
                _val(f"{bs.bot_volatility * 100:.2f}%"),
                _val(f"{bs.bench_volatility * 100:.2f}%"),
            ],
        ]
        t = Table(
            bench_data,
            colWidths=[width * 0.34, width * 0.33, width * 0.33],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [HexColor("#ffffff"), HexColor("#f8fafc")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

        # Alpha / Beta / IR / Correlation row
        relative_data = [
            [
                _lbl("Alpha (ann.)"),
                _color_val(bs.alpha * 100, "{:+.2f}%"),
                _lbl("Beta"),
                _val(f"{bs.beta:.2f}"),
            ],
            [
                _lbl("Information Ratio"),
                _val(f"{bs.information_ratio:+.2f}"),
                _lbl("Correlation"),
                _val(f"{bs.correlation:.2f}"),
            ],
        ]
        t2 = Table(
            relative_data,
            colWidths=[
                width * 0.20, width * 0.30,
                width * 0.20, width * 0.30,
            ],
        )
        t2.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t2)
        story.append(Spacer(1, 16))

    # Monthly returns heatmap
    monthly_buf = _chart_monthly_returns_heatmap(snapshots)
    if monthly_buf:
        story.append(KeepTogether([
            Paragraph("Monthly Returns", styles["SectionHead"]),
            Image(monthly_buf, width=width, height=width * 0.35),
            Spacer(1, 12),
        ]))

    # Rolling Sharpe
    rolling_sharpe_buf = _chart_rolling_sharpe(snapshots)
    if rolling_sharpe_buf:
        story.append(KeepTogether([
            Paragraph("Rolling Sharpe Ratio", styles["SectionHead"]),
            Image(rolling_sharpe_buf, width=width, height=width * 0.35),
            Spacer(1, 12),
        ]))

    # Cumulative P&L
    cum_buf = _chart_cumulative_pnl(trades)
    if cum_buf:
        story.append(KeepTogether([
            Paragraph("Cumulative P&amp;L", styles["SectionHead"]),
            Image(cum_buf, width=width, height=width * 0.3),
            Spacer(1, 12),
        ]))

    # P&L distribution
    pnl_buf = _chart_pnl_distribution(trades)
    if pnl_buf:
        story.append(KeepTogether([
            Paragraph("P&amp;L &amp; R-Multiple Distribution", styles["SectionHead"]),
            Image(pnl_buf, width=width, height=width * 0.35),
            Spacer(1, 12),
        ]))

    # Win/loss + strategy
    wl_buf = _chart_win_loss(metrics)
    if wl_buf:
        story.append(KeepTogether([
            Paragraph("Win Rate &amp; Strategy Attribution", styles["SectionHead"]),
            Image(wl_buf, width=width, height=width * 0.35),
            Spacer(1, 12),
        ]))

    # MFE vs MAE scatter + excursion summary table
    mfe_mae_buf = _chart_mfe_mae_scatter(trades)
    if mfe_mae_buf:
        exc = metrics.excursion
        exc_data = [
            [_lbl("Avg MFE (win)"), _val(f"{exc.avg_mfe_winners:.2f}%"),
             _lbl("Avg MFE (loss)"), _val(f"{exc.avg_mfe_losers:.2f}%")],
            [_lbl("Avg MAE (win)"), _val(f"{exc.avg_mae_winners:.2f}%"),
             _lbl("Avg MAE (loss)"), _val(f"{exc.avg_mae_losers:.2f}%")],
            [_lbl("Edge Ratio"),
             _cvt(f"{exc.avg_edge_ratio:.2f}", exc.avg_edge_ratio, 1.0),
             _lbl("Edge Ratio (win)"),
             _cvt(f"{exc.avg_edge_ratio_winners:.2f}",
                  exc.avg_edge_ratio_winners, 1.0)],
            [_lbl("Avg ETD (win)"), _val(f"{exc.avg_etd_winners:.2f}%"),
             _lbl("Avg ETD (loss)"), _val(f"{exc.avg_etd_losers:.2f}%")],
        ]
        t_exc = Table(exc_data, colWidths=[
            width * 0.20, width * 0.30,
            width * 0.20, width * 0.30,
        ])
        t_exc.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(KeepTogether([
            Paragraph("Trade Excursion: MFE vs MAE", styles["SectionHead"]),
            Image(mfe_mae_buf, width=width, height=width * 0.4),
            t_exc,
            Spacer(1, 12),
        ]))

    # Exit reason breakdown
    exit_buf = _chart_exit_reason_breakdown(metrics)
    if exit_buf:
        story.append(KeepTogether([
            Paragraph("Exit Reason Breakdown", styles["SectionHead"]),
            Image(exit_buf, width=width, height=width * 0.35),
            Spacer(1, 12),
        ]))

    # Holding period vs return
    hold_buf = _chart_holding_vs_return(trades)
    if hold_buf:
        story.append(KeepTogether([
            Paragraph("Holding Period vs Return", styles["SectionHead"]),
            Image(hold_buf, width=width, height=width * 0.4),
            Spacer(1, 12),
        ]))

    # ── R-Distribution table ────────────────────────────────────
    rd = metrics.r_distribution
    r_data = [
        [_lbl("Mean R"),
         _cv(f"{rd.mean_r:+.2f}", rd.mean_r),
         _lbl("Median R"),
         _cv(f"{rd.median_r:+.2f}", rd.median_r)],
        [_lbl("Std R"), _val(f"{rd.std_r:.2f}"),
         _lbl("Skewness"),
         _cv(f"{rd.skewness_r:+.3f}", rd.skewness_r)],
        [_lbl("&gt;2R"), _val(f"{rd.pct_above_2r:.0%}"),
         _lbl("&lt;-1R"), _val(f"{rd.pct_below_neg1r:.0%}")],
    ]
    t = Table(r_data, colWidths=[width * 0.20, width * 0.30,
                                  width * 0.20, width * 0.30])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(KeepTogether([
        Paragraph("R-Distribution", styles["SectionHead"]),
        t, Spacer(1, 12),
    ]))

    # ── Streak analysis ─────────────────────────────────────────
    st = metrics.streaks
    streak_type = "win" if st.current_streak_type == "win" else "loss"
    _streak_v = st.current_streak if streak_type == "win" else -st.current_streak
    streak_data = [
        [_lbl("Max Win Streak"),
         _cv(str(st.max_consecutive_wins), 1),
         _lbl("Max Loss Streak"),
         _cv(str(st.max_consecutive_losses), -1)],
        [_lbl("Current Streak"),
         _cv(f"{st.current_streak} {streak_type}(s)", _streak_v),
         _lbl("Exp. Max Loss Streak"),
         _val(str(st.expected_max_losing_streak))],
    ]
    t = Table(streak_data, colWidths=[width * 0.20, width * 0.30,
                                      width * 0.20, width * 0.30])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(KeepTogether([
        Paragraph("Streak Analysis", styles["SectionHead"]),
        t, Spacer(1, 12),
    ]))

    # ── Helper for breakdown table headers (white text on blue) ──
    hdr_style = ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.white, leading=11,
    )

    def _hdr(text: str) -> Paragraph:
        return Paragraph(text, hdr_style)

    breakdown_widths = [
        width * 0.25, width * 0.10, width * 0.13,
        width * 0.10, width * 0.13, width * 0.22,
    ]
    breakdown_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    # ── Trade type breakdown (swing vs day) ─────────────────────
    # Surface day-trade vs swing performance separately so the
    # 25% sub-portfolio can be evaluated independently from the
    # main long-horizon book.
    type_breakdown = breakdown_by_trade_type(trades)
    # Only render when there is more than one type present —
    # for swing-only history (the common case until the day-trader
    # ships) this section would be redundant with Key Metrics.
    if len(type_breakdown) > 1:
        header = [
            _hdr("Trade Type"), _hdr("Trades"), _hdr("Win Rate"),
            _hdr("PF"), _hdr("Avg R"), _hdr("Total P&amp;L"),
        ]
        rows = [header]
        # Stable ordering: swing first, then daytrade, then any
        # future types alphabetically.
        ordered = sorted(
            type_breakdown.items(),
            key=lambda kv: (
                0 if kv[0] == "swing"
                else 1 if kv[0] == "daytrade"
                else 2,
                kv[0],
            ),
        )
        for trade_type, m in ordered:
            o = m.overall
            r = m.r_distribution
            rows.append([
                _cell(trade_type),
                _cell(str(o.total_trades)),
                _cvt(f"{o.win_rate:.0%}", o.win_rate, 0.50,
                     "CellSmall"),
                _cvt(_fmt_pf(o.profit_factor), o.profit_factor, 1.0,
                     "CellSmall"),
                _cv(f"{r.mean_r:+.2f}", r.mean_r, "CellSmall"),
                Paragraph(
                    _fmt_pnl(o.total_pnl), styles["CellSmall"],
                ),
            ])
        t = Table(rows, colWidths=breakdown_widths)
        t.setStyle(breakdown_style)
        story.append(KeepTogether([
            Paragraph(
                "Performance by Trade Type",
                styles["SectionHead"],
            ),
            t,
            Paragraph(
                "Trade-level metrics only. Sharpe/Sortino/Drawdown "
                "remain account-wide because the equity curve is "
                "not split by trade type.",
                styles["SubText"],
            ),
            Spacer(1, 12),
        ]))

    # ── Strategy breakdown table ────────────────────────────────
    if metrics.by_strategy:
        header = [_hdr("Strategy"), _hdr("Trades"), _hdr("Win Rate"),
                  _hdr("PF"), _hdr("Avg R"), _hdr("Total P&amp;L")]
        rows = [header]
        for s in metrics.by_strategy.values():
            rows.append([
                _cell(s.strategy),
                _cell(str(s.trade_count)),
                _cvt(f"{s.win_rate:.0%}", s.win_rate, 0.50,
                     "CellSmall"),
                _cvt(_fmt_pf(s.profit_factor), s.profit_factor, 1.0,
                     "CellSmall"),
                _cv(f"{s.avg_r:+.2f}", s.avg_r, "CellSmall"),
                Paragraph(
                    _fmt_pnl(s.total_pnl), styles["CellSmall"],
                ),
            ])
        t = Table(rows, colWidths=breakdown_widths)
        t.setStyle(breakdown_style)
        story.append(KeepTogether([
            Paragraph("Strategy Breakdown", styles["SectionHead"]),
            t, Spacer(1, 12),
        ]))

    # ── Regime breakdown table ──────────────────────────────────
    if metrics.by_regime:
        header = [_hdr("Regime"), _hdr("Trades"), _hdr("Win Rate"),
                  _hdr("PF"), _hdr("Avg R"), _hdr("Total P&amp;L")]
        rows = [header]
        for r in metrics.by_regime.values():
            rows.append([
                _cell(r.regime),
                _cell(str(r.trade_count)),
                _cvt(f"{r.win_rate:.0%}", r.win_rate, 0.50,
                     "CellSmall"),
                _cvt(_fmt_pf(r.profit_factor), r.profit_factor, 1.0,
                     "CellSmall"),
                _cv(f"{r.avg_r:+.2f}", r.avg_r, "CellSmall"),
                Paragraph(
                    _fmt_pnl(r.total_pnl), styles["CellSmall"],
                ),
            ])
        t = Table(rows, colWidths=breakdown_widths)
        t.setStyle(breakdown_style)
        story.append(KeepTogether([
            Paragraph("Regime Breakdown", styles["SectionHead"]),
            t, Spacer(1, 12),
        ]))

    # ── Trade log table ─────────────────────────────────────────
    # Only break to a new page when less than 3 inches remain,
    # so the trade log header + a few rows stay together without
    # leaving a large blank gap after a short regime table.
    story.append(CondPageBreak(3 * inch))
    story.append(Paragraph("Trade Log", styles["SectionHead"]))

    log_hdr = [
        _hdr("Ticker"), _hdr("Strategy"), _hdr("Type"),
        _hdr("Entry"), _hdr("Exit"),
        _hdr("P&amp;L"), _hdr("R"), _hdr("Days"), _hdr("Exit Reason"),
    ]
    rows = [log_hdr]
    for t_entry in sorted(
        trades, key=lambda t: t.closed_at or t.exit_date or "",
    ):
        _pnl = t_entry.realized_pnl or 0.0
        pnl_color = GREEN if _pnl > 0 else (RED if _pnl < 0 else NEUTRAL)
        # Compact label: "day" stands out from the default "swing".
        type_label = (
            "day" if t_entry.trade_type == "daytrade"
            else (t_entry.trade_type or "swing")
        )
        rows.append([
            _cell(t_entry.ticker),
            _cell(t_entry.strategy[:16]),
            _cell(type_label),
            _cell(f"${t_entry.entry_fill_price:.2f}"
                  if t_entry.entry_fill_price else "-"),
            _cell(f"${t_entry.exit_price:.2f}"
                  if t_entry.exit_price else "-"),
            Paragraph(
                f'<font color="{pnl_color}">${_pnl:+.2f}</font>',
                styles["CellSmall"],
            ),
            (_cv(f"{t_entry.r_multiple:+.2f}", t_entry.r_multiple,
                 "CellSmall")
             if t_entry.r_multiple is not None
             and t_entry.initial_risk_dollars
             else _cell("-")),
            _cell(str(t_entry.holding_days)
                  if t_entry.holding_days else "-"),
            _cell(t_entry.exit_reason or "-"),
        ])

    col_widths = [width * 0.09, width * 0.13, width * 0.07,
                  width * 0.10, width * 0.10,
                  width * 0.11, width * 0.06, width * 0.08, width * 0.24]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    # ── Footer ──────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Report generated on {now_str}. "
        f"Data from {log_dir}/.",
        styles["SubText"],
    ))

    # ── Build PDF ───────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story)
    log.info(f"  PDF report written to {output_path}")
    return output_path


def generate_csv_export(
    log_dir: Path,
    output_path: Path,
) -> Path:
    """Export all closed trades as a CSV file.

    Args:
        log_dir: Directory containing journal/.
        output_path: Where to write the CSV.

    Returns:
        The output path.
    """
    import csv

    trades, _, _, _ = _load_journal_data(log_dir)

    columns = [
        "trade_id", "ticker", "strategy", "side",
        "entry_signal_price", "entry_fill_price",
        "exit_price", "exit_reason",
        "realized_pnl", "r_multiple", "holding_days",
        "mfe", "mae", "edge_ratio", "etd",
        "entry_vix", "entry_market_regime",
        "sl_price", "tp_price",
        "entry_slippage", "exit_slippage",
        "status",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            row = {}
            for col in columns:
                row[col] = getattr(t, col, "")
            writer.writerow(row)

    log.info(f"  CSV export written to {output_path}")
    return output_path
