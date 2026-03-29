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
) -> Optional[BenchmarkStats]:
    """Compute side-by-side bot vs SPY statistics."""
    if len(bot_dates) < 3 or len(spy_prices) < 3:
        return None

    # Build daily bot returns from equity series
    bot_returns: list[float] = []
    for i in range(1, len(bot_equity)):
        if bot_equity[i - 1] != 0:
            bot_returns.append(bot_equity[i] / bot_equity[i - 1] - 1)
        else:
            bot_returns.append(0.0)

    # Build daily SPY returns from price series
    spy_returns: list[float] = []
    for i in range(1, len(spy_prices)):
        prev_price = spy_prices[i - 1][1]
        if prev_price != 0:
            spy_returns.append(spy_prices[i][1] / prev_price - 1)
        else:
            spy_returns.append(0.0)

    # Align to same length (use shorter)
    min_len = min(len(bot_returns), len(spy_returns))
    if min_len < 2:
        return None
    bot_r = np.array(bot_returns[-min_len:])
    spy_r = np.array(spy_returns[-min_len:])

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
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(trading_days))

    def _sortino(returns: np.ndarray) -> float:
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(
            returns.mean() / downside.std() * np.sqrt(trading_days)
        )

    def _max_dd(returns: np.ndarray) -> float:
        cum = np.cumprod(1 + returns)
        hwm = np.maximum.accumulate(cum)
        dd = (cum - hwm) / hwm
        return float(dd.min()) if len(dd) > 0 else 0.0

    def _volatility(returns: np.ndarray) -> float:
        return float(returns.std() * np.sqrt(trading_days))

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
        bot_sharpe=_sharpe(bot_r),
        bot_sortino=_sortino(bot_r),
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

    pnl_vals = [t.realized_pnl for t in trades]
    r_vals = [t.r_multiple for t in trades if t.r_multiple != 0]

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
    """Cumulative P&L curve over trade sequence."""
    if len(trades) < 2:
        return None

    import matplotlib.pyplot as plt

    cum_pnl = []
    running = 0.0
    for t in trades:
        running += t.realized_pnl
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
            bench_stats = _compute_benchmark_stats(
                bot_dates, bot_equity, spy_prices,
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
        summary_data.append([
            _lbl("Starting Equity"), _val(f"${first.equity:,.2f}"),
            _lbl("Current Equity"), _val(f"${last.equity:,.2f}"),
            _lbl("HWM"), _val(f"${last.high_water_mark:,.2f}"),
        ])
        summary_data.append([
            _lbl("Period"),
            _val(f"{first.timestamp[:10]} to {last.timestamp[:10]}"),
            _lbl("Snapshots"), _val(str(len(snapshots))),
            _lbl("Drawdown"), _val(f"{last.drawdown_pct:.2f}%"),
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
        color = "#22c55e" if v >= 0 else "#ef4444"
        return f'<font color="{color}">${v:+,.2f}</font>'

    def _fmt_pct(v):
        color = "#22c55e" if v >= 0 else "#ef4444"
        return f'<font color="{color}">{v:+.1f}%</font>'

    metrics_data = [
        [_lbl("Win Rate"), _val(f"{o.win_rate:.1%}"),
         _lbl("Profit Factor"), _val(f"{o.profit_factor:.2f}")],
        [_lbl("Total P&amp;L"),
         Paragraph(_fmt_pnl(o.total_pnl), styles["CellValue"]),
         _lbl("Expectancy"), _val(f"${o.expectancy:+.2f}/trade")],
        [_lbl("Avg Win"), _val(f"${o.avg_win:+.2f}"),
         _lbl("Avg Loss"), _val(f"${o.avg_loss:+.2f}")],
        [_lbl("Largest Win"), _val(f"${o.largest_win:+.2f}"),
         _lbl("Largest Loss"), _val(f"${o.largest_loss:+.2f}")],
        [_lbl("Win/Loss Ratio"), _val(f"{o.win_loss_ratio:.2f}"),
         _lbl("Expectancy (R)"), _val(f"{o.expectancy_r:+.2f}R")],
    ]

    ra = metrics.risk_adjusted
    if ra.sharpe_ratio != 0:
        metrics_data.append([
            _lbl("Sharpe"), _val(f"{ra.sharpe_ratio:.2f}"),
            _lbl("Sortino"), _val(f"{ra.sortino_ratio:.2f}"),
        ])
        metrics_data.append([
            _lbl("Calmar"), _val(f"{ra.calmar_ratio:.2f}"),
            _lbl("PSR"), _val(f"{ra.probabilistic_sharpe:.1%}"),
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
            """Color a value green/red based on sign."""
            color = "#22c55e" if val >= 0 else "#ef4444"
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

    # ── R-Distribution table ────────────────────────────────────
    rd = metrics.r_distribution
    r_data = [
        [_lbl("Mean R"), _val(f"{rd.mean_r:+.2f}"),
         _lbl("Median R"), _val(f"{rd.median_r:+.2f}")],
        [_lbl("Std R"), _val(f"{rd.std_r:.2f}"),
         _lbl("Skewness"), _val(f"{rd.skewness_r:+.3f}")],
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
    streak_data = [
        [_lbl("Max Win Streak"), _val(str(st.max_consecutive_wins)),
         _lbl("Max Loss Streak"), _val(str(st.max_consecutive_losses))],
        [_lbl("Current Streak"),
         _val(f"{st.current_streak} {streak_type}(s)"),
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

    # ── Strategy breakdown table ────────────────────────────────
    if metrics.by_strategy:
        header = [_hdr("Strategy"), _hdr("Trades"), _hdr("Win Rate"),
                  _hdr("PF"), _hdr("Avg R"), _hdr("Total P&amp;L")]
        rows = [header]
        for s in metrics.by_strategy.values():
            rows.append([
                _cell(s.strategy),
                _cell(str(s.trade_count)),
                _cell(f"{s.win_rate:.0%}"),
                _cell(f"{s.profit_factor:.2f}"),
                _cell(f"{s.avg_r:+.2f}"),
                _cell(f"${s.total_pnl:+,.2f}"),
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
                _cell(f"{r.win_rate:.0%}"),
                _cell(f"{r.profit_factor:.2f}"),
                _cell(f"{r.avg_r:+.2f}"),
                _cell(f"${r.total_pnl:+,.2f}"),
            ])
        t = Table(rows, colWidths=breakdown_widths)
        t.setStyle(breakdown_style)
        story.append(KeepTogether([
            Paragraph("Regime Breakdown", styles["SectionHead"]),
            t, Spacer(1, 12),
        ]))

    # ── Trade log table ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Trade Log", styles["SectionHead"]))

    log_hdr = [
        _hdr("Ticker"), _hdr("Strategy"), _hdr("Entry"), _hdr("Exit"),
        _hdr("P&amp;L"), _hdr("R"), _hdr("Days"), _hdr("Exit Reason"),
    ]
    rows = [log_hdr]
    for t_entry in trades:
        pnl_color = "#22c55e" if t_entry.realized_pnl >= 0 else "#ef4444"
        rows.append([
            _cell(t_entry.ticker),
            _cell(t_entry.strategy[:16]),
            _cell(f"${t_entry.entry_fill_price:.2f}"
                  if t_entry.entry_fill_price else "-"),
            _cell(f"${t_entry.exit_price:.2f}"
                  if t_entry.exit_price else "-"),
            Paragraph(
                f'<font color="{pnl_color}">${t_entry.realized_pnl:+.2f}</font>',
                styles["CellSmall"],
            ),
            _cell(f"{t_entry.r_multiple:+.2f}"
                  if t_entry.r_multiple else "-"),
            _cell(str(t_entry.holding_days)
                  if t_entry.holding_days else "-"),
            _cell(t_entry.exit_reason or "-"),
        ])

    col_widths = [width * 0.09, width * 0.15, width * 0.10, width * 0.10,
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
