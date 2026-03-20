"""Generate PDF performance reports with charts from journal data.

Uses matplotlib for chart generation and reportlab for PDF assembly.
All chart images are rendered in-memory (no temp files needed).
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Optional

from trading_bot_bl.journal_analytics import (
    JournalMetrics,
    compute_journal_metrics,
)
from trading_bot_bl.models import EquitySnapshot, JournalEntry

log = logging.getLogger(__name__)


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


def _chart_equity_curve(snapshots: list[EquitySnapshot]) -> Optional[io.BytesIO]:
    """Equity over time with drawdown shading."""
    if len(snapshots) < 2:
        return None

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime as dt

    dates = []
    equity = []
    hwm = []
    for s in snapshots:
        try:
            dates.append(dt.fromisoformat(s.timestamp))
        except Exception:
            dates.append(dt.now())
        equity.append(s.equity)
        hwm.append(s.high_water_mark)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # Equity + HWM
    ax1.plot(dates, equity, color="#2563eb", linewidth=1.5, label="Equity")
    ax1.plot(dates, hwm, color="#94a3b8", linewidth=1, linestyle="--",
             label="High Water Mark", alpha=0.7)
    ax1.fill_between(dates, equity, hwm, alpha=0.08, color="red",
                     where=[e < h for e, h in zip(equity, hwm)])
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")

    # Drawdown
    dd = [s.drawdown_pct for s in snapshots]
    ax2.fill_between(dates, dd, 0, alpha=0.4, color="#ef4444")
    ax2.plot(dates, dd, color="#ef4444", linewidth=1)
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
    )

    trades, snapshots, open_count, pending_count = _load_journal_data(log_dir)
    metrics = compute_journal_metrics(trades, snapshots) if trades else None

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
    # Equity curve
    eq_buf = _chart_equity_curve(snapshots)
    if eq_buf:
        story.append(Paragraph("Equity Curve &amp; Drawdown", styles["SectionHead"]))
        story.append(Image(eq_buf, width=width, height=width * 0.5))
        story.append(Spacer(1, 12))

    # Cumulative P&L
    cum_buf = _chart_cumulative_pnl(trades)
    if cum_buf:
        story.append(Paragraph("Cumulative P&amp;L", styles["SectionHead"]))
        story.append(Image(cum_buf, width=width, height=width * 0.3))
        story.append(Spacer(1, 12))

    # P&L distribution
    pnl_buf = _chart_pnl_distribution(trades)
    if pnl_buf:
        story.append(PageBreak())
        story.append(Paragraph("P&amp;L &amp; R-Multiple Distribution", styles["SectionHead"]))
        story.append(Image(pnl_buf, width=width, height=width * 0.35))
        story.append(Spacer(1, 12))

    # Win/loss + strategy
    wl_buf = _chart_win_loss(metrics)
    if wl_buf:
        story.append(Paragraph("Win Rate &amp; Strategy Attribution", styles["SectionHead"]))
        story.append(Image(wl_buf, width=width, height=width * 0.35))
        story.append(Spacer(1, 12))

    # ── R-Distribution table ────────────────────────────────────
    rd = metrics.r_distribution
    story.append(Paragraph("R-Distribution", styles["SectionHead"]))
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
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Streak analysis ─────────────────────────────────────────
    st = metrics.streaks
    story.append(Paragraph("Streak Analysis", styles["SectionHead"]))
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
    story.append(t)
    story.append(Spacer(1, 12))

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
        story.append(Paragraph("Strategy Breakdown", styles["SectionHead"]))
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
        story.append(t)
        story.append(Spacer(1, 12))

    # ── Regime breakdown table ──────────────────────────────────
    if metrics.by_regime:
        story.append(Paragraph("Regime Breakdown", styles["SectionHead"]))
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
        story.append(t)
        story.append(Spacer(1, 12))

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
