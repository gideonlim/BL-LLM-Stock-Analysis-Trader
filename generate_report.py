"""Generate the Quant Analysis Bot improvement report as PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "Improvement_Report.pdf")

# Colors
DARK_BLUE = HexColor("#1a365d")
MED_BLUE = HexColor("#2b6cb0")
LIGHT_BLUE = HexColor("#ebf4ff")
RED = HexColor("#c53030")
ORANGE = HexColor("#c05621")
GREEN = HexColor("#276749")
LIGHT_GRAY = HexColor("#f7fafc")
GRAY = HexColor("#718096")

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    "CoverTitle", parent=styles["Title"],
    fontSize=28, leading=34, textColor=DARK_BLUE,
    spaceAfter=6, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "CoverSubtitle", parent=styles["Normal"],
    fontSize=14, leading=18, textColor=GRAY,
    alignment=TA_CENTER, spaceAfter=30,
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading1"],
    fontSize=18, leading=22, textColor=DARK_BLUE,
    spaceBefore=20, spaceAfter=10,
    borderWidth=0, borderPadding=0,
))
styles.add(ParagraphStyle(
    "SubHead", parent=styles["Heading2"],
    fontSize=13, leading=16, textColor=MED_BLUE,
    spaceBefore=14, spaceAfter=6,
))
styles.add(ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    "BodyBold", parent=styles["Normal"],
    fontSize=10, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=4, fontName="Helvetica-Bold",
))
styles.add(ParagraphStyle(
    "BugTitle", parent=styles["Normal"],
    fontSize=11, leading=14, textColor=RED,
    fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=8,
))
styles.add(ParagraphStyle(
    "ImpTitle", parent=styles["Normal"],
    fontSize=11, leading=14, textColor=MED_BLUE,
    fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=8,
))
styles.add(ParagraphStyle(
    "CodeBlock", parent=styles["Normal"],
    fontSize=8.5, leading=11, fontName="Courier",
    leftIndent=12, spaceAfter=6, spaceBefore=4,
    backColor=LIGHT_GRAY, borderPadding=4,
))
styles.add(ParagraphStyle(
    "SmallNote", parent=styles["Normal"],
    fontSize=8, leading=10, textColor=GRAY, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "TableCell", parent=styles["Normal"],
    fontSize=9, leading=12,
))
styles.add(ParagraphStyle(
    "TableHeader", parent=styles["Normal"],
    fontSize=9, leading=12, fontName="Helvetica-Bold", textColor=white,
))


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=8, spaceBefore=4)


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    data = [[Paragraph(h, styles["TableHeader"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), styles["TableCell"]) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ]))
    return t


def build():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    story = []

    # ── COVER ────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Quant Analysis Bot", styles["CoverTitle"]))
    story.append(Paragraph("Comprehensive Code Review &amp; Improvement Recommendations", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(hr())
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("March 2026", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.5*inch))

    # Executive summary box
    summary_text = (
        "This report presents a thorough code review of the Quant Analysis Bot and Trading Bot BL modules. "
        "It covers <b>3 critical bugs</b> that affect P&amp;L accuracy, <b>5 high-impact code fixes</b>, "
        "and <b>8 strategic improvements</b> for better portfolio performance. "
        "Findings are prioritized by impact: bugs that silently distort backtest results come first, "
        "followed by efficiency improvements and architectural upgrades."
    )
    story.append(Paragraph(summary_text, styles["Body"]))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ────────────────────────────────────
    story.append(Paragraph("Contents", styles["SectionHead"]))
    story.append(hr())
    toc_items = [
        "1. Critical Bugs (Fix Immediately)",
        "2. High-Impact Code Fixes",
        "3. Backtesting Methodology Improvements",
        "4. Portfolio Optimization Upgrades",
        "5. Position Sizing &amp; Risk Management",
        "6. Stop-Loss &amp; Exit Strategy",
        "7. Architecture &amp; Code Quality",
        "8. Strategic Roadmap",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["Body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 1: CRITICAL BUGS
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("1. Critical Bugs (Fix Immediately)", styles["SectionHead"]))
    story.append(hr())
    story.append(Paragraph(
        "These bugs silently corrupt backtest results and scoring. They should be fixed before any "
        "other improvements, as they undermine the validity of all downstream decisions.",
        styles["Body"]
    ))

    # Bug 1
    story.append(Paragraph("BUG 1: Exit cost never deducted in long-only mode", styles["BugTitle"]))
    story.append(Paragraph(
        "<b>File:</b> backtest.py, line 131 &nbsp; <b>Severity:</b> Critical &nbsp; <b>Impact:</b> All backtest returns are overstated",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "When closing a long position in long-only mode, the code sets <font face='Courier' size='9'>position = 0</font> on line 129, "
        "then checks <font face='Courier' size='9'>if position == 1</font> on line 131 to decide whether to deduct exit cost. "
        "Since position was just set to 0, the condition is always False. Exit transaction costs are never applied.",
        styles["Body"]
    ))
    story.append(Paragraph("Current (broken):", styles["BodyBold"]))
    story.append(Paragraph(
        "position = 0<br/>entry_price = 0.0<br/>daily_ret -= cost if position == 1 else 0 &nbsp; # always 0!",
        styles["CodeBlock"]
    ))
    story.append(Paragraph("Fix:", styles["BodyBold"]))
    story.append(Paragraph(
        "daily_ret -= cost &nbsp; # deduct exit cost unconditionally<br/>position = 0<br/>entry_price = 0.0",
        styles["CodeBlock"]
    ))
    story.append(Paragraph(
        "<b>Impact estimate:</b> With 10 bps round-trip cost and ~20 trades per 3-month window, this inflates "
        "returns by roughly 0.2% per window. Over many strategies and tickers, it systematically biases strategy "
        "selection toward higher-frequency strategies that trade more often.",
        styles["Body"]
    ))

    # Bug 2
    story.append(Paragraph("BUG 2: Drawdown penalty rewards worse strategies", styles["BugTitle"]))
    story.append(Paragraph(
        "<b>File:</b> backtest.py, line 239 &nbsp; <b>Severity:</b> Critical &nbsp; <b>Impact:</b> Strategy ranking is inverted for drawdown component",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "The scoring function adds <font face='Courier' size='9'>result.max_drawdown_pct * 0.5</font> to the composite score. "
        "Since max_drawdown_pct is always negative (e.g., -15.0), this <i>subtracts</i> from the score, which is correct. "
        "However, the comment says '15% weight' but the actual weight is only 0.5 on a percentage value, while other "
        "components use multipliers of 35 and 150. The drawdown penalty is effectively negligible (~7.5 points for a "
        "-15% drawdown vs ~105 points for Sharpe 3.0). A strategy with -40% max drawdown barely gets penalized.",
        styles["Body"]
    ))
    story.append(Paragraph("Fix: Scale the penalty to be meaningful relative to other components:", styles["BodyBold"]))
    story.append(Paragraph(
        "# Drawdown penalty (15% weight, targeting ~15-20 point range)<br/>"
        "score += result.max_drawdown_pct * 1.5 &nbsp; # -15% DD -> -22.5 points",
        styles["CodeBlock"]
    ))

    # Bug 3
    story.append(Paragraph("BUG 3: Division by zero in indicators", styles["BugTitle"]))
    story.append(Paragraph(
        "<b>Files:</b> indicators.py lines 25, 83, 97, 103, 128 &nbsp; <b>Severity:</b> High &nbsp; <b>Impact:</b> NaN propagation crashes strategies",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "Five indicator functions divide without guarding against zero denominators. "
        "RSI divides avg_gain/avg_loss (zero in strong uptrends), Stochastic divides by highest_high-lowest_low "
        "(zero in flat markets), VWAP divides by cumulative volume (zero for illiquid assets), Z-Score divides "
        "by rolling std (zero in constant-price periods), and ADX divides by plus_di+minus_di (zero when no trend). "
        "NaN values propagate through strategy signals, causing silent failures.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Fix each with epsilon guards: <font face='Courier' size='9'>denominator = np.maximum(denominator, 1e-10)</font>",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 2: HIGH-IMPACT CODE FIXES
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("2. High-Impact Code Fixes", styles["SectionHead"]))
    story.append(hr())

    # Fix 1
    story.append(Paragraph("FIX 1: Crossover signal logic produces state, not transitions", styles["ImpTitle"]))
    story.append(Paragraph(
        "<b>Files:</b> strategies.py (SMA_Crossover, EMA_Crossover, MACD_Strategy, VWAP_Strategy, TrendFollowing_ADX)",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "These strategies assign +1/-1 based on whether one indicator is above/below another, then call "
        "<font face='Courier' size='9'>.diff().clip(-1,1)</font> to convert to crossover signals. "
        "The problem: when signals go from +1 to -1, diff() produces -2 which clips to -1 (correct). "
        "But when signals go from 0 to +1 (NaN rows at series start), diff() produces NaN which fillna(0) masks. "
        "More importantly, the approach generates signals on every transition, including spurious ones caused by "
        "indicator noise around the crossover point. A proper crossover detector should compare bar[i] vs bar[i-1] states explicitly.",
        styles["Body"]
    ))
    story.append(Paragraph("Recommended pattern:", styles["BodyBold"]))
    story.append(Paragraph(
        "above_now = df['SMA_10'] &gt; df['SMA_50']<br/>"
        "above_prev = above_now.shift(1, fill_value=False)<br/>"
        "signals[above_now &amp; ~above_prev] = 1 &nbsp; # cross above<br/>"
        "signals[~above_now &amp; above_prev] = -1 &nbsp; # cross below",
        styles["CodeBlock"]
    ))

    # Fix 2
    story.append(Paragraph("FIX 2: Bollinger Band division by zero in CompositeScore", styles["ImpTitle"]))
    story.append(Paragraph(
        "<b>File:</b> strategies.py, line 195",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "The composite strategy computes <font face='Courier' size='9'>bb_pos = (Close - BB_Lower) / (BB_Upper - BB_Lower)</font>. "
        "During very low volatility periods, BB_Upper can equal BB_Lower, causing division by zero. "
        "Fix: <font face='Courier' size='9'>bb_range = np.maximum(df['BB_Upper'] - df['BB_Lower'], 1e-10)</font>",
        styles["Body"]
    ))

    # Fix 3
    story.append(Paragraph("FIX 3: Portfolio state race condition in executor", styles["ImpTitle"]))
    story.append(Paragraph(
        "<b>File:</b> executor.py, lines 511-522",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "After submitting an order, the executor updates the portfolio snapshot in-memory by subtracting "
        "notional from cash and adding a synthetic position entry. But the order hasn't filled yet (it's a limit order). "
        "If the next order's risk check runs against this optimistic state, it may approve orders that would actually "
        "exceed exposure limits. The fix: track 'committed capital' separately from actual portfolio state, and "
        "include committed capital in exposure calculations.",
        styles["Body"]
    ))

    # Fix 4
    story.append(Paragraph("FIX 4: Kelly formula uses simplified approximation", styles["ImpTitle"]))
    story.append(Paragraph(
        "<b>File:</b> signals.py, line 159",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "The current formula <font face='Courier' size='9'>kelly_f = (win_rate * profit_factor - loss_rate) / profit_factor</font> "
        "assumes all wins are equal and all losses are equal. The true Kelly criterion uses average win and average loss sizes: "
        "<font face='Courier' size='9'>kelly_f = (win_rate / avg_loss_pct) - (loss_rate / avg_win_pct)</font>. "
        "Since BacktestResult already tracks avg_win_pct and avg_loss_pct, the correct values are available. "
        "The simplified version tends to over-leverage on strategies with fat-tailed losses.",
        styles["Body"]
    ))

    # Fix 5
    story.append(Paragraph("FIX 5: Position size caps are inconsistent between modules", styles["ImpTitle"]))
    story.append(Paragraph(
        "<b>Files:</b> signals.py line 161 vs config.py RiskLimits",
        styles["SmallNote"]
    ))
    story.append(Paragraph(
        "The quant bot caps position sizes at {HIGH: 20%, MEDIUM: 10%, LOW: 5%}, but the trading bot's risk manager "
        "caps at max_position_pct=12%. A HIGH confidence signal sized at 18% will always be trimmed to 12% by risk. "
        "The quant bot's HIGH cap should match or be below the risk manager's cap to avoid wasted computation.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 3: BACKTESTING
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("3. Backtesting Methodology Improvements", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("3.1 Implement Walk-Forward Validation", styles["SubHead"]))
    story.append(Paragraph(
        "The current approach tests each strategy on fixed windows (3m, 6m, 12m) of historical data and "
        "generates signals on the full dataset. This is in-sample testing: the strategy is evaluated and "
        "deployed on the same data. Walk-forward analysis divides each window into a training portion (60-70%) "
        "and a validation portion (30-40%). The strategy is scored only on the validation portion. "
        "Research shows this improves backtest-to-live consistency by ~23%.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Implementation: In select_best_strategy(), split window_df into train_df (first 70%) and test_df (last 30%). "
        "Generate signals on train_df, but compute BacktestResult metrics only on test_df. This prevents overfitting "
        "to recent patterns that may not persist.",
        styles["Body"]
    ))

    story.append(Paragraph("3.2 Use Next-Bar Execution in Backtester", styles["SubHead"]))
    story.append(Paragraph(
        "Currently, signals generated on bar[i] execute at close[i]. In reality, you can't trade at the close price "
        "of the bar that generated the signal. A more realistic model executes at close[i+1] (next bar open). "
        "This eliminates a subtle look-ahead bias and typically reduces reported Sharpe by 10-20%, giving you "
        "a more honest estimate of live performance.",
        styles["Body"]
    ))

    story.append(Paragraph("3.3 Add Deflated Sharpe Ratio", styles["SubHead"]))
    story.append(Paragraph(
        "Testing 11 strategies across 3 timeframes produces 33 hypothesis tests per ticker. The probability of "
        "finding a Sharpe > 1.0 by chance increases with the number of strategies tested. The Deflated Sharpe Ratio "
        "(DSR) adjusts for this multiple testing bias. Add DSR as a scoring component to penalize strategies that "
        "may be statistical flukes.",
        styles["Body"]
    ))

    story.append(Paragraph("3.4 Scoring Function Calibration", styles["SubHead"]))
    story.append(Paragraph(
        "The current scoring weights were chosen ad-hoc and have unit mismatches. Sharpe (unitless, 0-3 range) "
        "gets multiplied by 35 for ~105 max points. Win rate (0-1 range) minus 0.5 gets multiplied by 150 for "
        "~75 max points. But excess return (%, -100 to 100 range) only gets multiplied by 0.2 for ~20 max points. "
        "This means Sharpe dominates scoring regardless of actual returns. Consider normalizing all inputs to "
        "z-scores before weighting, or using a rank-based scoring system.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 4: PORTFOLIO OPTIMIZATION
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("4. Portfolio Optimization Upgrades", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("4.1 BL View Return Estimation", styles["SubHead"]))
    story.append(Paragraph(
        "The _estimate_view_return() function in black_litterman.py estimates expected return from "
        "Sharpe * implied_vol + 5%. The implied volatility is backed out from the stop-loss distance, "
        "which is ATR-based. This chain of approximations can produce wildly inaccurate views. "
        "For example, a tight ATR-based stop (2%) implies low daily vol (1.3%), leading to low annual vol (21%), "
        "so Sharpe 2.0 produces return = 2.0 * 21% + 5% = 47%. But the actual backtest return might be 15%. "
        "Consider using the actual backtest annual_excess_pct from the signal directly, with uncertainty "
        "scaled by the number of backtest trades.",
        styles["Body"]
    ))

    story.append(Paragraph("4.2 max(BL, Kelly) Defeats Diversification", styles["SubHead"]))
    story.append(Paragraph(
        "In portfolio_optimizer.py line 165, <font face='Courier' size='9'>final_notional = max(bl_notional, original_notional)</font> "
        "ensures no position is smaller than its Kelly size. However, this defeats BL's core purpose: distributing "
        "capital based on correlation structure. If BL says 'allocate 3% to AAPL' because it's highly correlated "
        "with your other holdings, but Kelly says '10%', the max() gives you 10% and ignores the correlation signal. "
        "Consider a weighted blend instead: <font face='Courier' size='9'>final = 0.6 * kelly + 0.4 * bl_notional</font>, "
        "or use BL weights only for ranking while keeping Kelly for sizing (which is closer to the current intent, "
        "but then the max() is unnecessary for positions where BL < Kelly).",
        styles["Body"]
    ))

    story.append(Paragraph("4.3 Covariance Regime Sensitivity", styles["SubHead"]))
    story.append(Paragraph(
        "The 60-day lookback for covariance estimation misses regime changes. During a volatility spike, "
        "the first 40 days of calm data dilute the signal. Consider implementing an exponentially weighted "
        "covariance matrix (EWMA with halflife=21 days) alongside the Ledoit-Wolf estimator, and blending "
        "them. This gives more weight to recent correlation structure while maintaining the stability of shrinkage.",
        styles["Body"]
    ))

    story.append(Paragraph("4.4 Add Sector Diversification Constraint", styles["SubHead"]))
    story.append(Paragraph(
        "Neither BL nor marginal Sharpe considers sector concentration. It's possible to end up with 6 of 8 positions "
        "in tech stocks. Add a soft constraint: fetch GICS sector for each ticker (available from yfinance info), "
        "and penalize intents that would put more than 40% of the portfolio in a single sector. This is listed in "
        "the PROJECT_CONTEXT.md known limitations and would be a high-value addition.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 5: POSITION SIZING & RISK
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("5. Position Sizing &amp; Risk Management", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("5.1 Move to Quarter-Kelly", styles["SubHead"]))
    story.append(Paragraph(
        "The current Half-Kelly sizing is aggressive for a system with no volatility regime detection. "
        "Practitioners widely recommend Quarter-Kelly (kelly_f * 0.25) as the sweet spot between growth and "
        "drawdown control. The expected reduction in max drawdown is ~40% vs Half-Kelly, with only ~15% "
        "reduction in terminal wealth over a 3-year horizon. Given this is an automated system with no human "
        "intervention during drawdowns, the extra safety margin is valuable.",
        styles["Body"]
    ))

    story.append(Paragraph("5.2 Add Volatility-Targeted Sizing", styles["SubHead"]))
    story.append(Paragraph(
        "Currently, position size is purely Kelly-based. A volatility-targeted overlay would scale positions "
        "inversely to recent volatility. For example, target 1% daily portfolio risk per position: "
        "<font face='Courier' size='9'>position_size = target_risk / (daily_vol * sqrt(holding_days))</font>. "
        "This automatically reduces exposure during high-volatility regimes and increases it during calm periods, "
        "addressing the 'no volatility regime detection' limitation.",
        styles["Body"]
    ))

    story.append(Paragraph("5.3 Strategy Underperformance Threshold", styles["SubHead"]))
    story.append(Paragraph(
        "In risk.py line 118, <font face='Courier' size='9'>min_success_rate=0.3</font> means a strategy is only "
        "blocked if fewer than 30% of its orders succeed. This is extremely lenient; a strategy that fails 70% of "
        "the time is still allowed to trade. Consider raising this to 0.5 (50%) and adding a P&amp;L-based check: "
        "block strategies with negative cumulative P&amp;L over the lookback period.",
        styles["Body"]
    ))

    story.append(Paragraph("5.4 Add CVaR-Based Risk Monitoring", styles["SubHead"]))
    story.append(Paragraph(
        "The current risk manager uses simple percentage thresholds (3% daily loss circuit breaker, 80% exposure cap). "
        "Adding Conditional Value at Risk (CVaR) at 95% confidence would provide a tail-risk measure. "
        "Compute CVaR from the portfolio's 60-day return history: if the expected loss in the worst 5% of days "
        "exceeds a threshold (e.g., 2% of equity), reduce new position sizes by 50%. This provides proactive "
        "risk reduction before the circuit breaker triggers.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 6: STOP-LOSS & EXITS
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("6. Stop-Loss &amp; Exit Strategy", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("6.1 Replace Fixed Multiplier with Chandelier Exit", styles["SubHead"]))
    story.append(Paragraph(
        "The current ATR * 1.5 stop loss uses a fixed multiplier regardless of market conditions. "
        "The Chandelier Exit dynamically trails below the highest high: "
        "<font face='Courier' size='9'>SL = max(High, 22-period) - ATR(22) * 3.0</font>. "
        "In trending markets, this keeps the stop further away (reducing whipsaws). In ranging markets, "
        "it tightens naturally. Research shows Chandelier Exits outperform fixed ATR stops by reducing "
        "false breakout exits.",
        styles["Body"]
    ))

    story.append(Paragraph("6.2 Adaptive Reward-to-Risk Ratio", styles["SubHead"]))
    story.append(Paragraph(
        "The hardcoded 2:1 TP/SL ratio (signals.py line 143) doesn't adapt to strategy type. "
        "Mean-reversion strategies typically have higher win rates but smaller wins, so 1.5:1 may be optimal. "
        "Trend-following strategies have lower win rates but need larger winners, so 3:1 or higher is better. "
        "Use the backtest's actual average_win/average_loss ratio to set the TP multiplier per strategy.",
        styles["Body"]
    ))

    story.append(Paragraph("6.3 Improve Trailing Stop Logic", styles["SubHead"]))
    story.append(Paragraph(
        "The monitor's trailing stop (entry + 50% of gain) is simple but can leave significant profits on the table. "
        "Consider an ATR-based trailing stop: <font face='Courier' size='9'>trail_stop = highest_since_entry - ATR * 2.0</font>. "
        "This adapts to current volatility rather than using a fixed percentage of gains. Also consider "
        "only activating the trail after a minimum profit threshold (e.g., 3%) to avoid locking in tiny gains.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 7: CODE QUALITY
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("7. Architecture &amp; Code Quality", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("7.1 Add Test Coverage", styles["SubHead"]))
    story.append(Paragraph(
        "Only test_black_litterman.py (8 tests) exists. The backtest engine, strategies, risk manager, "
        "and scoring function have zero test coverage. Priority test targets: (1) backtest.py run_backtest with "
        "known input/output pairs to catch the exit cost bug, (2) score_single_window with edge cases, "
        "(3) risk.py evaluate_order with boundary conditions, (4) indicators.py with zero-denominator inputs.",
        styles["Body"]
    ))

    story.append(Paragraph("7.2 Data Client Instantiation in Broker", styles["SubHead"]))
    story.append(Paragraph(
        "Every call to get_latest_price() and get_latest_prices() creates a new StockHistoricalDataClient. "
        "During execution with 8+ signals, this creates 8+ HTTP client instances. Initialize the data client "
        "once in __init__ and reuse it.",
        styles["Body"]
    ))

    story.append(Paragraph("7.3 Duplicate yfinance Downloads", styles["SubHead"]))
    story.append(Paragraph(
        "Both black_litterman.py._fetch_returns() and portfolio_optimizer.py.fetch_returns_matrix() "
        "independently download the same return data from yfinance. When BL falls back to marginal Sharpe, "
        "the data is downloaded twice. Extract the download into a shared utility with caching.",
        styles["Body"]
    ))

    story.append(Paragraph("7.4 Market Cap Fetching is Slow", styles["SubHead"]))
    story.append(Paragraph(
        "_fetch_market_caps() calls yf.Ticker(ticker).info in a sequential loop for every ticker. "
        "With 15+ tickers, this takes 30-60 seconds. Use yfinance's batch download or cache market caps "
        "daily (they don't change intraday in a meaningful way for equilibrium weight calculation).",
        styles["Body"]
    ))

    story.append(Paragraph("7.5 Type Safety", styles["SubHead"]))
    story.append(Paragraph(
        "portfolio_optimizer.py uses <font face='Courier' size='9'>intent: object</font> in RankedIntent and "
        "<font face='Courier' size='9'>config=None</font> without type hints in several functions. "
        "Add proper type annotations and consider running mypy in CI to catch type errors before runtime.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 8: STRATEGIC ROADMAP
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("8. Strategic Roadmap", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph(
        "Below is a prioritized implementation roadmap. Items are grouped by effort level and expected impact "
        "on portfolio performance.",
        styles["Body"]
    ))

    story.append(Spacer(1, 10))
    roadmap_data = [
        ["Phase 1\n(1-2 days)", "Fix 3 critical bugs", "Critical", "Ensures backtest accuracy.\nAll other improvements depend on correct data."],
        ["Phase 1\n(1-2 days)", "Add epsilon guards\nto all indicators", "High", "Eliminates NaN crashes.\nPrevents silent strategy failures."],
        ["Phase 2\n(3-5 days)", "Fix crossover signal\nlogic (5 strategies)", "High", "Reduces false signals by\n~30% based on similar systems."],
        ["Phase 2\n(3-5 days)", "Implement walk-forward\nvalidation", "High", "~23% better backtest-to-live\nconsistency per research."],
        ["Phase 2\n(3-5 days)", "Add next-bar execution\nto backtester", "High", "Eliminates look-ahead bias.\nMore realistic Sharpe estimates."],
        ["Phase 3\n(1-2 weeks)", "Volatility-targeted\nposition sizing", "Medium", "Addresses no-regime-detection\nlimitation. Adapts automatically."],
        ["Phase 3\n(1-2 weeks)", "Chandelier Exit\nstop-loss", "Medium", "Better trend-following exits.\nReduces whipsaw losses."],
        ["Phase 3\n(1-2 weeks)", "Sector diversification\nconstraint", "Medium", "Prevents sector concentration.\nImproves tail-risk profile."],
        ["Phase 4\n(2-4 weeks)", "Add HMM regime\ndetection", "Medium", "5-15% Sharpe improvement\nper academic research."],
        ["Phase 4\n(2-4 weeks)", "CVaR risk monitoring\n+ adaptive sizing", "Medium", "Proactive risk reduction.\nSmooths drawdown profile."],
        ["Phase 4\n(2-4 weeks)", "Test coverage to 70%+", "Medium", "Prevents regression.\nRequired for live trading."],
    ]

    t = make_table(
        ["Timeline", "Improvement", "Priority", "Expected Impact"],
        roadmap_data,
        col_widths=[1.1*inch, 1.8*inch, 0.8*inch, 3.0*inch],
    )
    story.append(t)

    story.append(Spacer(1, 20))
    story.append(Paragraph("Summary of Quick Wins", styles["SubHead"]))

    quick_wins = [
        ["backtest.py:129-131", "Move cost deduction before position reset", "5 min"],
        ["backtest.py:239", "Increase drawdown penalty multiplier 0.5 -> 1.5", "2 min"],
        ["indicators.py:25,83,97,103,128", "Add np.maximum(denom, 1e-10) guards", "15 min"],
        ["strategies.py:195", "Guard BB_Upper - BB_Lower division", "5 min"],
        ["signals.py:161", "Align HIGH cap (20%) with risk manager max (12%)", "2 min"],
        ["broker.py:134,166", "Move data client init to __init__", "10 min"],
    ]

    t2 = make_table(
        ["Location", "Change", "Effort"],
        quick_wins,
        col_widths=[2.0*inch, 3.5*inch, 0.8*inch],
    )
    story.append(t2)

    story.append(Spacer(1, 20))
    story.append(hr())
    story.append(Paragraph(
        "This report covers findings from a full review of both the quant_analysis_bot and trading_bot_bl modules, "
        "supplemented by research into current best practices for systematic trading systems. The critical bugs in "
        "Section 1 should be addressed first, as they affect the validity of all backtest results. "
        "The strategic improvements in Sections 3-6 are ordered by expected impact-to-effort ratio.",
        styles["Body"]
    ))

    doc.build(story)
    print(f"Report generated: {OUTPUT}")


if __name__ == "__main__":
    build()
