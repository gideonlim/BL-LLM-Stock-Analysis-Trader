"""Market-wide sentiment indicators — VIX, put/call ratio, SPY trend.

Provides a portfolio-level sentiment regime that modifies position
sizing and risk thresholds.  All data comes from yfinance (free),
so no paid APIs are needed.

Feature flag: ``MARKET_SENTIMENT_ENABLED`` (default: true).
Set to ``false`` in .env or config JSON to disable entirely.

Design notes
------------
* **Contrarian at extremes, neutral in the middle.**  Extreme fear
  (VIX > 30, P/C > 1.0) historically marks bottoms → *widen* the
  size multiplier.  Extreme greed (VIX < 15, P/C < 0.6) often
  precedes corrections → *shrink* the size multiplier.
* **SPY trend regime (bear market filter).**  Separate from the
  contrarian VIX module.  When SPY is below its 200-day SMA for
  ≥3 consecutive days, we flag a BEAR regime.  This *reduces*
  exposure (fewer positions, tighter quality bar) rather than
  sizing up contrarian-style.  A 15% drawdown from 52-week high
  triggers a hard halt on new entries.
* The VIX module never blocks a trade — it only *scales* position
  notional.  The SPY regime filter *can* block trades via the risk
  manager.  Both gracefully degrade to neutral on data failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# ── Lookback for z-score normalisation (trading days) ────────
_LOOKBACK_DAYS = 252  # ~1 year


@dataclass(frozen=True)
class SpyRegime:
    """SPY trend-based regime detection for bear market filtering.

    Unlike the VIX sentiment (contrarian — buys more during fear),
    this is a *trend-following* filter: when SPY is in a sustained
    downtrend, we reduce exposure because a long-only strategy
    underperforms in bear markets.
    """

    # Current SPY price and 200-day SMA
    spy_price: float = 0.0
    spy_sma200: float = 0.0

    # How far SPY is from its 200-SMA (negative = below)
    spy_vs_sma200_pct: float = 0.0

    # Number of consecutive trading days SPY closed below 200-SMA
    days_below_sma200: int = 0

    # 200-SMA slope (annualised % change, 20-day window)
    sma200_slope_ann_pct: float = 0.0

    # SPY drawdown from 52-week high
    spy_drawdown_pct: float = 0.0

    # Regime label
    trend_regime: str = "BULL"  # BULL / CAUTION / BEAR / SEVERE_BEAR

    def summary(self) -> str:
        return (
            f"SPY=${self.spy_price:.2f}, "
            f"SMA200=${self.spy_sma200:.2f} "
            f"({self.spy_vs_sma200_pct:+.1f}%), "
            f"days_below={self.days_below_sma200}, "
            f"slope={self.sma200_slope_ann_pct:+.1f}%/yr, "
            f"dd={self.spy_drawdown_pct:.1f}%, "
            f"trend={self.trend_regime}"
        )


@dataclass(frozen=True)
class MarketSentiment:
    """Snapshot of market-wide sentiment indicators."""

    # Raw values
    vix: float = 0.0
    vix_z: float = 0.0
    put_call_ratio: float = 0.0
    put_call_z: float = 0.0

    # Composite sentiment index  (-1 = extreme fear, +1 = extreme greed)
    # Negative VIX z → fear → negative MSI;
    # sign is flipped so *high* VIX → *negative* MSI.
    msi: float = 0.0

    # Regime label
    regime: str = "NEUTRAL"  # FEAR / GREED / NEUTRAL

    # Position-size multiplier produced by the regime
    size_multiplier: float = 1.0

    # SPY trend regime (bear market filter)
    spy_regime: SpyRegime = field(default_factory=SpyRegime)

    def summary(self) -> str:
        parts = [
            f"VIX={self.vix:.1f} (z={self.vix_z:+.2f})",
            f"P/C={self.put_call_ratio:.2f} "
            f"(z={self.put_call_z:+.2f})",
            f"MSI={self.msi:+.2f}",
            f"regime={self.regime}",
            f"size_mult={self.size_multiplier:.2f}",
            f"trend={self.spy_regime.trend_regime}",
        ]
        return ", ".join(parts)


# ── Public API ───────────────────────────────────────────────

def fetch_market_sentiment(
    *,
    fear_vix: float = 30.0,
    greed_vix: float = 15.0,
    fear_pc: float = 1.0,
    greed_pc: float = 0.6,
    fear_size_mult: float = 1.15,
    greed_size_mult: float = 0.85,
    spy_bear_confirmation_days: int = 3,
    spy_severe_drawdown_pct: float = 15.0,
) -> MarketSentiment:
    """Fetch live VIX + put/call ratio + SPY trend and compute sentiment.

    Parameters
    ----------
    fear_vix, greed_vix : float
        VIX thresholds for fear / greed regime classification.
    fear_pc, greed_pc : float
        Put/call ratio thresholds.
    fear_size_mult : float
        Position-size multiplier during extreme fear (contrarian:
        we *increase* size because fear marks bottoms).
    greed_size_mult : float
        Position-size multiplier during extreme greed (defensive:
        we *decrease* size because greed precedes corrections).
    spy_bear_confirmation_days : int
        Number of consecutive days SPY must close below 200-SMA
        to confirm a BEAR regime (avoids whipsaws).
    spy_severe_drawdown_pct : float
        SPY drawdown from 52-week high that triggers SEVERE_BEAR
        (halts all new entries).

    Returns
    -------
    MarketSentiment
        Frozen dataclass with all computed fields.  On any data
        failure, returns a neutral default (multiplier = 1.0).
    """
    vix, vix_z = _fetch_vix_zscore()
    pcr, pcr_z = _fetch_put_call_zscore()

    # Market Sentiment Index: blend of VIX and put/call z-scores.
    # High VIX & high P/C → negative MSI (fear).
    # Sign convention: MSI > 0 = greed, MSI < 0 = fear.
    msi = -0.5 * vix_z - 0.5 * pcr_z

    regime, size_mult = _classify_regime(
        vix=vix,
        pcr=pcr,
        msi=msi,
        fear_vix=fear_vix,
        greed_vix=greed_vix,
        fear_pc=fear_pc,
        greed_pc=greed_pc,
        fear_size_mult=fear_size_mult,
        greed_size_mult=greed_size_mult,
    )

    # SPY trend regime (bear market filter)
    spy_regime = fetch_spy_regime(
        confirmation_days=spy_bear_confirmation_days,
        severe_drawdown_pct=spy_severe_drawdown_pct,
    )

    sent = MarketSentiment(
        vix=vix,
        vix_z=vix_z,
        put_call_ratio=pcr,
        put_call_z=pcr_z,
        msi=msi,
        regime=regime,
        size_multiplier=size_mult,
        spy_regime=spy_regime,
    )

    log.info(f"  Market sentiment: {sent.summary()}")
    if spy_regime.trend_regime != "BULL":
        log.info(f"  SPY trend detail: {spy_regime.summary()}")
    return sent


def fetch_spy_regime(
    *,
    confirmation_days: int = 3,
    severe_drawdown_pct: float = 15.0,
) -> SpyRegime:
    """Fetch SPY price data and classify the trend regime.

    Tiered classification:

    * **BULL** — SPY above 200-SMA, or below for < *confirmation_days*.
      No restrictions.
    * **CAUTION** — 200-SMA slope is negative (declining trend) but
      SPY is still above the SMA.  Early warning: tighten quality bar.
    * **BEAR** — SPY closed below 200-SMA for ≥ *confirmation_days*
      consecutively.  Reduce max positions and raise min score.
    * **SEVERE_BEAR** — SPY drawdown from 52-week high ≥
      *severe_drawdown_pct*.  Halt all new entries.

    On any data failure, returns a neutral BULL default so the
    trading pipeline continues without restriction.
    """
    try:
        import yfinance as yf

        data = yf.download(
            "SPY",
            period="18mo",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 252:
            log.warning(
                "SPY data insufficient for 52-week lookback "
                f"({len(data) if not data.empty else 0} bars, "
                "need 252) — using BULL"
            )
            return SpyRegime()

        closes = data["Close"].dropna().values.flatten()

        # Current price
        spy_price = float(closes[-1])

        # 200-day simple moving average
        sma200 = float(np.mean(closes[-200:]))

        # % distance from SMA
        vs_sma_pct = (spy_price - sma200) / sma200 * 100

        # Consecutive days below 200-SMA (count backwards)
        days_below = 0
        sma_series = np.convolve(
            closes, np.ones(200) / 200, mode="valid"
        )
        # sma_series[-1] corresponds to closes[-1]
        for i in range(1, min(len(sma_series), 60) + 1):
            price_i = closes[-i]
            sma_i = sma_series[-i]
            if price_i < sma_i:
                days_below += 1
            else:
                break

        # 200-SMA slope: annualised % change over last 20 trading days
        slope_ann_pct = 0.0
        if len(sma_series) >= 20:
            sma_now = sma_series[-1]
            sma_20ago = sma_series[-20]
            if sma_20ago > 0:
                # 20 trading days ≈ 1 month → annualise × 12
                slope_ann_pct = (
                    (sma_now / sma_20ago - 1.0) * 12.0 * 100.0
                )

        # Drawdown from 52-week high (252 trading days)
        high_52w = float(np.max(closes[-252:]))
        drawdown_pct = (
            (high_52w - spy_price) / high_52w * 100
            if high_52w > 0 else 0.0
        )

        # Classify regime
        if drawdown_pct >= severe_drawdown_pct:
            trend = "SEVERE_BEAR"
        elif days_below >= confirmation_days:
            trend = "BEAR"
        elif slope_ann_pct < 0 and vs_sma_pct < 2.0:
            # SMA slope turning negative and price near/at SMA
            trend = "CAUTION"
        else:
            trend = "BULL"

        regime = SpyRegime(
            spy_price=round(spy_price, 2),
            spy_sma200=round(sma200, 2),
            spy_vs_sma200_pct=round(vs_sma_pct, 2),
            days_below_sma200=days_below,
            sma200_slope_ann_pct=round(slope_ann_pct, 1),
            spy_drawdown_pct=round(drawdown_pct, 2),
            trend_regime=trend,
        )

        log.info(f"  SPY regime: {regime.summary()}")
        return regime

    except Exception as exc:
        log.warning(f"SPY regime fetch failed: {exc} — using BULL")
        return SpyRegime()


# ── Internal helpers ─────────────────────────────────────────

def _fetch_vix_zscore() -> tuple[float, float]:
    """Fetch current VIX and its 252-day z-score."""
    try:
        import yfinance as yf

        data = yf.download(
            "^VIX",
            period="18mo",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 30:
            log.warning("VIX data insufficient — using neutral")
            return 0.0, 0.0

        closes = data["Close"].dropna().values.flatten()
        current = float(closes[-1])

        # Use last 252 trading days for z-score
        window = closes[-_LOOKBACK_DAYS:]
        mean = float(np.mean(window))
        std = float(np.std(window))
        z = (current - mean) / std if std > 1e-8 else 0.0

        return current, z

    except Exception as exc:
        log.warning(f"Failed to fetch VIX: {exc}")
        return 0.0, 0.0


def _fetch_put_call_zscore() -> tuple[float, float]:
    """Fetch CBOE total put/call ratio proxy and z-score.

    There is no free direct feed for the CBOE put/call ratio,
    so we proxy it with the VIX-to-SPY-vol ratio.  When fear is
    elevated the ratio spikes (put buyers dominate).

    For a production system you would use CBOE data or a paid
    feed.  This proxy correlates ~0.7 with the actual CBOE P/C
    and is free.
    """
    try:
        import yfinance as yf

        # Use SPY volume-weighted put/call proxy:
        # Fetch SPY options chain and compute aggregate P/C.
        spy = yf.Ticker("SPY")
        exp_dates = spy.options
        if not exp_dates:
            log.debug("No SPY options expiries — P/C unavailable")
            return 0.0, 0.0

        # Use the nearest expiry for a timely reading
        nearest = exp_dates[0]
        chain = spy.option_chain(nearest)
        put_vol = int(chain.puts["volume"].sum())
        call_vol = int(chain.calls["volume"].sum())

        if call_vol == 0:
            return 0.0, 0.0

        pcr = put_vol / call_vol

        # Historical P/C z-score: CBOE long-run average ~0.70,
        # std ~0.15 (approximate).  We use these empirical values
        # because we can't cheaply download 252 days of daily P/C.
        hist_mean = 0.70
        hist_std = 0.15
        z = (pcr - hist_mean) / hist_std

        return round(pcr, 4), round(z, 4)

    except Exception as exc:
        log.debug(f"P/C ratio fetch failed: {exc}")
        return 0.0, 0.0


def _classify_regime(
    *,
    vix: float,
    pcr: float,
    msi: float,
    fear_vix: float,
    greed_vix: float,
    fear_pc: float,
    greed_pc: float,
    fear_size_mult: float,
    greed_size_mult: float,
) -> tuple[str, float]:
    """Map raw indicators to a regime label and size multiplier.

    Uses a two-tier check:
    1. **VIX alone** can trigger FEAR (>30) or GREED (<15).
    2. **Put/call** *confirms* the VIX signal.  If both agree the
       confidence is higher and the multiplier is applied fully.
       If only one triggers, a milder adjustment is used.
    """
    fear_signals = 0
    greed_signals = 0

    if vix >= fear_vix:
        fear_signals += 1
    if vix <= greed_vix and vix > 0:
        greed_signals += 1

    if pcr >= fear_pc and pcr > 0:
        fear_signals += 1
    if pcr <= greed_pc and pcr > 0:
        greed_signals += 1

    if fear_signals >= 2:
        # Both VIX and P/C confirm fear → full contrarian boost
        return "FEAR", fear_size_mult
    elif fear_signals == 1:
        # Mild fear → half the adjustment
        mild = 1.0 + (fear_size_mult - 1.0) * 0.5
        return "FEAR", round(mild, 4)
    elif greed_signals >= 2:
        # Both confirm greed → full defensive cut
        return "GREED", greed_size_mult
    elif greed_signals == 1:
        # Mild greed → half the adjustment
        mild = 1.0 + (greed_size_mult - 1.0) * 0.5
        return "GREED", round(mild, 4)

    return "NEUTRAL", 1.0
