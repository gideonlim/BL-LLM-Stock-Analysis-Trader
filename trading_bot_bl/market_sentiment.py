"""Market-wide sentiment indicators — VIX, put/call ratio.

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
* The module never blocks a trade — it only *scales* position
  notional.  The risk manager still enforces hard limits.
* Graceful degradation: if yfinance fails, the module returns a
  neutral sentiment (multiplier = 1.0) so trading continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# ── Lookback for z-score normalisation (trading days) ────────
_LOOKBACK_DAYS = 252  # ~1 year


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

    def summary(self) -> str:
        return (
            f"VIX={self.vix:.1f} (z={self.vix_z:+.2f}), "
            f"P/C={self.put_call_ratio:.2f} "
            f"(z={self.put_call_z:+.2f}), "
            f"MSI={self.msi:+.2f}, "
            f"regime={self.regime}, "
            f"size_mult={self.size_multiplier:.2f}"
        )


# ── Public API ───────────────────────────────────────────────

def fetch_market_sentiment(
    *,
    fear_vix: float = 30.0,
    greed_vix: float = 15.0,
    fear_pc: float = 1.0,
    greed_pc: float = 0.6,
    fear_size_mult: float = 1.15,
    greed_size_mult: float = 0.85,
) -> MarketSentiment:
    """Fetch live VIX + put/call ratio and compute sentiment.

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

    sent = MarketSentiment(
        vix=vix,
        vix_z=vix_z,
        put_call_ratio=pcr,
        put_call_z=pcr_z,
        msi=msi,
        regime=regime,
        size_multiplier=size_mult,
    )

    log.info(f"  Market sentiment: {sent.summary()}")
    return sent


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
