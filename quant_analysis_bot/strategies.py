"""Trading strategies -- each returns a signal Series: +1 buy, -1 sell, 0 hold."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _fear_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask indicating fear/stress regime.

    Used by mean-reversion strategies to gate buy signals.
    If the ``Regime_Fear`` column is present (from regime.py
    enrichment), use it.  Otherwise default to True everywhere
    so strategies behave as if unfiltered (backward compatible).
    """
    if "Regime_Fear" in df.columns:
        return df["Regime_Fear"].fillna(True).astype(bool)
    return pd.Series(True, index=df.index)


class Strategy:
    """Base class for all strategies."""

    name: str = "base"
    description: str = ""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


# ── Moving-average crossovers ─────────────────────────────────────────


class SMA_Crossover(Strategy):
    name = "SMA Crossover (10/50)"
    description = (
        "Buy when SMA10 crosses above SMA50, sell on cross below"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["SMA_10"] > df["SMA_50"]] = 1
        signals[df["SMA_10"] < df["SMA_50"]] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class EMA_Crossover(Strategy):
    name = "EMA Crossover (9/21)"
    description = (
        "Buy when EMA9 crosses above EMA21, sell on cross below"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["EMA_9"] > df["EMA_21"]] = 1
        signals[df["EMA_9"] < df["EMA_21"]] = -1
        return signals.diff().clip(-1, 1).fillna(0)


# ── Mean-reversion ────────────────────────────────────────────────────


class RSI_MeanReversion(Strategy):
    name = "RSI Mean Reversion"
    description = (
        "Buy when RSI<30 (oversold) during fear regime, "
        "sell when RSI>70 (overbought)"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        fear = _fear_mask(df)
        signals[fear & (df["RSI_14"] < 30)] = 1
        signals[df["RSI_14"] > 70] = -1
        return signals


class BollingerBand_Reversion(Strategy):
    name = "Bollinger Band Mean Reversion"
    description = (
        "Buy at lower band during fear regime, sell at upper band"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        fear = _fear_mask(df)
        signals[fear & (df["Close"] < df["BB_Lower"])] = 1
        signals[df["Close"] > df["BB_Upper"]] = -1
        return signals


class ZScore_MeanReversion(Strategy):
    name = "Z-Score Mean Reversion"
    description = (
        "Buy when price >1.5 std below mean during fear regime, "
        "sell >1.5 std above"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        fear = _fear_mask(df)
        signals[fear & (df["ZScore_20"] < -1.5)] = 1
        signals[df["ZScore_20"] > 1.5] = -1
        return signals


# ── Momentum / trend ──────────────────────────────────────────────────


class MACD_Strategy(Strategy):
    name = "MACD Crossover"
    description = (
        "Buy on MACD bullish crossover, sell on bearish crossover"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["MACD_Hist"] > 0] = 1
        signals[df["MACD_Hist"] < 0] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class Momentum_ROC(Strategy):
    name = "Momentum (Rate of Change)"
    description = (
        "Buy on strong positive momentum, sell on strong negative"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        threshold = df["ROC_10"].rolling(50).std()
        signals[df["ROC_10"] > threshold] = 1
        signals[df["ROC_10"] < -threshold] = -1
        return signals


class Stochastic_Strategy(Strategy):
    name = "Stochastic Oscillator"
    description = (
        "Buy when %K crosses above %D in oversold zone, "
        "sell in overbought"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        buy = (df["Stoch_K"] > df["Stoch_D"]) & (df["Stoch_K"] < 25)
        sell = (df["Stoch_K"] < df["Stoch_D"]) & (df["Stoch_K"] > 75)
        signals[buy] = 1
        signals[sell] = -1
        return signals


class VWAP_Strategy(Strategy):
    name = "VWAP Trend"
    description = (
        "Buy when price crosses above VWAP with volume, sell below"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        vol_confirm = df["Vol_Ratio"] > 1.2
        signals[(df["Close"] > df["VWAP"]) & vol_confirm] = 1
        signals[(df["Close"] < df["VWAP"]) & vol_confirm] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class TrendFollowing_ADX(Strategy):
    name = "ADX Trend Following"
    description = (
        "Follow trend when ADX>25 (strong trend), use MA direction"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        strong_trend = df["ADX_14"] > 25
        signals[strong_trend & (df["EMA_9"] > df["SMA_50"])] = 1
        signals[strong_trend & (df["EMA_9"] < df["SMA_50"])] = -1
        return signals.diff().clip(-1, 1).fillna(0)


# ── Composite ─────────────────────────────────────────────────────────


class CompositeScore(Strategy):
    name = "Composite Multi-Indicator"
    description = (
        "Weighted score across RSI, MACD, BB, and trend indicators"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)

        # RSI component
        score += (
            np.where(
                df["RSI_14"] < 35,
                1,
                np.where(df["RSI_14"] > 65, -1, 0),
            )
            * 0.2
        )
        # MACD component
        score += (
            np.where(
                df["MACD_Hist"] > 0,
                1,
                np.where(df["MACD_Hist"] < 0, -1, 0),
            )
            * 0.2
        )
        # Bollinger component
        bb_pos = (df["Close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"]
        )
        score += (
            np.where(bb_pos < 0.2, 1, np.where(bb_pos > 0.8, -1, 0))
            * 0.2
        )
        # Trend component (EMA)
        score += np.where(df["EMA_9"] > df["EMA_21"], 1, -1) * 0.2
        # Volume confirmation
        score += np.where(df["Vol_Ratio"] > 1.3, 0.2, 0)

        signals = pd.Series(0, index=df.index)
        signals[score > 0.4] = 1
        signals[score < -0.4] = -1
        return signals


# ── Breakout ─────────────────────────────────────────────────────────


class DonchianBreakout(Strategy):
    name = "Donchian Breakout (20/55)"
    description = (
        "Buy when price breaks above 20d high with ADX>20 "
        "and above 200-SMA; sell on break below 55d low"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Use *previous* day's channel to avoid look-ahead bias
        upper_20 = df["Donchian_Upper_20"].shift(1)
        lower_55 = df["Donchian_Lower_55"].shift(1)

        # Filters: trend must exist (ADX>20) and price above
        # 200-SMA (uptrend filter prevents counter-trend entries)
        trend_filter = (df["ADX_14"] > 20) & (
            df["Close"] > df["SMA_200"]
        )

        # Build a position state: +1 when in long, -1 when
        # breakout conditions lost, 0 otherwise.
        # Entry: close breaks above yesterday's 20d high
        # with trend confirmation.
        long_entry = trend_filter & (df["Close"] > upper_20)

        # Exit: close breaks below yesterday's 55d low
        # (wider exit channel lets winners run — asymmetric
        # entry/exit is the classic turtle trader insight)
        long_exit = df["Close"] < lower_55

        # Build state vector: only emit crossover signals.
        # A buy entry that doesn't pass filters stays at 0.
        state = pd.Series(0, index=df.index)
        state[long_entry] = 1
        state[long_exit] = -1

        # Forward-fill state so we're "in" or "out",
        # then diff to get transition signals only.
        state = state.replace(0, np.nan).ffill().fillna(0)
        return state.diff().clip(-1, 1).fillna(0)


# ── 52-Week High Momentum ────────────────────────────────────────────


class FiftyTwoWeekHighMomentum(Strategy):
    name = "52-Week High Momentum"
    description = (
        "Buy when price is within 5% of 52-week high with "
        "trend confirmation (above 200-SMA); exit when price "
        "drops >10% below the high"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on nearness to 52-week high.

        The 52-week high momentum anomaly: stocks near their
        52-week high tend to continue outperforming because
        traders anchor to the high and under-react to positive
        information that pushes the stock toward it.

        Requires columns:
          - Nearness_52w_High: Close / 52-week High (0-1 ratio)
          - SMA_200: 200-day simple moving average
          - ADX_14: average directional index

        If columns are missing, returns all-zero signals.
        """
        signals = pd.Series(0, index=df.index)

        # Graceful degradation
        required = ["Nearness_52w_High", "SMA_200", "ADX_14"]
        if not all(col in df.columns for col in required):
            return signals

        nearness = df["Nearness_52w_High"]

        # ── Entry conditions ────────────────────────────────────
        # Price within 5% of 52-week high (nearness > 0.95)
        near_high = nearness > 0.95

        # Trend confirmation: above 200-SMA (uptrend)
        above_trend = df["Close"] > df["SMA_200"]

        # Trend strength: ADX > 20 (meaningful trend exists)
        trending = df["ADX_14"] > 20

        # BUY: near 52-week high, in uptrend, with trend strength
        buy = near_high & above_trend & trending
        signals[buy] = 1

        # ── Exit conditions ─────────────────────────────────────
        # Price dropped >10% from 52-week high (momentum faded)
        faded = nearness < 0.90

        # Or price fell below 200-SMA (trend broken)
        below_trend = df["Close"] < df["SMA_200"]

        exit_signal = faded | below_trend
        signals[exit_signal] = -1

        return signals


# ── Event-driven ─────────────────────────────────────────────────────


class PEAD_Drift(Strategy):
    name = "PEAD Earnings Drift"
    description = (
        "Buy after positive earnings surprise with confirming "
        "price gap; ride drift for up to 60 days post-earnings"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on post-earnings announcement drift.

        Requires PEAD columns to be present in the DataFrame:
          - PEAD_Surprise_Pct: standardised earnings surprise (%)
          - PEAD_Days_Since: trading days since last earnings
          - PEAD_Gap_Pct: price gap on earnings day (%)

        If columns are missing (no earnings data available),
        returns all-zero signals (HOLD).
        """
        signals = pd.Series(0, index=df.index)

        # Graceful degradation: no PEAD columns → no signals
        required = [
            "PEAD_Surprise_Pct",
            "PEAD_Days_Since",
            "PEAD_Gap_Pct",
        ]
        if not all(col in df.columns for col in required):
            return signals

        surprise = df["PEAD_Surprise_Pct"]
        days_since = df["PEAD_Days_Since"]
        gap_pct = df["PEAD_Gap_Pct"]

        # ── Entry conditions ────────────────────────────────────
        # Positive surprise: actual beat estimate by > 5%
        positive_surprise = surprise > 5.0

        # Confirming price reaction: stock gapped up > 1%
        # on earnings day (market agrees with surprise)
        confirmed_reaction = gap_pct > 1.0

        # Within drift window: 2-60 trading days post-earnings
        # Start at day 2 to avoid the immediate post-earnings
        # volatility (and respect the 1-day blackout post-period)
        in_drift_window = (days_since >= 2) & (days_since <= 60)

        # Trend filter: only ride drift in uptrend (above 200-SMA)
        above_trend = df["Close"] > df["SMA_200"]

        # BUY signal: all conditions met
        buy = (
            positive_surprise
            & confirmed_reaction
            & in_drift_window
            & above_trend
        )
        signals[buy] = 1

        # EXIT signal: drift window expired (>60 days)
        # or negative surprise with confirming gap down
        negative_surprise = surprise < -5.0
        negative_gap = gap_pct < -1.0
        exit_drift = (days_since > 60) | (
            negative_surprise & negative_gap & in_drift_window
        )
        signals[exit_drift] = -1

        return signals


# ── Registry ──────────────────────────────────────────────────────────

ALL_STRATEGIES: List[Strategy] = [
    SMA_Crossover(),
    EMA_Crossover(),
    RSI_MeanReversion(),
    MACD_Strategy(),
    BollingerBand_Reversion(),
    Momentum_ROC(),
    ZScore_MeanReversion(),
    Stochastic_Strategy(),
    VWAP_Strategy(),
    TrendFollowing_ADX(),
    CompositeScore(),
    DonchianBreakout(),
    FiftyTwoWeekHighMomentum(),
    PEAD_Drift(),
]
