"""Trading strategies -- each returns a signal Series: +1 buy, -1 sell, 0 hold."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


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
        "Buy when RSI<30 (oversold), sell when RSI>70 (overbought)"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["RSI_14"] < 30] = 1
        signals[df["RSI_14"] > 70] = -1
        return signals


class BollingerBand_Reversion(Strategy):
    name = "Bollinger Band Mean Reversion"
    description = "Buy at lower band, sell at upper band"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["Close"] < df["BB_Lower"]] = 1
        signals[df["Close"] > df["BB_Upper"]] = -1
        return signals


class ZScore_MeanReversion(Strategy):
    name = "Z-Score Mean Reversion"
    description = (
        "Buy when price >1.5 std below mean, sell >1.5 std above"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["ZScore_20"] < -1.5] = 1
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
]
