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


# ── Multi-factor momentum + mean reversion ──────────────────────────


class MultiFactorMomentumMR(Strategy):
    """Composite momentum strategy with mean-reversion entry timing.

    Combines up to five z-scored factors based on academic anomalies:
      - 6-month momentum (skip last month) — Jegadeesh & Titman
      - 1-month momentum — short-term confirmation
      - RSI mean reversion (inverted) — buy dips in uptrends
      - Inverse volatility — Barroso & Santa-Clara (2015)
      - Volume surge — institutional participation

    On short scoring windows (3mo/6mo) where 6-month momentum is
    unavailable, the strategy adaptively redistributes its weight
    across the remaining factors so it can still participate in
    selection rather than returning all-zero.

    Signals are transition-based (``+1`` on entry, ``-1`` on exit,
    ``0`` otherwise) to avoid the backtester charging repeated
    transaction costs on level signals while flat.

    Filters: above 200-SMA, RSI < 70, positive momentum.
    """

    name = "Multi-Factor Momentum MR"
    description = (
        "Buy momentum stocks on pullbacks using z-scored "
        "factors (momentum, inverted RSI, inverse vol, "
        "volume surge) with trend and overbought filters"
    )

    # Factor weights (full 5-factor mode)
    _FACTOR_DEFS = {
        # key: (base_weight, min_bars_needed)
        "mom_6m":    (0.35, 126 + 21),  # MOM_6M + SKIP
        "mom_1m":    (0.25, 21),
        "rsi_inv":   (0.15, 14),
        "inv_vol":   (0.15, 20),
        "vol_surge": (0.10, 20),
    }

    # Lookback periods (trading days)
    MOM_6M_DAYS = 126      # ~6 months
    MOM_SKIP_DAYS = 21     # skip most recent month
    MOM_1M_DAYS = 21       # ~1 month
    ZSCORE_WINDOW = 63     # rolling z-score lookback (~3 months)

    # Filter thresholds
    RSI_OVERBOUGHT = 70
    SCORE_BUY_THRESHOLD = 0.5
    SCORE_EXIT_THRESHOLD = -0.3

    # Minimum bars: need at least 1-month momentum + z-score min
    # periods to produce any meaningful signal at all.
    _ABSOLUTE_MIN_BARS = 21 + 20  # MOM_1M + z-score min_periods

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        # Graceful degradation: check required columns
        required = [
            "Close", "SMA_200", "RSI_14", "Volatility_20",
            "Vol_Ratio",
        ]
        if not all(col in df.columns for col in required):
            return signals

        c = df["Close"]
        n = len(df)

        # Need enough bars for at least the short-window factors
        if n < self._ABSOLUTE_MIN_BARS:
            return signals

        # ── Compute raw factors ─────────────────────────────
        # 6-month momentum (skip last month) — may be NaN on
        # short windows, which is fine; we adapt weights below.
        mom_6m = (
            c.shift(self.MOM_SKIP_DAYS)
            / c.shift(self.MOM_6M_DAYS + self.MOM_SKIP_DAYS)
            - 1
        )

        mom_1m = c / c.shift(self.MOM_1M_DAYS) - 1
        rsi_inv = 100 - df["RSI_14"]

        vol = df["Volatility_20"].replace(0, np.nan)
        inv_vol = 1.0 / vol

        vol_surge = df["Vol_Ratio"]

        # ── Z-score each factor over rolling window ─────────
        def _rolling_zscore(series: pd.Series) -> pd.Series:
            mu = series.rolling(
                self.ZSCORE_WINDOW, min_periods=20,
            ).mean()
            sigma = series.rolling(
                self.ZSCORE_WINDOW, min_periods=20,
            ).std()
            z = (series - mu) / sigma.replace(0, np.nan)
            return z.clip(-3, 3).fillna(0)

        raw_factors = {
            "mom_6m":    mom_6m,
            "mom_1m":    mom_1m,
            "rsi_inv":   rsi_inv,
            "inv_vol":   inv_vol,
            "vol_surge": vol_surge,
        }
        z_factors = {k: _rolling_zscore(v) for k, v in raw_factors.items()}

        # ── Adaptive weighting ──────────────────────────────
        # Determine which factors have enough history to be
        # meaningful.  If 6-month momentum is unavailable (short
        # window), redistribute its weight proportionally across
        # the remaining factors.
        available = {
            k for k, (_, min_n) in self._FACTOR_DEFS.items()
            if n >= min_n + 20  # +20 for z-score min_periods
        }

        if not available:
            return signals

        raw_weights = {
            k: w for k, (w, _) in self._FACTOR_DEFS.items()
            if k in available
        }
        total_w = sum(raw_weights.values())
        weights = {k: w / total_w for k, w in raw_weights.items()}

        # ── Weighted composite score ────────────────────────
        score = pd.Series(0.0, index=df.index)
        for k, w in weights.items():
            score += w * z_factors[k]

        # ── Warm-up mask ────────────────────────────────────
        # SMA_200 is NaN until bar 200 (min_periods=200 in
        # indicators.py).  Before SMA_200 is valid, the trend
        # filter can't be evaluated, so we must NOT emit any
        # signal — otherwise ~above_200sma produces spurious
        # -1 exits across the whole warm-up region, and the
        # backtester charges exit cost on each bar while flat.
        sma_valid = df["SMA_200"].notna()
        tradeable = sma_valid

        # ── Filters ─────────────────────────────────────────
        above_200sma = c > df["SMA_200"]
        not_overbought = df["RSI_14"] < self.RSI_OVERBOUGHT

        # Use 6-month momentum filter when available, fall back
        # to 1-month momentum on short windows.
        if "mom_6m" in available:
            positive_momentum = mom_6m > 0
        else:
            positive_momentum = mom_1m > 0

        # All filters must pass for a BUY, and warm-up must be done
        buy_eligible = (
            tradeable & above_200sma & not_overbought & positive_momentum
        )

        # ── Desired-position state machine ─────────────────
        # Track the desired position at every bar:
        #   +1 = want to be long
        #   -1 = want to be flat
        #
        # Entry and exit conditions set explicit values; bars
        # with no new decision are NaN and forward-filled from
        # the last explicit decision.  This avoids two bugs:
        #
        #  a) Sparse-level false buys: a naive level vector of
        #     [-1, 0, 0, ...] diffs to [+1, 0, ...], creating
        #     phantom entries when "exit stopped firing" rather
        #     than a genuine buy condition.
        #
        #  b) Repeated-exit cost leak: level-based -1 on every
        #     bar below 200-SMA causes the backtester to charge
        #     exit cost each bar while already flat.
        #
        # By forward-filling, consecutive identical states diff
        # to 0, and only genuine state changes produce ±1.

        desired = pd.Series(np.nan, index=df.index)

        # Entry: we want to be long
        desired[buy_eligible & (score > self.SCORE_BUY_THRESHOLD)] = 1

        # Exit: score collapses or price drops below 200-SMA.
        # Only in the tradeable region.
        exit_cond = tradeable & (
            (score < self.SCORE_EXIT_THRESHOLD)
            | (~above_200sma)
        )
        desired[exit_cond] = -1

        # Non-tradeable bars (SMA warm-up): force flat so no
        # signal leaks out of the warm-up region, and so the
        # first tradeable exit bar doesn't diff against NaN/0.
        desired[~tradeable] = -1

        # Forward-fill: bars with no new decision hold the last
        # explicit state.  Initial NaN gap (if any) fills to -1.
        desired = desired.ffill().fillna(-1)

        # Convert desired position to transition signals:
        # +1 on the bar we enter, -1 on the bar we exit, 0 hold.
        signals = desired.diff().clip(-1, 1).fillna(0).astype(int)

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
    MultiFactorMomentumMR(),
]
