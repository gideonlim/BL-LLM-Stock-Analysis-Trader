"""Day-trader strategies.

Each strategy is a subclass of :class:`DayTradeStrategy` that emits
``DayTradeSignal`` objects from intraday market state. The signals
flow into the filter pipeline → risk manager → broker.

v1: ``orb_vwap`` (Opening Range Breakout + VWAP + RVOL),
    ``vwap_pullback`` (VWAP pullback continuation).
v2: ``vwap_reversion`` (no-news mean reversion to session VWAP).
v2.5: ``catalyst_momentum`` — needs a real-time news vendor.
v3: ``pairs_stat_arb`` — stretch.
"""
