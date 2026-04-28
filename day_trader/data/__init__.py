"""Data layer for the day-trader.

The data layer's job: turn Alpaca's WebSocket stream into the typed
``Bar`` / ``Quote`` / ``Trade`` events strategies consume, and
maintain the rolling per-ticker state (BarCache) needed for VWAP /
ATR / volume math.

Currently shipped: BarCache (the in-memory rolling buffer).
Coming next: the live feed adapter, premarket scanner, catalyst
classifier, universe builder.
"""
