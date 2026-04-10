"""Shared utilities used by both trading_bot_bl and quant_analysis_bot.

This package exists to avoid cross-package coupling — neither package
imports from the other; both import from shared/ for common math
and calendar helpers.
"""
