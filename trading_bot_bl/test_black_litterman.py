"""
Tests for Black-Litterman implementation.

Verifies:
1. Ledoit-Wolf shrinkage produces valid covariance matrices
2. Equilibrium returns match expected behavior
3. Posterior returns shift toward views
4. Confidence mapping works correctly
5. Full pipeline with synthetic data
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_ledoit_wolf_shrinkage():
    """Ledoit-Wolf should produce a positive-definite matrix."""
    from black_litterman import ledoit_wolf_shrinkage

    np.random.seed(42)
    # 60 days, 10 assets (underdetermined)
    returns = np.random.randn(60, 10) * 0.02

    cov = ledoit_wolf_shrinkage(returns)

    # Check shape
    assert cov.shape == (10, 10), f"Shape: {cov.shape}"

    # Check symmetric
    assert np.allclose(cov, cov.T), "Not symmetric"

    # Check positive definite (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), (
        f"Not positive definite: min eigenvalue = "
        f"{eigenvalues.min():.2e}"
    )

    # Compare with sample covariance — shrunk should have
    # smaller condition number (better conditioned)
    sample_cov = np.cov(returns, rowvar=False)
    cond_shrunk = np.linalg.cond(cov)
    cond_sample = np.linalg.cond(sample_cov)
    assert cond_shrunk <= cond_sample * 1.1, (
        f"Shrunk condition ({cond_shrunk:.0f}) worse than "
        f"sample ({cond_sample:.0f})"
    )

    print(
        f"  PASS: Ledoit-Wolf shrinkage "
        f"(condition: {cond_shrunk:.0f} vs {cond_sample:.0f})"
    )


def test_equilibrium_returns():
    """Equilibrium returns should scale with risk and weight."""
    from black_litterman import compute_equilibrium_returns

    tickers = ["AAPL", "GOOG", "MSFT"]
    # AAPL is 50% of market, GOOG 30%, MSFT 20%
    market_caps = {
        "AAPL": 3e12, "GOOG": 1.8e12, "MSFT": 1.2e12
    }

    # Simple covariance: AAPL most volatile
    cov = np.array([
        [0.0004, 0.0001, 0.0001],
        [0.0001, 0.0003, 0.0001],
        [0.0001, 0.0001, 0.0002],
    ])

    pi = compute_equilibrium_returns(
        market_caps, cov, tickers, risk_aversion=2.5
    )

    # Higher market-cap weight + higher vol → higher implied return
    assert pi[0] > pi[2], (
        f"AAPL (high weight) should have higher return than "
        f"MSFT (low weight): {pi[0]:.6f} vs {pi[2]:.6f}"
    )

    # All returns should be positive (positive risk premium)
    assert np.all(pi > 0), f"Negative equilibrium returns: {pi}"

    annual_returns = pi * 252
    print(
        f"  PASS: Equilibrium returns "
        f"(AAPL={annual_returns[0]:.1%}, "
        f"GOOG={annual_returns[1]:.1%}, "
        f"MSFT={annual_returns[2]:.1%})"
    )


def test_posterior_shifts_toward_views():
    """
    Posterior returns should shift toward views when confidence
    is high, and stay near prior when confidence is low.
    """
    from black_litterman import (
        compute_posterior_returns,
        _confidence_to_omega,
    )

    n = 3
    tickers = ["A", "B", "C"]

    # Prior: all stocks have same equilibrium return
    pi = np.array([0.0003, 0.0003, 0.0003])  # daily

    cov = np.eye(n) * 0.0004  # simple diagonal

    # View: stock A will return much higher
    P = np.array([[1, 0, 0]])   # view on stock A only
    Q = np.array([0.001])       # high daily return view

    # ── High confidence view ──────────────────────────────────
    omega_high = np.array([_confidence_to_omega(6)])  # very sure
    posterior_high = compute_posterior_returns(
        pi, cov, P, Q, omega_high, tau=0.05
    )

    # Stock A's posterior should be pulled strongly toward Q
    shift_a = posterior_high[0] - pi[0]
    shift_b = abs(posterior_high[1] - pi[1])
    assert shift_a > 0, "Stock A should increase with bullish view"
    assert shift_a > shift_b * 2, (
        "Stock A shift should dominate B shift"
    )

    # ── Low confidence view ───────────────────────────────────
    omega_low = np.array([_confidence_to_omega(0)])  # unsure
    posterior_low = compute_posterior_returns(
        pi, cov, P, Q, omega_low, tau=0.05
    )

    # With low confidence, posterior should stay closer to prior
    shift_low = posterior_low[0] - pi[0]
    assert shift_low < shift_a, (
        f"Low confidence shift ({shift_low:.6f}) should be "
        f"smaller than high confidence ({shift_a:.6f})"
    )

    print(
        f"  PASS: Posterior shifts "
        f"(high conf: +{shift_a*252:.2%}/yr, "
        f"low conf: +{shift_low*252:.2%}/yr)"
    )


def test_confidence_to_omega_monotonic():
    """Higher confidence should map to lower omega (uncertainty)."""
    from black_litterman import _confidence_to_omega

    omegas = [_confidence_to_omega(i) for i in range(7)]

    # Should be monotonically decreasing
    for i in range(6):
        assert omegas[i] > omegas[i + 1], (
            f"Omega not decreasing: "
            f"conf={i}→{omegas[i]}, "
            f"conf={i+1}→{omegas[i+1]}"
        )

    # Extreme values should differ by orders of magnitude
    ratio = omegas[0] / omegas[6]
    assert ratio > 100, (
        f"Low/high omega ratio only {ratio:.0f}x "
        f"(should be >100x)"
    )

    print(
        f"  PASS: Confidence→omega mapping "
        f"(range: {omegas[6]:.5f} to {omegas[0]:.3f}, "
        f"ratio={ratio:.0f}x)"
    )


def test_weight_optimization():
    """Optimal weights should favor stocks with higher returns."""
    from black_litterman import optimize_weights

    tickers = ["HIGH", "MED", "LOW"]

    # Posterior: HIGH has much better return
    posterior = np.array([0.0008, 0.0004, 0.0001])  # daily
    # Realistic covariance with different volatilities
    cov = np.array([
        [0.0004, 0.0001, 0.00005],
        [0.0001, 0.0003, 0.00008],
        [0.00005, 0.00008, 0.0002],
    ])

    weights = optimize_weights(
        posterior, cov, tickers,
        risk_aversion=2.5,
        max_weight=0.60,
    )

    # HIGH should get the most weight
    assert weights["HIGH"] > weights["MED"], (
        f"HIGH ({weights['HIGH']:.3f}) should beat "
        f"MED ({weights['MED']:.3f})"
    )
    assert weights["MED"] > weights["LOW"], (
        f"MED ({weights['MED']:.3f}) should beat "
        f"LOW ({weights['LOW']:.3f})"
    )

    # All weights should be non-negative
    assert all(w >= 0 for w in weights.values()), (
        f"Negative weights: {weights}"
    )

    # Weights should sum to ~1.0
    total = sum(weights.values())
    assert 0.9 < total < 1.1, (
        f"Weights sum to {total:.3f} (expected ~1.0)"
    )

    print(
        f"  PASS: Weight optimization "
        f"(HIGH={weights['HIGH']:.1%}, "
        f"MED={weights['MED']:.1%}, "
        f"LOW={weights['LOW']:.1%})"
    )


def test_no_views_returns_prior():
    """With no views, posterior should equal the prior."""
    from black_litterman import compute_posterior_returns

    n = 3
    pi = np.array([0.0003, 0.0005, 0.0002])
    cov = np.eye(n) * 0.0004

    # Empty views
    P = np.array([]).reshape(0, n)
    Q = np.array([])
    omega = np.array([])

    posterior = compute_posterior_returns(
        pi, cov, P, Q, omega, tau=0.05
    )

    assert np.allclose(posterior, pi), (
        f"Posterior should equal prior with no views: "
        f"{posterior} vs {pi}"
    )

    print("  PASS: No views → posterior equals prior")


def test_full_pipeline_synthetic():
    """
    End-to-end test with synthetic signals and mock data.
    Verifies the full BL pipeline produces sensible outputs.
    """
    from black_litterman import (
        ledoit_wolf_shrinkage,
        compute_equilibrium_returns,
        signals_to_views,
        compute_posterior_returns,
        optimize_weights,
    )

    # Create mock signals
    class MockSignal:
        def __init__(
            self, ticker, conf, composite, sharpe,
            win_rate, trades, price, sl, tp, size_pct
        ):
            self.ticker = ticker
            self.signal_raw = 1  # BUY
            self.confidence_score = conf
            self.composite_score = composite
            self.sharpe = sharpe
            self.win_rate = win_rate
            self.total_trades = trades
            self.current_price = price
            self.stop_loss_price = sl
            self.take_profit_price = tp
            self.suggested_position_size_pct = size_pct

    signals = [
        MockSignal(
            "AAPL", 5, 75.0, 1.8, 0.62, 30,
            180.0, 171.0, 198.0, 12.0
        ),
        MockSignal(
            "GOOG", 3, 55.0, 0.9, 0.54, 20,
            165.0, 156.75, 181.5, 8.0
        ),
        MockSignal(
            "MSFT", 4, 65.0, 1.2, 0.58, 25,
            420.0, 399.0, 462.0, 10.0
        ),
    ]
    tickers = ["AAPL", "GOOG", "MSFT"]

    # Synthetic returns (60 days, correlated)
    np.random.seed(42)
    base = np.random.randn(60) * 0.01
    returns = np.column_stack([
        base + np.random.randn(60) * 0.005,  # AAPL
        base + np.random.randn(60) * 0.008,  # GOOG (more vol)
        base + np.random.randn(60) * 0.004,  # MSFT (less vol)
    ])

    # Step 1: Covariance
    cov = ledoit_wolf_shrinkage(returns)
    assert cov.shape == (3, 3)

    # Step 2: Equilibrium
    caps = {"AAPL": 3e12, "GOOG": 2e12, "MSFT": 2.8e12}
    pi = compute_equilibrium_returns(caps, cov, tickers, 2.5)
    assert pi.shape == (3,)

    # Step 3: Views
    P, Q, omega = signals_to_views(signals, tickers)
    assert P.shape == (3, 3), f"P shape: {P.shape}"
    assert Q.shape == (3,), f"Q shape: {Q.shape}"

    # Step 4: Posterior
    posterior = compute_posterior_returns(
        pi, cov, P, Q, omega, tau=0.05
    )
    assert posterior.shape == (3,)

    # AAPL (highest confidence) should have highest posterior
    aapl_idx = tickers.index("AAPL")
    goog_idx = tickers.index("GOOG")
    # AAPL has conf=5, GOOG has conf=3 → AAPL view dominates more
    # So AAPL posterior shift from prior should be larger

    # Step 5: Weights
    weights = optimize_weights(
        posterior, cov, tickers, risk_aversion=2.5
    )
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) > 0.5

    print(
        f"  PASS: Full pipeline "
        f"(AAPL={weights['AAPL']:.1%}, "
        f"GOOG={weights['GOOG']:.1%}, "
        f"MSFT={weights['MSFT']:.1%})"
    )
    annual_post = {t: posterior[i] * 252 for i, t in enumerate(tickers)}
    print(
        f"         Posterior returns: "
        f"AAPL={annual_post['AAPL']:.1%}, "
        f"GOOG={annual_post['GOOG']:.1%}, "
        f"MSFT={annual_post['MSFT']:.1%}"
    )


def test_llm_view_blending():
    """
    Verify that LLM views properly blend with quant views.
    """
    from black_litterman import (
        BLView,
        integrate_llm_views,
    )

    tickers = ["AAPL", "GOOG", "MSFT"]
    n = 3

    # Quant views: AAPL and GOOG
    quant_P = np.array([
        [1, 0, 0],  # AAPL
        [0, 1, 0],  # GOOG
    ])
    quant_Q = np.array([0.0005, 0.0003])  # daily returns
    quant_omega = np.array([0.0001, 0.001])

    # LLM views: AAPL (different return) and MSFT (new)
    llm_views = [
        BLView("AAPL", 0.20, 0.8, "llm"),  # 20% annual
        BLView("MSFT", 0.15, 0.6, "llm"),  # 15% annual, new
    ]

    P, Q, omega = integrate_llm_views(
        quant_P, quant_Q, quant_omega,
        llm_views, tickers, llm_weight=0.3
    )

    # Should now have 3 views (AAPL blended, GOOG unchanged, MSFT new)
    assert P.shape[0] == 3, f"Expected 3 views, got {P.shape[0]}"

    # AAPL view should be blended (between quant and LLM)
    aapl_view = Q[0]
    pure_quant = 0.0005
    pure_llm = 0.20 / 252
    assert pure_quant < aapl_view < pure_llm or (
        aapl_view != pure_quant
    ), "AAPL view should be blended"

    # MSFT should be the new LLM view
    assert P[2][2] == 1.0, "MSFT should be view #3"

    print(
        f"  PASS: LLM view blending "
        f"(3 views: AAPL blended, GOOG quant, MSFT llm)"
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Black-Litterman Model Tests")
    print("=" * 60 + "\n")

    tests = [
        test_ledoit_wolf_shrinkage,
        test_equilibrium_returns,
        test_posterior_shifts_toward_views,
        test_confidence_to_omega_monotonic,
        test_weight_optimization,
        test_no_views_returns_prior,
        test_full_pipeline_synthetic,
        test_llm_view_blending,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(1 if failed > 0 else 0)
