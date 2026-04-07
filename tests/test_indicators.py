"""Pure-math tests for the indicator helpers."""
from agent.data.indicators import atr, compute_indicators, rsi, sma


def test_sma_simple_average():
    assert sma([1, 2, 3, 4, 5], 5) == 3.0
    assert sma([1, 2, 3, 4, 5], 3) == 4.0


def test_sma_returns_zero_when_insufficient_data():
    assert sma([1, 2], 5) == 0.0
    assert sma([], 3) == 0.0
    assert sma([1, 2, 3], 0) == 0.0


def test_rsi_all_gains_pegs_at_100():
    closes = list(range(1, 30))  # strictly increasing
    assert rsi(closes, 14) == 100.0


def test_rsi_all_losses_pegs_at_zero():
    closes = list(range(30, 1, -1))  # strictly decreasing
    assert rsi(closes, 14) == 0.0


def test_rsi_returns_zero_with_insufficient_data():
    assert rsi([1.0, 2.0], 14) == 0.0


def test_atr_returns_positive_for_volatile_series():
    highs = [10 + i for i in range(20)]
    lows = [9 + i for i in range(20)]
    closes = [9.5 + i for i in range(20)]
    assert atr(highs, lows, closes, period=14) > 0


def test_atr_returns_zero_with_insufficient_data():
    assert atr([1.0], [0.5], [0.75], period=14) == 0.0


def test_compute_indicators_packs_full_snapshot():
    closes = [float(i) for i in range(1, 101)]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    snap = compute_indicators(closes, highs, lows)
    assert snap.ma_fast > 0
    assert snap.ma_slow > 0
    assert snap.ma_fast > snap.ma_slow  # rising series → fast > slow
    keys = snap.as_dict().keys()
    assert set(keys) == {"ma_fast", "ma_slow", "rsi", "atr"}
