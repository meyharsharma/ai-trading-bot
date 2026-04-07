"""Tests for the Kraken MCP tool wrappers and JSON normalizers."""
import pytest

from agent.kraken_mcp.client import _extract_payload
from agent.kraken_mcp.tools import (
    KrakenMCPError,
    OHLCVBar,
    _from_kraken_pair,
    _kraken_pair,
    _normalize_ohlcv,
    _normalize_ticker,
    _parse_ts,
)


# ---------------- pair translation ----------------

def test_kraken_pair_btc_to_xbt():
    assert _kraken_pair("BTC/USD") == "XBTUSD"


def test_kraken_pair_eth_pass_through():
    assert _kraken_pair("ETH/USD") == "ETHUSD"


def test_from_kraken_pair_recognizes_kraken_codes():
    assert _from_kraken_pair("XXBTZUSD") == "BTC/USD"
    assert _from_kraken_pair("XBTUSD") == "BTC/USD"
    assert _from_kraken_pair("ETHUSD") == "ETH/USD"
    assert _from_kraken_pair("BTC/USD") == "BTC/USD"
    assert _from_kraken_pair("DOGEUSD") is None


# ---------------- ticker normalization ----------------

def test_normalize_ticker_kraken_rest_shape():
    raw = {
        "result": {
            "XXBTZUSD": {
                "a": ["65000.1", "1", "1.000"],
                "b": ["64999.9", "1", "1.000"],
                "c": ["65000.0", "0.001"],
            }
        },
        "error": [],
    }
    t = _normalize_ticker(raw)
    assert t["bid"] == 64999.9
    assert t["ask"] == 65000.1
    assert t["last"] == 65000.0
    assert t["mid"] == pytest.approx(65000.0)


def test_normalize_ticker_flat_shape():
    raw = {"bid": 100.0, "ask": 102.0, "last": 101.0}
    t = _normalize_ticker(raw)
    assert t == {"bid": 100.0, "ask": 102.0, "last": 101.0, "mid": 101.0}


def test_normalize_ticker_falls_back_to_last_when_no_book():
    raw = {"last": 50.0}
    t = _normalize_ticker(raw)
    assert t["mid"] == 50.0


def test_normalize_ticker_rejects_non_dict():
    with pytest.raises(KrakenMCPError):
        _normalize_ticker("not-a-dict")


# ---------------- ohlcv normalization ----------------

def test_normalize_ohlcv_kraken_rest_list_shape():
    raw = {
        "result": {
            "XXBTZUSD": [
                [1700000000, "65000", "65100", "64900", "65050", "65010", "1.5", 10],
                [1700000300, "65050", "65200", "65000", "65150", "65100", "2.0", 12],
            ],
            "last": 1700000300,
        },
        "error": [],
    }
    bars = _normalize_ohlcv(raw)
    assert len(bars) == 2
    assert bars[0].open == 65000
    assert bars[0].close == 65050
    assert bars[1].close == 65150
    assert bars[1].volume == 2.0


def test_normalize_ohlcv_dict_rows():
    raw = {
        "candles": [
            {"time": 1700000000, "o": 100, "h": 110, "l": 90, "c": 105, "v": 5},
        ]
    }
    bars = _normalize_ohlcv(raw)
    assert bars[0].open == 100
    assert bars[0].close == 105
    assert bars[0].volume == 5


def test_normalize_ohlcv_empty_when_no_match():
    assert _normalize_ohlcv({}) == []
    assert _normalize_ohlcv("garbage") == []


def test_parse_ts_handles_iso_string():
    ts = _parse_ts("2025-01-01T00:00:00Z")
    assert ts.year == 2025
    assert ts.tzinfo is not None


def test_parse_ts_handles_unix_seconds_and_ms():
    sec = _parse_ts(1700000000)
    ms = _parse_ts(1700000000 * 1000)
    assert sec == ms


# ---------------- _extract_payload helper ----------------

class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, blocks, error: bool = False) -> None:
        self.content = blocks
        self.isError = error


def test_extract_payload_parses_json():
    assert _extract_payload(_Result([_Block('{"a": 1}')])) == {"a": 1}


def test_extract_payload_returns_text_on_non_json():
    assert _extract_payload(_Result([_Block("hello")])) == "hello"


def test_extract_payload_handles_no_content():
    assert _extract_payload(_Result([])) is None
