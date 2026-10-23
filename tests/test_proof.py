import pytest

pytest.importorskip("numpy")

from wescalp import bitstamp, brain, proof  # noqa: E402


def test_criteria_need_all_four():
    hold = {"sharpe": 0.8, "dd": 0.8}
    assert proof.criteria({"total": 1.0, "sharpe": 1.0, "dd": 0.5, "p": 0.01}, hold) == []
    assert proof.criteria({"total": -0.1, "sharpe": 0.8, "dd": 0.8, "p": 0.2}, hold) == ["lucro", "Sharpe", "queda",
                                                                                            "sorte"]


def test_refuses_data_that_changed(tmp_path):
    p = tmp_path / "x.csv.gz"
    brain.save_candles(p, bitstamp.parse({"data": {"ohlc": [
        {"timestamp": "1356998400", "open": "1", "high": "1", "low": "1", "close": "1", "volume": "0"}]}}))
    with pytest.raises(SystemExit):
        proof.check_digest(p)
