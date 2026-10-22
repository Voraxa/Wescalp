import pytest

from wescalp.flow import (
    Quote, Tick, aggregate, bin_means, correlation, depth_imbalance, forward_returns, micro_dev_bps, microprice,
    ofi, report, spread_bps, trade_delta,
)

T0 = 1_792_699_800.0  # múltiplo de 10 s


def q(ts: float, bid: float = 100.0, bsz: float = 10, ask: float = 100.1, asz: float = 10,
      b5: float = 50, a5: float = 50) -> Quote:
    return Quote(T0 + ts, bid, bsz, ask, asz, b5, a5)


def series(seconds: float, step: float = 0.1, **kw) -> list[Quote]:
    """Fotos a cada `step` segundos em [0, seconds), como o level2Depth50 manda."""
    return [q(i * step, **kw) for i in range(round(seconds / step))]


def test_ofi_size_change_only():
    assert ofi(q(0, bsz=10, asz=10), q(0.1, bsz=15, asz=10)) == 5  # fila do bid cresce: compra
    assert ofi(q(0, bsz=10, asz=10), q(0.1, bsz=10, asz=13)) == -3  # fila do ask cresce: venda
    assert ofi(q(0), q(0.1)) == 0


def test_ofi_bid_up_counts_whole_new_queue():
    # Bid sobe de 100 para 100.05: conta os 7 contratos novos inteiros; ask parado e igual não contribui.
    assert ofi(q(0, bid=100, bsz=10), q(0.1, bid=100.05, bsz=7)) == 7


def test_ofi_bid_down_removes_whole_old_queue():
    assert ofi(q(0, bid=100, bsz=10), q(0.1, bid=99.9, bsz=30)) == -10


def test_ofi_ask_moves_mirror_bid():
    assert ofi(q(0, ask=100.1, asz=10), q(0.1, ask=100.05, asz=4)) == -4  # ask desce: venda
    assert ofi(q(0, ask=100.1, asz=10), q(0.1, ask=100.2, asz=40)) == 10  # ask sobe: compra


def test_book_state_signals():
    assert depth_imbalance(30, 10) == pytest.approx(0.5)
    assert depth_imbalance(0, 0) is None
    book = q(0, bid=100, bsz=30, ask=100.2, asz=10)
    assert microprice(book) == pytest.approx(100.15)  # fila do ask menor puxa para o ask
    assert micro_dev_bps(book) == pytest.approx(0.05 / 100.1 * 1e4)
    assert spread_bps(book) == pytest.approx(0.2 / 100.1 * 1e4)


def test_trade_delta():
    assert trade_delta([Tick(0, "buy", 5), Tick(0, "sell", 2), Tick(0, "buy", 1)]) == 4
    assert trade_delta([]) == 0


def test_buckets_sum_flow_and_take_state_at_end():
    quotes = series(3)
    quotes[15] = q(1.5, bsz=12)  # +2 em t=1.5 e −2 de volta em t=1.6
    quotes[25] = q(2.5, bsz=13, b5=70, a5=30)
    ticks = [Tick(T0 + 1.2, "buy", 3), Tick(T0 + 1.7, "sell", 1), Tick(T0 + 2.0, "sell", 4)]
    c = aggregate(quotes, ticks, width=1)
    assert c["t"] == [T0 + 1, T0 + 2, T0 + 3]
    assert c["mid"][0] is None and c["mid"][2] is None  # primeiro e último balde ficam de fora
    assert c["ofi"][1] == 0  # sobe 2 e volta dentro do balde
    assert c["delta"][1] == 2  # o negócio em t=2.0 é do balde seguinte
    assert c["imb5"][1] == 0 and c["mid"][1] == pytest.approx(100.05)


def test_empty_bucket_without_gap_carries_state():
    # Fotos a cada 0,5 s e baldes de 0,2 s: alguns baldes não têm mensagem, mas não há buraco.
    c = aggregate(series(3, step=0.5, bsz=20, asz=10), [], width=0.2)
    empty = [k for k in range(1, len(c["t"]) - 1) if round((c["t"][k] - T0) / 0.2) % 5 not in (1, 3)]
    assert len(empty) == 7 and all(c["ofi"][k] == 0 and c["mid"][k] is not None for k in empty)


def test_gap_invalidates_touching_buckets_and_skips_ofi():
    quotes = series(5) + [q(8.0 + i * 0.1, bsz=99) for i in range(30)]  # buraco de 4,9 s até t=8
    c = aggregate(quotes, [Tick(T0 + 6, "buy", 7)], width=1, max_gap=2)
    k = {round(t - T0) - 1: i for i, t in enumerate(c["t"])}  # balde [s, s+1) -> índice
    assert all(c["mid"][k[s]] is None for s in (4, 5, 6, 7, 8))
    assert c["delta"][k[6]] is None  # negócio no buraco não vira sinal
    assert c["mid"][k[3]] is not None and c["mid"][k[9]] is not None
    assert c["ofi"][k[9]] == 0  # o salto de tamanho atravessou o buraco e não entra no OFI


def test_crossed_book_is_ignored():
    quotes = series(3)
    quotes[12] = q(1.2, bid=100.2, ask=100.1)
    assert aggregate(quotes, [], width=1)["spread_bps"][1] == pytest.approx(0.1 / 100.05 * 1e4)


def test_forward_returns_align_with_future_bucket():
    assert forward_returns([100, 101, None, 103], 1) == [pytest.approx(0.01), None, None, None]
    assert forward_returns([100, 101, None, 103], 3)[0] == pytest.approx(0.03)


def test_no_lookahead():
    base = series(20)
    c = aggregate(base, [], width=1)
    # Mexer no que vem a partir de t=10 (inclusive: ts == fim do balde é do balde seguinte) não muda o sinal
    # nem o mid até o balde [9, 10).
    changed = base[:100] + [q(10 + i * 0.1, bid=105, ask=105.1, bsz=1, b5=1) for i in range(100)]
    c2 = aggregate(changed, [Tick(T0 + 10.0, "buy", 50)], width=1)
    for name in ("mid", "ofi", "delta", "imb5", "micro_bps"):
        assert c2[name][:10] == c[name][:10]
    # O retorno de 5 s do balde que fecha em t=10 vai até o mid em t=15, que já mudou.
    r = forward_returns(c2["mid"], 5)
    assert r[9] == pytest.approx(105.05 / 100.05 - 1)
    assert r[3] == 0  # 4 -> 9: tudo antes da mudança


def test_bin_means_keeps_ties_together():
    signal = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    rets = [0.0] * 6 + [0.1, 0.2, 0.3, 0.4]
    means = bin_means(signal, rets, 5)
    assert means[1] == (6, 0.0)  # os seis zeros juntos, na faixa do posto médio
    assert means[0] is None
    assert means[-1] == (2, pytest.approx(0.35))
    assert sum(m[0] for m in means if m) == 10


def test_correlation_handles_constant_signal():
    assert correlation([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]) is None
    assert correlation([1, 2, 3, None], [0.1, 0.2, 0.3, 0.9]) == pytest.approx(1)


def test_report_finds_planted_signal():
    # O OFI do balde decide o próximo passo do mid: +10 bps se positivo, −10 bps se negativo.
    signs = [1 if (i * 7) % 3 else -1 for i in range(100)]
    mids = [100.0]
    for s in signs[:-1]:
        mids.append(mids[-1] * (1 + s * 0.001))
    cols = {"t": list(range(100)), "mid": mids, "ofi": [s * 5.0 for s in signs], "delta": [0.0] * 100,
            "imb5": [None] * 100, "micro_bps": [0.0] * 100, "spread_bps": [0.0] * 100}
    rows = report(cols, 10, [10, 15], bins=2)
    assert [r["h"] for r in rows if r["signal"] == "ofi"] == [10]  # 15 s não é múltiplo do balde
    r = next(r for r in rows if r["signal"] == "ofi")
    assert r["n"] == 99 and r["n_ef"] == 99
    assert r["corr"] == pytest.approx(1) and r["extreme_bps"] == pytest.approx(10)
    assert r["bins_bps"] == [pytest.approx(-10), pytest.approx(10)]
    assert next(r for r in rows if r["signal"] == "imb5")["n"] == 0
