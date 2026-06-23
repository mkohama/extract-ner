"""``src.llm.windows.iter_windows`` のテスト（spaCy/pii-masker/Azure 非依存）。

窓は J1 本文 ``text`` の連続スパンで、CHUNK_SEPARATOR 境界でのみ切れること、
``text[ws:we]`` が窓本文になること、トークン予算で分割されることを固定する。
"""

from __future__ import annotations

from src.llm.windows import iter_windows
from src.ner.preprocess import CHUNK_SEPARATOR


def test_empty_text_yields_no_windows() -> None:
    assert iter_windows("") == []
    assert iter_windows(CHUNK_SEPARATOR) == []  # 空白のみ→セグメントなし


def test_small_text_is_single_window() -> None:
    text = "田中さんは横浜にいます"
    wins = iter_windows(text, max_tokens=7000, overlap=0)
    assert wins == [(0, len(text))]
    ws, we = wins[0]
    assert text[ws:we] == text


def test_windows_are_substrings_of_text() -> None:
    segs = ["第一章の本文", "第二章の本文", "第三章の本文"]
    text = CHUNK_SEPARATOR.join(segs)
    # 各セグメントが別窓になるよう極端に小さい予算で切る
    wins = iter_windows(text, max_tokens=1, overlap=0)
    assert len(wins) == 3
    # 窓は CHUNK_SEPARATOR 境界で切れ、各窓本文はセグメントそのもの（区切りは含まない）
    assert [text[s:e] for s, e in wins] == segs


def test_greedy_packing_groups_segments_under_budget() -> None:
    segs = ["あ", "い", "う", "え"]  # 各 1 トークン程度
    text = CHUNK_SEPARATOR.join(segs)
    wins = iter_windows(text, max_tokens=2, overlap=0)
    # 2 トークン予算 → 2 セグメントずつ ＝ 2 窓
    assert len(wins) == 2
    assert text[wins[0][0]:wins[0][1]] == "あ" + CHUNK_SEPARATOR + "い"
    assert text[wins[1][0]:wins[1][1]] == "う" + CHUNK_SEPARATOR + "え"


def test_oversized_segment_is_its_own_window() -> None:
    big = "長い本文" * 50
    text = CHUNK_SEPARATOR.join(["短い", big, "短い2"])
    wins = iter_windows(text, max_tokens=5, overlap=0)
    # big 単独で予算超でも 1 窓として出る（さらに分割しない）
    bodies = [text[s:e] for s, e in wins]
    assert big in bodies


def test_overlap_extends_later_windows_backward() -> None:
    segs = ["seg0", "seg1", "seg2", "seg3"]
    text = CHUNK_SEPARATOR.join(segs)
    no_ov = iter_windows(text, max_tokens=1, overlap=0)
    ov = iter_windows(text, max_tokens=1, overlap=100)
    assert len(no_ov) == len(ov) == 4
    # overlap=0 の 2 窓目は seg1 のみ。overlap 有では直前 seg0 までさかのぼって含む。
    assert text[no_ov[1][0]:no_ov[1][1]] == "seg1"
    assert ov[1][0] < no_ov[1][0]
    assert "seg0" in text[ov[1][0]:ov[1][1]]
