"""``src.llm.detect_layer.detect_document`` のテスト（pii-masker/Azure/spaCy 非依存）。

pii-masker の ``detect`` / ``locate_all`` を**スタブ**で差し替え、アダプタの責務だけを検証する：
窓内スパンへの ``window_start`` 加算・not_found の伝播・窓 overlap 由来の重複解消・メタ保持。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.detect_layer import detect_document
from src.ner.preprocess import CHUNK_SEPARATOR


# --- pii-masker の Entity / Match を模した最小スタブ ------------------------- #
@dataclass
class FakeEntity:
    ene_type: str
    text: str
    reason: str | None = None


@dataclass
class FakeMatch:
    start: int
    end: int
    entity: FakeEntity
    how: str = "exact"


def detect_from(targets: list[FakeEntity]):
    """窓本文に ``text`` が現れる検出だけ返すスタブ detect。"""

    def _detect(document: str, *, model: str):
        return [e for e in targets if e.text in document]

    return _detect


def locate_first(window: str, entities):
    """各検出の最初の出現を窓内オフセットで返すスタブ locate（pii-masker locate_all 相当）。"""
    matches, not_found = [], []
    for e in entities:
        i = window.find(e.text)
        if i >= 0:
            matches.append(FakeMatch(i, i + len(e.text), e))
        else:
            not_found.append(e)
    return matches, not_found


# --- テスト ----------------------------------------------------------------- #
def test_single_window_propagates_fields_and_offsets() -> None:
    text = "田中さんと佐藤さん"
    targets = [
        FakeEntity("Person", "田中", "姓"),
        FakeEntity("Person", "佐藤", "姓"),
    ]
    det = detect_document(
        text,
        detector_version="v1",
        detect_fn=detect_from(targets),
        locate_fn=locate_first,
    )
    assert det.model == "gpt-4.1-mini"
    assert det.detector_version == "v1"
    assert det.not_found == ()
    by_text = {text[s.start:s.end]: s for s in det.spans}
    assert set(by_text) == {"田中", "佐藤"}
    assert by_text["田中"].ene_type == "Person"
    assert by_text["田中"].reason == "姓"
    assert by_text["田中"].how == "exact"


def test_window_start_applied_in_later_window() -> None:
    """2 窓目で検出された語の span が全文座標（+window_start）になること。"""
    text = CHUNK_SEPARATOR.join(["AAAA", "佐藤"])  # 佐藤 は offset 6
    det = detect_document(
        text,
        detector_version="v1",
        max_tokens=1,  # 各セグメントを別窓に
        overlap_tokens=0,
        detect_fn=detect_from([FakeEntity("Person", "佐藤")]),
        locate_fn=locate_first,
    )
    assert len(det.spans) == 1
    s = det.spans[0]
    assert (s.start, s.end) == (6, 8)
    assert text[s.start:s.end] == "佐藤"


def test_not_found_is_propagated() -> None:
    text = "本文だけ"

    def detect_ghost(document: str, *, model: str):
        return [FakeEntity("Person", "GHOST")]  # 本文に無い＝locate 不能

    det = detect_document(
        text,
        detector_version="v1",
        detect_fn=detect_ghost,
        locate_fn=locate_first,
    )
    assert det.spans == ()
    assert det.not_found == (("Person", "GHOST"),)


def test_overlap_duplicate_is_resolved() -> None:
    """窓 overlap で同一実体が2窓に出ても 1 スパンに畳まれる。"""
    text = CHUNK_SEPARATOR.join(["AAAA", "TARGET", "BBBB"])  # TARGET は offset 6..12
    det = detect_document(
        text,
        detector_version="v1",
        max_tokens=1,
        overlap_tokens=100,  # 2 窓目以降が直前セグメントを含む＝TARGET が2窓に出る
        detect_fn=detect_from([FakeEntity("Org", "TARGET")]),
        locate_fn=locate_first,
    )
    assert len(det.spans) == 1
    s = det.spans[0]
    assert (s.start, s.end) == (6, 12)
    assert text[s.start:s.end] == "TARGET"


def test_empty_text() -> None:
    det = detect_document(
        "",
        detector_version="v1",
        detect_fn=detect_from([FakeEntity("Person", "田中")]),
        locate_fn=locate_first,
    )
    assert det.spans == ()
    assert det.not_found == ()
    assert det.detector_version == "v1"
