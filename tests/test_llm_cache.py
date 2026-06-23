"""LLM 検出キャッシュの round-trip と ``cached_detect``（pii-masker/Azure 非依存）。

- ``detection_to_json`` / ``detection_from_json`` の往復で内容が保たれる。
- ``NerCache.get_llm`` / ``put_llm`` が (content_hash, model, flatten, detector_version) で引ける。
- ``cached_detect`` は初回のみ検出を走らせ、2 回目はキャッシュを返す（detector_version 改版でミス）。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.detect_layer import cached_detect
from src.llm.schema import (
    LlmDetection,
    LlmSpan,
    detection_from_json,
    detection_to_json,
)
from src.masking.cache import NerCache


def _sample() -> LlmDetection:
    return LlmDetection(
        spans=(
            LlmSpan(0, 2, "Person", "姓", "exact"),
            LlmSpan(5, 9, "Company", None, "normalized"),
        ),
        not_found=(("Email", "x@example.com"),),
        model="gpt-4.1-mini",
        detector_version="v1",
    )


def test_json_roundtrip() -> None:
    d = _sample()
    assert detection_from_json(detection_to_json(d)) == d


def test_cache_get_put_roundtrip(tmp_path) -> None:
    cache = NerCache(tmp_path / "cache.db")
    h, model, ver = "hash1", "gpt-4.1-mini", "v1"
    assert cache.get_llm(h, model, False, ver) is None  # miss
    cache.put_llm(h, model, False, ver, detection_to_json(_sample()))
    got = cache.get_llm(h, model, False, ver)
    assert got is not None
    assert detection_from_json(got) == _sample()
    # 鍵の各次元が効く（flatten / detector_version 違いは別エントリ＝ミス）
    assert cache.get_llm(h, model, True, ver) is None
    assert cache.get_llm(h, model, False, "v2") is None


# --- cached_detect のスタブ ------------------------------------------------- #
@dataclass
class _Ent:
    ene_type: str
    text: str
    reason: str | None = None


@dataclass
class _Match:
    start: int
    end: int
    entity: _Ent
    how: str = "exact"


def test_cached_detect_runs_once_then_hits_cache(tmp_path) -> None:
    cache = NerCache(tmp_path / "cache.db")
    calls = {"n": 0}

    def detect_fn(document: str, *, model: str):
        calls["n"] += 1
        return [_Ent("Person", "田中")] if "田中" in document else []

    def locate_fn(window: str, ents):
        out = []
        for e in ents:
            i = window.find(e.text)
            if i >= 0:
                out.append(_Match(i, i + len(e.text), e))
        return out, []

    text = "田中さんの記録"

    def run(version: str = "v1") -> LlmDetection:
        return cached_detect(
            cache,
            "h",
            text,
            flatten=False,
            detector_version=version,
            detect_fn=detect_fn,
            locate_fn=locate_fn,
        )

    d1 = run()
    assert calls["n"] == 1
    assert len(d1.spans) == 1 and text[d1.spans[0].start : d1.spans[0].end] == "田中"

    d2 = run()
    assert calls["n"] == 1  # 2 回目は検出を走らせない（キャッシュヒット）
    assert d2 == d1

    # detector_version を上げるとミス＝再検出
    run("v2")
    assert calls["n"] == 2

    # force=True は同じ鍵でもキャッシュを無視して再検出する（やり直し）
    cached_detect(
        cache,
        "h",
        text,
        flatten=False,
        detector_version="v1",
        detect_fn=detect_fn,
        locate_fn=locate_fn,
        force=True,
    )
    assert calls["n"] == 3
