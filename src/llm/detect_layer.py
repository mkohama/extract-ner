"""Stage A: LLM 検出層（薄いアダプタ）。

J1 本文 ``text`` を窓に切り、各窓で **pii-masker** の ``detect`` / ``locate_all`` を呼び、
窓内スパンに ``window_start`` を足して全文（merge）座標の :class:`~src.llm.schema.LlmDetection` を作る。
LLM 検出の本体（プロンプト・Azure・locate）は pii-masker 側。ここはオーケストレーションのみ。

実機では ``pii_masker`` を依存（git submodule + path-injection, §8）として呼ぶ。開発・テストでは
``detect_fn`` / ``locate_fn`` を差し替えて pii-masker・Azure・GiNZA 無しで検証できる。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.llm.schema import LlmDetection, LlmSpan
from src.llm.windows import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, iter_windows

# 既定モデル（N 社制約: PII を含むデータは Azure OpenAI gpt-4.1-mini のみ）。
DEFAULT_MODEL = "gpt-4.1-mini"

# pii-masker のオブジェクト（Entity: .ene_type/.text/.reason、Match: .start/.end/.entity/.how）を
# 構造的に受けるため Any で扱う（アダプタ境界。型は pii-masker 管轄）。
DetectFn = Callable[..., Sequence[Any]]
LocateFn = Callable[..., tuple[Sequence[Any], Sequence[Any]]]


def _default_detect(document: str, *, model: str) -> Sequence[Any]:
    """pii-masker の検出（遅延 import＝実機でのみ pii_masker を要求）。"""
    from pii_masker.detector_llm import detect

    return detect(document, model=model)


def _default_locate(body: str, detections: Sequence[Any]) -> tuple[Sequence[Any], Sequence[Any]]:
    """pii-masker の text→span（遅延 import）。窓 ``body`` 基準のスパンを返す。"""
    from pii_masker.locate import locate_all

    return locate_all(body, list(detections))


def detect_document(
    text: str,
    *,
    detector_version: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    detect_fn: DetectFn | None = None,
    locate_fn: LocateFn | None = None,
) -> LlmDetection:
    """J1 本文 ``text`` に対し LLM 検出を行い :class:`LlmDetection`（全文座標）を返す。

    手順: ``iter_windows`` で窓化 → 各窓 ``w=text[ws:we]`` で ``detect_fn(w)`` →
    ``locate_fn(w, ents)`` で窓内スパン → ``+ws`` で全文座標へ → 全窓を集約し重なりを解消。

    ``detect_fn`` / ``locate_fn`` 未指定時は pii-masker を呼ぶ（実機）。テストでは差し替える。
    """
    detect_fn = detect_fn or _default_detect
    locate_fn = locate_fn or _default_locate

    spans: list[LlmSpan] = []
    not_found: list[tuple[str, str]] = []
    for ws, we in iter_windows(text, max_tokens=max_tokens, overlap=overlap_tokens):
        window = text[ws:we]
        entities = detect_fn(window, model=model)
        matches, nf = locate_fn(window, entities)
        for m in matches:
            ent = m.entity
            spans.append(
                LlmSpan(
                    start=m.start + ws,
                    end=m.end + ws,
                    ene_type=ent.ene_type,
                    reason=getattr(ent, "reason", None),
                    how=getattr(m, "how", ""),
                )
            )
        for e in nf:
            not_found.append((e.ene_type, e.text))

    return LlmDetection(
        spans=tuple(_resolve_overlaps(spans)),
        not_found=tuple(dict.fromkeys(not_found)),  # 重複（窓 overlap 由来）を畳む・順序保持
        model=model,
        detector_version=detector_version,
    )


def _resolve_overlaps(spans: list[LlmSpan]) -> list[LlmSpan]:
    """窓をまたいだ重複・内包スパンを解消する（pii-masker locate._resolve と同方針）。

    長いスパン優先で確定し、既存の確定スパンに**完全内包される**スパン（重複も含む）は捨てる。
    部分重複（互いに内包しない）は両方残す。窓 overlap で同一実体が2回出ても 1 つに畳まれる。
    """
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
    kept: list[LlmSpan] = []
    for s in ordered:
        if s.start >= s.end:
            continue
        if any(k.start <= s.start and s.end <= k.end for k in kept):
            continue  # 既存の確定スパンに完全内包（重複含む）→ 捨てる
        kept.append(s)
    return sorted(kept, key=lambda s: (s.start, -(s.end - s.start)))


__all__ = ["detect_document", "DEFAULT_MODEL"]
