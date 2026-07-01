"""Stage A: LLM 検出層（薄いアダプタ）。

J1 本文 ``text`` を窓に切り、各窓で **pii-masker** の ``detect`` / ``locate_all`` を呼び、
窓内スパンに ``window_start`` を足して全文（merge）座標の :class:`~src.llm.schema.LlmDetection` を作る。
LLM 検出の本体（プロンプト・Azure・locate）は pii-masker 側。ここはオーケストレーションのみ。

実機では ``pii_masker`` を依存（git submodule + path-injection, §8）として呼ぶ。開発・テストでは
``detect_fn`` / ``locate_fn`` を差し替えて pii-masker・Azure・GiNZA 無しで検証できる。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Protocol

import openai

from src.llm.schema import (
    LlmDetection,
    LlmSpan,
    detection_from_json,
    detection_to_json,
)
from src.llm.windows import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, iter_windows

# 既定モデル（N 社制約: PII を含むデータは Azure OpenAI gpt-4.1-mini のみ）。
DEFAULT_MODEL = "gpt-4.1-mini"

# --- 止血: 一過性 429（Azure バックエンド混雑）を呼び出し側で吸収する ---------------------
# pii-masker の get_client が max_retries を指定しておらず SDK 既定=2 では吸収しきれないため、
# 実機の検出呼び出し（_default_detect）をリトライ＋バックオフで包み、窓間にスロットルを挟む。
# 恒久対応（pii-masker 側で max_retries を増やす・1窓の呼び出し回数を減らす）は報告済み。
# 429 が窓内 7〜8 発のどこで出ても窓ごと再試行する粗い単位（内部個別リクエスト単位では回せない）。
LLM_MAX_RETRIES = int(os.getenv("LLM_DETECT_MAX_RETRIES", "5"))
# 指数バックオフの基数（秒）。Retry-After ヘッダがあればそちらを優先。
LLM_RETRY_BASE_WAIT = float(os.getenv("LLM_DETECT_RETRY_BASE_WAIT", "2.0"))
# 窓ごとの検出前に挟む待機（秒）。連射のピークを均して 429 を引きにくくする。0 で無効。
LLM_WINDOW_THROTTLE = float(os.getenv("LLM_DETECT_WINDOW_THROTTLE", "0.5"))


def _retry_wait(exc: openai.RateLimitError, attempt: int) -> float:
    """429 の待機秒。Retry-After ヘッダがあれば尊重、無ければ指数バックオフ（2^attempt 倍）。"""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        val = headers.get("retry-after")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    return LLM_RETRY_BASE_WAIT * (2**attempt)


def _call_with_retry(fn: Callable[[], Sequence[Any]]) -> Sequence[Any]:
    """``fn`` を呼び、一過性 429（RateLimitError）は待機して最大 LLM_MAX_RETRIES 回まで再試行。"""
    last: openai.RateLimitError | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return fn()
        except openai.RateLimitError as e:  # 一過性。待って再試行。
            last = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(_retry_wait(e, attempt))
    assert last is not None  # ループは最低1回回るので 429 経由なら必ず設定される
    raise last


# pii-masker のオブジェクト（Entity: .ene_type/.text/.reason、Match: .start/.end/.entity/.how）を
# 構造的に受けるため Any で扱う（アダプタ境界。型は pii-masker 管轄）。
DetectFn = Callable[..., Sequence[Any]]
LocateFn = Callable[..., tuple[Sequence[Any], Sequence[Any]]]
# progress(window_index, total_windows)。UI の進捗表示用（任意）。
ProgressFn = Callable[[int, int], None]


def _default_detect(document: str, *, model: str, target: str = "all") -> Sequence[Any]:
    """pii-masker の検出（遅延 import＝実機でのみ pii_masker を要求）。

    ``target`` は pii-masker の検出対象プリセット名（``all``＝人名/社名/商標の調教済み3種、
    ``pii``＝従来の全type汎用）。このアダプタは値を解釈せず pii-masker へ素通しする。
    """
    from pii_masker.detector_llm import detect

    if LLM_WINDOW_THROTTLE > 0:
        time.sleep(LLM_WINDOW_THROTTLE)  # 窓間スロットル（連射のピークを均す）
    return _call_with_retry(lambda: detect(document, model=model, target=target))


def _default_locate(
    body: str, detections: Sequence[Any]
) -> tuple[Sequence[Any], Sequence[Any]]:
    """pii-masker の text→span（遅延 import）。窓 ``body`` 基準のスパンを返す。"""
    from pii_masker.locate import locate_all

    return locate_all(body, list(detections))


def detect_document(
    text: str,
    *,
    detector_version: str,
    model: str = DEFAULT_MODEL,
    target: str = "all",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    detect_fn: DetectFn | None = None,
    locate_fn: LocateFn | None = None,
    progress: ProgressFn | None = None,
) -> LlmDetection:
    """J1 本文 ``text`` に対し LLM 検出を行い :class:`LlmDetection`（全文座標）を返す。

    手順: ``iter_windows`` で窓化 → 各窓 ``w=text[ws:we]`` で ``detect_fn(w)`` →
    ``locate_fn(w, ents)`` で窓内スパン → ``+ws`` で全文座標へ → 全窓を集約し重なりを解消。

    ``detect_fn`` / ``locate_fn`` 未指定時は pii-masker を呼ぶ（実機）。テストでは差し替える。
    ``target`` は pii-masker の検出対象プリセット名で、**既定 detect_fn にだけ**束ねて渡す
    （自前 ``detect_fn`` を渡したときは無視＝テスト/差し替えは target を意識しなくてよい）。
    ``progress(i, n)`` を渡すと各窓の検出前に (窓index, 全窓数) を通知する（UI 進捗表示用）。
    """
    detect_fn = detect_fn or partial(_default_detect, target=target)
    locate_fn = locate_fn or _default_locate

    spans: list[LlmSpan] = []
    not_found: list[tuple[str, str]] = []
    windows = iter_windows(text, max_tokens=max_tokens, overlap=overlap_tokens)
    for i, (ws, we) in enumerate(windows):
        if progress is not None:
            progress(i, len(windows))
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
        not_found=tuple(
            dict.fromkeys(not_found)
        ),  # 重複（窓 overlap 由来）を畳む・順序保持
        model=model,
        detector_version=detector_version,
    )


class LlmDetectionCache(Protocol):
    """``cached_detect`` が必要とする最小インターフェース（src.masking.cache.NerCache が満たす）。

    Protocol にして src.llm → src.masking の直接依存を避ける（鍵に flatten・detector_version を含む）。
    """

    def get_llm(
        self, content_hash: str, model: str, flatten: bool, detector_version: str
    ) -> str | None: ...

    def put_llm(
        self,
        content_hash: str,
        model: str,
        flatten: bool,
        detector_version: str,
        detections_json: str,
    ) -> None: ...


def cached_detect(
    cache: LlmDetectionCache,
    content_hash: str,
    text: str,
    *,
    flatten: bool,
    detector_version: str,
    model: str = DEFAULT_MODEL,
    target: str = "all",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    detect_fn: DetectFn | None = None,
    locate_fn: LocateFn | None = None,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> LlmDetection:
    """LLM 検出をキャッシュ越しに行う（NER 層と同じ「激重層だけキャッシュ」）。

    ``(content_hash, model, flatten, detector_version)`` でヒットすれば pii-masker を呼ばない。
    ``detector_version`` を上げると自動ミス→再取得（プロンプト/窓ポリシー改版の反映）。
    ``target`` は pii-masker の検出対象プリセット名（既定 detect_fn へ素通し）。**target を切り替える側は
    必ず ``detector_version`` にも target を織り込むこと**（版に入れないと別 target 同士でキャッシュが衝突する）。
    ``force=True`` でキャッシュを無視して再検出し上書きする（NER キャッシュには触れない）。
    ``progress(i, n)`` は detect_document へ渡る（キャッシュヒット時は呼ばれない＝即返り）。
    """
    if not force:
        hit = cache.get_llm(content_hash, model, flatten, detector_version)
        if hit is not None:
            return detection_from_json(hit)
    detection = detect_document(
        text,
        detector_version=detector_version,
        model=model,
        target=target,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        detect_fn=detect_fn,
        locate_fn=locate_fn,
        progress=progress,
    )
    cache.put_llm(
        content_hash, model, flatten, detector_version, detection_to_json(detection)
    )
    return detection


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


__all__ = ["detect_document", "cached_detect", "LlmDetectionCache", "DEFAULT_MODEL"]
