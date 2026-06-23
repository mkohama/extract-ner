"""LLM 検出結果の型（data-redactor 側の表現）。

pii-masker の ``Entity``/``Match`` から詰め直して、キャッシュ・表示（出口1）・票変換（Stage B）で使う。
pii-masker の型をそのまま外に漏らさない（依存をアダプタ境界で閉じる）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlmSpan:
    """LLM が検出し本文に位置特定できた 1 スパン（``Body.text`` 座標）。

    ``start``/``end`` は窓内 locate の結果に ``window_start`` を足した**全文（merge）座標**。
    """

    start: int
    end: int
    ene_type: str  # pii-masker の ENE type（Person / Company / ... / Trademark）
    reason: str | None  # LLM の判断理由（出口1 で表示）
    how: str  # pii-masker Match.how（"exact"/"normalized"/...）由来の可視化用


@dataclass(frozen=True)
class LlmDetection:
    """1 文書（チャンク列）に対する LLM 検出結果（Stage A の出力）。

    ``spans`` は位置特定できた検出、``not_found`` は本文に当てられなかった検出の
    ``(ene_type, text)``（出口1 で「要確認」として見せる）。``model``/``detector_version``
    はキャッシュ鍵・再現性のために保持する（``detector_version`` ＝ pii-masker の版＋窓ポリシー）。
    """

    spans: tuple[LlmSpan, ...]
    not_found: tuple[tuple[str, str], ...]
    model: str
    detector_version: str


__all__ = ["LlmSpan", "LlmDetection"]
