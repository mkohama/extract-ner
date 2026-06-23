"""LLM 検出アダプタ（薄い層）。

LLM による PII 検出の**本体は pii-masker**（プロンプト・Azure クライアント・detect・locate を所有）。
本パッケージは pii-masker を**依存として呼ぶだけ**のアダプタで、data-redactor 側に閉じるのは：

- :mod:`src.llm.windows`     … J1 本文（`build_body` の text）を ~6-8k トークン窓に切る
- :mod:`src.llm.detect_layer`… 窓ループ → ``pii_masker.detect`` + ``locate_all`` → 全文スパンへ
- :mod:`src.llm.schema`      … 我々の型（``LlmSpan`` / ``LlmDetection``）

設計は docs-dev/LLM適用_調査と設計たたき台.md の §11（Stage A）を参照。
``MaskingEngine`` はこれらを import せず、算出済みの :class:`~src.llm.schema.LlmDetection` を受け取るだけ。
"""

from src.llm.detect_layer import detect_document
from src.llm.schema import LlmDetection, LlmSpan
from src.llm.windows import iter_windows

__all__ = [
    "LlmSpan",
    "LlmDetection",
    "iter_windows",
    "detect_document",
]
