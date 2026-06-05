"""固有表現抽出エンジン（UI 非依存）。

公開 API:
    NerEngine        : 固有表現抽出エンジン
    Entity           : 抽出された 1 件の固有表現
    ExtractionResult : 抽出結果（カテゴリ絞り込み用の filter / labels を持つ）
    AVAILABLE_MODELS, DEFAULT_MODEL
    prepare_for_ner, flatten_markdown_tables : 前処理
    ENT_COLORS, build_color_map, render_html, to_displacy_data : 表示（displaCy）
"""

from src.ner.engine import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    Entity,
    ExtractionResult,
    NerEngine,
    TokenInfo,
)
from src.ner.preprocess import flatten_markdown_tables, prepare_for_ner
from src.ner.rendering import (
    DEFAULT_COLOR,
    ENT_COLORS,
    MASKING_CRITICAL_LABELS,
    build_color_map,
    render_html,
    to_displacy_data,
)

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "Entity",
    "ExtractionResult",
    "NerEngine",
    "TokenInfo",
    "prepare_for_ner",
    "flatten_markdown_tables",
    "ENT_COLORS",
    "DEFAULT_COLOR",
    "MASKING_CRITICAL_LABELS",
    "build_color_map",
    "render_html",
    "to_displacy_data",
]
