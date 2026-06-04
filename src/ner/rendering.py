"""固有表現の displaCy 表示（プレゼンテーション層）。

エンジンの抽出結果 (ExtractionResult) を displaCy のハイライト HTML に変換する。
色マップの生成もここで行う。CLI / UI の両方から再利用する。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spacy import displacy

from src.ner.engine import ExtractionResult

# 拡張固有表現ラベルごとの配色（大文字キーで定義）
ENT_COLORS: dict[str, str] = {
    "PERSON": "#b69bf2",
    "N_PERSON": "#5b9bf5",
    "POSITION_VOCATION": "#c9b3f7",
    "DATE": "#9be88a",
    "TIME": "#a6e22e",
    "AGE": "#5b9bf5",
    "MONEY": "#f7e84a",
    "N_EVENT": "#5b9bf5",
    "N_LOCATION_OTHER": "#5b9bf5",
    "OCCASION_OTHER": "#6ddf6d",
    "PRINTING_OTHER": "#d9d9d9",
    "TITLE_OTHER": "#d9d9d9",
    "RANK": "#c9b3f7",
    "URL": "#d9d9d9",
}
# 未定義ラベルの既定色
DEFAULT_COLOR = "#fdd8a5"


def build_color_map(labels: Iterable[str]) -> dict[str, str]:
    """ラベル一覧から displaCy 用の色マップを作る。

    GiNZA の実ラベルは Title case（例: "Person"）だが ENT_COLORS のキーは大文字。
    displaCy は内部で色マップのキーを大文字化するため、両者を混在させると
    "Person"(既定色) が "PERSON"(指定色) を上書きしてしまう。これを防ぐため、
    モデルの実ラベル表記をキーにして、大文字で突き合わせた色を割り当てる。
    """
    palette = {key.upper(): color for key, color in ENT_COLORS.items()}
    return {label: palette.get(label.upper(), DEFAULT_COLOR) for label in labels}


def to_displacy_data(result: ExtractionResult) -> dict[str, Any]:
    """ExtractionResult を displaCy の manual 形式に変換する。"""
    return {
        "text": result.text,
        "ents": [
            {"start": ent.start, "end": ent.end, "label": ent.label}
            for ent in result.entities
        ],
        "title": None,
    }


def render_html(
    result: ExtractionResult,
    colors: dict[str, str],
    *,
    page: bool = False,
) -> str:
    """抽出結果を displaCy のハイライト HTML にレンダリングする。"""
    return displacy.render(
        to_displacy_data(result),
        style="ent",
        manual=True,
        options={"colors": colors},
        page=page,
    )
