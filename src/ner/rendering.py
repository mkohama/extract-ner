"""固有表現の displaCy 表示（プレゼンテーション層）。

エンジンの抽出結果 (ExtractionResult) を displaCy のハイライト HTML に変換する。
色マップの生成もここで行う。CLI / UI の両方から再利用する。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from spacy import displacy

from src.ner.engine import ExtractionResult

# 配色方針（マスキング用途）:
#   このツールの目的は「LLM に渡す前の機密情報マスキング」。そこで
#   「マスク対象として重要なカテゴリほど目立つ」配色にする。
#   - 重要 PII（人名 / 社名・組織 / 商標・製品 / 地名・住所 / 連絡先）= 彩度の高い固有色。
#   - 非 PII（日付 / 時刻 / 数値 / 肩書 / 印刷物 など）= 淡いグレーで背景化（DEFAULT_COLOR）。
#   GiNZA の実ラベルは Title case（例: "Company"）だが displaCy が色マップのキーを
#   大文字化するため、ここでは大文字キーで定義する（build_color_map 参照）。
#   背景色の上に黒文字が乗るので、いずれも黒文字が読める明度にしている。

# グループ色（マスク対象として重要なカテゴリ）
_PERSON = "#ff8fab"  # 人名: ピンク（最重要・本丸）
_ORG = "#ffab5e"  # 社名・組織: オレンジ（最重要・本丸）
_PRODUCT = "#c792ea"  # 商標・製品: 紫（中に秘匿語が紛れる。要確認）
_LOCATION = "#4fc3b0"  # 地名・住所: ティール
_CONTACT = "#6cb6ff"  # 連絡先（電話/メール/URL）: 青

# ラベル → グループ（大文字キー）。関根の拡張固有表現体系のうちマスクで効くものを束ねる。
# 注: 「N_*」（N_Person=人数 等）は数値表現（個数）であり固有名詞ではないので含めない。
_PERSON_LABELS = ("PERSON",)
_ORG_LABELS = (
    "COMPANY",
    "COMPANY_GROUP",
    "CORPORATION_OTHER",
    "SHOW_ORGANIZATION",
    "INTERNATIONAL_ORGANIZATION",
    "POLITICAL_ORGANIZATION_OTHER",
    "GOVERNMENT",
    "POLITICAL_PARTY",
    "SPORTS_ORGANIZATION_OTHER",
)
_PRODUCT_LABELS = (
    "PRODUCT_OTHER",
    "AWARD",
    "DECORATION",
    "OFFENSE",
    "SERVICE",
    "CLASS",
)
_LOCATION_LABELS = (
    "CITY",
    "PROVINCE",
    "COUNTRY",
    "GPE_OTHER",
    "ADDRESS",
    "POSTAL_ADDRESS",
    "GEOLOGICAL_REGION_OTHER",
    "LOCATION_OTHER",
    "DOMESTIC_REGION_OTHER",
    "CONTINENTAL_REGION_OTHER",
    "FACILITY_OTHER",
    "FACILITY_PART",
    "STATION",
    "AIRPORT",
)
_CONTACT_LABELS = ("PHONE_NUMBER", "EMAIL", "URL")

# 拡張固有表現ラベルごとの配色（大文字キー）
ENT_COLORS: dict[str, str] = {
    **{label: _PERSON for label in _PERSON_LABELS},
    **{label: _ORG for label in _ORG_LABELS},
    **{label: _PRODUCT for label in _PRODUCT_LABELS},
    **{label: _LOCATION for label in _LOCATION_LABELS},
    **{label: _CONTACT for label in _CONTACT_LABELS},
}

# マスク対象として固有色を割り当てる「重要カテゴリ」の集合（大文字）。
# 将来のマスキング/戦略比較ロジックからも参照できるよう公開する。
MASKING_CRITICAL_LABELS: frozenset[str] = frozenset(ENT_COLORS)

# 重要カテゴリ以外（非 PII = 日付・数値・肩書・印刷物など）の既定色。
# 淡いグレーにして視覚的に背景化し、PII を際立たせる。
DEFAULT_COLOR = "#dcdcdc"


def build_color_map(labels: Iterable[str]) -> dict[str, str]:
    """ラベル一覧から displaCy 用の色マップを作る。

    GiNZA の実ラベルは Title case（例: "Person"）だが ENT_COLORS のキーは大文字。
    displaCy は内部で色マップのキーを大文字化するため、両者を混在させると
    "Person"(既定色) が "PERSON"(指定色) を上書きしてしまう。これを防ぐため、
    モデルの実ラベル表記をキーにして、大文字で突き合わせた色を割り当てる。
    """
    palette = {key.upper(): color for key, color in ENT_COLORS.items()}
    return {label: palette.get(label.upper(), DEFAULT_COLOR) for label in labels}


# マスキングのカテゴリ → 色（src.masking のカテゴリ名に対応）
MASKING_CATEGORY_COLORS: dict[str, str] = {
    "人名": _PERSON,
    "社名": _ORG,
    "商標": _PRODUCT,
    "地名": _LOCATION,
    "連絡先": _CONTACT,
    "その他": DEFAULT_COLOR,
}


def render_masking_html(
    text: str,
    spans: Iterable[tuple[int, int, str]],
    *,
    page: bool = False,
) -> str:
    """マスク候補スパンを displaCy のハイライト HTML にする。

    Args:
        text: 元テキスト。
        spans: (start, end, category) のイテラブル。category は MASKING_CATEGORY_COLORS のキー。
    """
    ents: list[dict[str, Any]] = []
    last_end = -1
    for start, end, category in sorted(spans, key=lambda s: s[0]):
        if start < last_end:  # displaCy は重なりを許さないので捨てる
            continue
        ents.append({"start": start, "end": end, "label": category})
        last_end = end
    colors = {k.upper(): v for k, v in MASKING_CATEGORY_COLORS.items()}
    return displacy.render(
        {"text": text, "ents": ents, "title": None},
        style="ent",
        manual=True,
        options={"colors": colors},
        page=page,
    )


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
