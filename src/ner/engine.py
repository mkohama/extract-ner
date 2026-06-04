"""固有表現抽出エンジン（UI 非依存）。

テキストから固有表現を抽出する。特定のカテゴリ（ラベル）だけを抜き出す
フィルタリングにも対応する。Streamlit / CLI などの表示層からはこのエンジンを
呼び出すだけにし、エンジン自体は表示・IO に依存しない。

使用例::

    from src.ner import NerEngine

    engine = NerEngine("ja_ginza_electra")
    result = engine.extract("銀座のSONYに勤める由利さん", labels=["Company"])
    for ent in result.entities:
        print(ent.text, ent.label)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import spacy

from src.ner.preprocess import prepare_for_ner

# 利用可能な GiNZA モデル（先頭が既定）
AVAILABLE_MODELS: tuple[str, ...] = ("ja_ginza_electra", "ja_ginza")
DEFAULT_MODEL = AVAILABLE_MODELS[0]


@dataclass(frozen=True)
class Entity:
    """抽出された 1 件の固有表現。"""

    text: str
    label: str
    start: int  # 解析対象テキスト中の開始文字位置
    end: int  # 同・終了文字位置


@dataclass(frozen=True)
class ExtractionResult:
    """抽出結果。

    Attributes:
        text: 実際に解析対象となったテキスト（前処理を行った場合は前処理後）。
        entities: 抽出された固有表現のタプル。
    """

    text: str
    entities: tuple[Entity, ...]

    @property
    def labels(self) -> list[str]:
        """結果に含まれるラベル（カテゴリ）の一覧（ソート済み）。"""
        return sorted({ent.label for ent in self.entities})

    def filter(self, labels: Iterable[str]) -> ExtractionResult:
        """指定したカテゴリの固有表現だけを残した結果を返す。"""
        allow = set(labels)
        return ExtractionResult(
            text=self.text,
            entities=tuple(e for e in self.entities if e.label in allow),
        )


class NerEngine:
    """GiNZA を用いた固有表現抽出エンジン。

    モデルは初回の解析時に遅延ロードする（生成自体は軽量）。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @cached_property
    def nlp(self) -> spacy.language.Language:
        """GiNZA モデル（遅延ロードしてインスタンス内でキャッシュ）。"""
        return spacy.load(self.model_name)

    def available_labels(self) -> list[str]:
        """このモデルが出力しうる全ラベル（カテゴリ）の一覧。"""
        return sorted(self.nlp.get_pipe("ner").labels)

    def extract(
        self,
        text: str,
        *,
        labels: Iterable[str] | None = None,
        flatten_tables: bool = False,
    ) -> ExtractionResult:
        """テキストから固有表現を抽出する。

        Args:
            text: 解析対象のテキスト。
            labels: 残すカテゴリ（ラベル）。None なら全件。
            flatten_tables: True なら Markdown テーブルを平文化してから解析する。

        Returns:
            ExtractionResult（解析対象テキストと抽出結果）。
        """
        if flatten_tables:
            text = prepare_for_ner(text)

        doc = self.nlp(text)
        entities = tuple(
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        )
        result = ExtractionResult(text=text, entities=entities)
        if labels is not None:
            result = result.filter(labels)
        return result
