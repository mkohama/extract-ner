"""マスク辞書（社名・商標・人名の登録リスト）。

こちらが用意する固有名詞の名簿。**SudachiPy の内部辞書とは別物**。テキスト中の語を
**トークン単位**で照合し、確定のマスク対象として拾う（部分文字列照合はしない＝登録語が
より長い語の一部になっているとき、例「社A」を「社A工場」の中では誤って拾わない）。
大小文字・全角半角は正規化で吸収する。

YAML 形式（data/mask_dict.yaml。書式は data/mask_dict.sample.yaml 参照）::

    社名:
      - canonical: 社A
        aliases: [社A株式会社, ｼｬA]   # 別表記（カタカナ↔英語・略称・旧称）だけ書く
      - 社B                            # 文字列だけなら canonical 扱い（別名なし）
    商標:
      - 商標X
    人名:
      - 姓A
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

# YAML のセクション名 → マスク用カテゴリ（「社員名」は「人名」の別名として受理）
_SECTION_CATEGORY = {"社名": "社名", "商標": "商標", "人名": "人名", "社員名": "人名"}
# 照合する最大トークン数（多トークン語の上限。コスト上限を兼ねる）
MAX_MATCH_TOKENS = 12


def normalize(text: str) -> str:
    """照合用の正規化（全角半角を NFKC で統一し、大小文字を畳む）。"""
    return unicodedata.normalize("NFKC", text).casefold()


@dataclass(frozen=True)
class DictMatch:
    """辞書一致（トークンインデックスの半開区間 [start_token, end_token)）。"""

    start_token: int
    end_token: int
    canonical: str
    category: str


class MaskDictionary:
    """正規化表層 → (canonical, category) の対応表。"""

    def __init__(self, surface_map: dict[str, tuple[str, str]]) -> None:
        self._map = surface_map

    @classmethod
    def empty(cls) -> MaskDictionary:
        return cls({})

    @classmethod
    def load(cls, path: str | Path) -> MaskDictionary:
        """YAML を読み込んでマスク辞書を作る。"""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        surface_map: dict[str, tuple[str, str]] = {}
        for section, items in data.items():
            category = _SECTION_CATEGORY.get(section, section)
            for item in items or []:
                aliases: list[str]
                if isinstance(item, str):
                    canonical, aliases = item, []
                else:
                    canonical = item.get("canonical") or item.get("name") or ""
                    aliases = item.get("aliases") or []
                if not canonical:
                    continue
                for surface in [canonical, *aliases]:
                    surface_map[normalize(surface)] = (canonical, category)
        return cls(surface_map)

    def canonical_of(self, surface: str) -> str | None:
        """表層に対応する canonical（代表表記）を返す。辞書に無ければ None。

        別表記（英語表記↔カタカナ表記・略称・旧称 等）を 1 つの canonical に束ねるのに使う
        ＝マスク後のプレースホルダを表記ゆれによらず統一できる。
        """
        entry = self._map.get(normalize(surface))
        return entry[0] if entry is not None else None

    def __bool__(self) -> bool:
        return bool(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def match(self, token_surfaces: list[str]) -> list[DictMatch]:
        """正規化トークン列に対し、各位置で最長一致の語を拾う（重なりなし）。"""
        norm = [normalize(s) for s in token_surfaces]
        matches: list[DictMatch] = []
        i = 0
        n = len(norm)
        while i < n:
            hit: DictMatch | None = None
            for length in range(min(MAX_MATCH_TOKENS, n - i), 0, -1):
                key = "".join(norm[i : i + length])
                if key in self._map:
                    canonical, category = self._map[key]
                    hit = DictMatch(i, i + length, canonical, category)
                    break
            if hit is not None:
                matches.append(hit)
                i = hit.end_token
            else:
                i += 1
        return matches
