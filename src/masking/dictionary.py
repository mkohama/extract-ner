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
    """正規化表層 → (canonical, category) の対応表（＋任意の置換語）。"""

    def __init__(
        self,
        surface_map: dict[str, tuple[str, str]],
        placeholders: dict[str, str] | None = None,
    ) -> None:
        self._map = surface_map
        # canonical → 置換語（マスク後の伏せ字）。指定が無い canonical は自動採番に従う。
        self._placeholders = placeholders or {}

    @classmethod
    def empty(cls) -> MaskDictionary:
        return cls({})

    @classmethod
    def load(cls, path: str | Path) -> MaskDictionary:
        """YAML を読み込んでマスク辞書を作る。

        各エントリは文字列（canonical のみ）か、``canonical``/``aliases``/``mask`` を持つ辞書。
        ``mask`` を指定すると、その実体のマスク後の置換語を固定できる（未指定なら自動採番）。
        """
        surface_map: dict[str, tuple[str, str]] = {}
        placeholders: dict[str, str] = {}
        for entry in load_entries(path):
            canonical, category = entry["canonical"], entry["category"]
            if not canonical:
                continue
            for surface in [canonical, *entry["aliases"]]:
                surface_map[normalize(surface)] = (canonical, category)
            if entry.get("mask"):
                placeholders[canonical] = entry["mask"]
        return cls(surface_map, placeholders)

    def canonical_of(self, surface: str) -> str | None:
        """表層に対応する canonical（代表表記）を返す。辞書に無ければ None。

        別表記（英語表記↔カタカナ表記・略称・旧称 等）を 1 つの canonical に束ねるのに使う
        ＝マスク後のプレースホルダを表記ゆれによらず統一できる。
        """
        entry = self._map.get(normalize(surface))
        return entry[0] if entry is not None else None

    def custom_placeholder(self, canonical: str) -> str | None:
        """canonical に対して指定された置換語（あれば）。無ければ None（自動採番に従う）。"""
        return self._placeholders.get(canonical)

    def __bool__(self) -> bool:
        return bool(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def embedded(self, token_surfaces: list[str]) -> list[tuple[int, str, str]]:
        """辞書語を**部分文字列として内包するが、トークン単位では一致しない**トークンを返す。

        返り値は ``(token_index, canonical, category)``。``SmashMark``（商標 Smash を内包）や
        ``SonyXXX``（社名 Sony を内包）のように、トークン単位照合では取りこぼす語の**監査用**。
        トークン全体一致（``match`` で拾える分）や、雑音になりやすい 1 文字辞書語は除く。
        """
        out: list[tuple[int, str, str]] = []
        seen: set[tuple[int, str]] = set()
        for i, surface in enumerate(token_surfaces):
            ns = normalize(surface)
            if ns in self._map:
                continue  # トークン全体が辞書語＝match() で拾える（漏れではない）
            for key, (canonical, category) in self._map.items():
                if len(key) >= 2 and key in ns and (i, canonical) not in seen:
                    seen.add((i, canonical))
                    out.append((i, canonical, category))
        return out

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


# YAML セクション名（カテゴリ → 書き出し時のセクション）。読み込みは _SECTION_CATEGORY で吸収。
_CATEGORY_SECTION = {"社名": "社名", "商標": "商標", "人名": "人名"}


def load_entries(path: str | Path) -> list[dict]:
    """YAML を**構造のまま**読み込む（UI 編集・round-trip 用）。

    返り値は ``{"category", "canonical", "aliases": list[str], "mask": str}`` の列。
    （:meth:`MaskDictionary.load` はこれを正規化表層マップに畳む）

    旧バグで置換に書かれてしまった文字列 ``"nan"`` は空（未指定）として読み込む（自己修復。
    再保存すれば YAML からも消える）。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries: list[dict] = []
    for section, items in raw.items():
        category = _SECTION_CATEGORY.get(section, section)
        for item in items or []:
            if isinstance(item, str):
                entries.append(
                    {"category": category, "canonical": item, "aliases": [], "mask": ""}
                )
            else:
                mask = str(item.get("mask") or "")
                if mask.strip().lower() == "nan":  # 旧バグの掃除
                    mask = ""
                entries.append(
                    {
                        "category": category,
                        "canonical": item.get("canonical") or item.get("name") or "",
                        "aliases": list(item.get("aliases") or []),
                        "mask": mask,
                    }
                )
    return entries


def save_entries(path: str | Path, entries: list[dict]) -> None:
    """構造化エントリを YAML に書き出す（UI 保存用）。

    canonical が空のエントリは捨てる。別名・置換が無ければ文字列だけの簡潔形で書く。
    セクション順は 社名 → 商標 → 人名 →（その他）。
    """
    sections: dict[str, list] = {}
    for e in entries:
        canonical = (e.get("canonical") or "").strip()
        if not canonical:
            continue
        category = e.get("category") or "社名"
        section = _CATEGORY_SECTION.get(category, category)
        aliases = [a.strip() for a in (e.get("aliases") or []) if a.strip()]
        mask = (e.get("mask") or "").strip()
        item: dict | str
        if aliases or mask:
            obj: dict = {"canonical": canonical}
            if aliases:
                obj["aliases"] = aliases
            if mask:
                obj["mask"] = mask
            item = obj
        else:
            item = canonical
        sections.setdefault(section, []).append(item)

    ordered = {s: sections[s] for s in ("社名", "商標", "人名") if s in sections}
    ordered.update({s: v for s, v in sections.items() if s not in ordered})
    Path(path).write_text(
        yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )
