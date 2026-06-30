"""マスク辞書（社名・商標・人名の登録リスト）。

こちらが用意する固有名詞の名簿。**SudachiPy の内部辞書とは別物**。テキスト中の語を既定では
**トークン単位**で照合し、確定のマスク対象として拾う（部分文字列照合はしない＝登録語が
より長い語の一部になっているとき、例「社A」を「社A工場」の中では誤って拾わない）。
大小文字・全角半角は正規化で吸収する。

**`embed: true`** を付けた語だけは、複合トークンの**サブワード境界**でも内包照合する
（例 `Smash`→`SmashMark` の `Smash`、`CB`→`CBMark` の `CB`。境界一致なので `ECBType` の `CB` は
拾わない）。命中は一致したサブワード部分だけをマスクする（`SmashMark`→`[商標X]Mark`）。

YAML 形式（data/mask_dict.yaml。書式は data/mask_dict.sample.yaml 参照）::

    社名:
      - canonical: 社A
        aliases: [社A株式会社, ｼｬA]   # 別表記（カタカナ↔英語・略称・旧称）だけ書く
      - 社B                            # 文字列だけなら canonical 扱い（別名なし）
    商標:
      - canonical: 商標X
        embed: true                    # 複合語の中の 商標X も伏字にする（SmashMark の Smash 等）
    人名:
      - 姓A
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

# YAML のセクション名 → マスク用カテゴリ（「社員名」は「人名」の別名として受理）
_SECTION_CATEGORY = {"社名": "社名", "商標": "商標", "人名": "人名", "社員名": "人名"}
# 照合する最大トークン数（多トークン語の上限。コスト上限を兼ねる）
MAX_MATCH_TOKENS = 12


# 連続する空白（半角/全角/タブ/改行）。`match` はトークンを空白なしで連結して照合するため、
# 登録語のスペース（例 `Tokyo Electron`）も除去して一致させる。
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """照合用の正規化（NFKC で全角半角統一・casefold で大小無視・**空白除去**）。

    `match` はトークン列を空白なしで連結して辞書キーと突き合わせる。登録語が複数語
    （`Tokyo Electron` / `Nikon Precision Inc`）でも一致するよう、空白を除去して正規化する。
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text).casefold())


# 識別子をサブワードに割る正規表現（camelCase/略語/数字。区切り記号 `_-::@.` 等は自然に境界になる）。
# 例：SmashMark→[Smash,Mark] / CBMark→[CB,Mark] / ECBType→[ECB,Type] / HTTPServer→[HTTP,Server]。
_SUBWORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _split_identifier(surface: str) -> list[tuple[int, int]]:
    """識別子のサブワード境界 (start, end) のリストを返す（区切り記号自身は含めない）。"""
    return [(m.start(), m.end()) for m in _SUBWORD_RE.finditer(surface)]


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
        embed_map: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._map = surface_map
        # canonical → 置換語（マスク後の伏せ字）。指定が無い canonical は自動採番に従う。
        self._placeholders = placeholders or {}
        # `embed: true` の語だけのマップ（複合トークンのサブワード境界で内包照合する対象）。
        self._embed_map = embed_map or {}

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
        embed_map: dict[str, tuple[str, str]] = {}
        for entry in load_entries(path):
            canonical, category = entry["canonical"], entry["category"]
            if not canonical:
                continue
            for surface in [canonical, *entry["aliases"]]:
                surface_map[normalize(surface)] = (canonical, category)
                if entry.get("embed"):
                    embed_map[normalize(surface)] = (canonical, category)
            if entry.get("mask"):
                placeholders[canonical] = entry["mask"]
        return cls(surface_map, placeholders, embed_map)

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
        """辞書語を**部分文字列として内包するが、トークン単位では一致しない**トークンを返す（監査用）。

        返り値は ``(token_index, canonical, category)``。境界を見ない素朴な部分文字列照合なので
        ``ECBType`` の ``CB`` 等も拾う＝**監査専用**（実マスクは境界を見る :meth:`embedded_matches`）。
        トークン全体一致（``match`` で拾う分）や雑音になりやすい 1 文字辞書語は除く。
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

    def embedded_matches(
        self, token_surfaces: list[str]
    ) -> list[tuple[int, int, int, str, str]]:
        """`embed: true` の辞書語を、複合トークンの**サブワード境界**で内包照合する。

        トークンを camelCase/略語/区切り/数字で割り、サブワード（連続）に一致した語を返す。
        部分文字列ではなく**境界一致**なので、`SmashMark`→`Smash` / `CBMark`→`CB` は拾い、
        `ECBType` の `CB`（サブワード `ECB` の一部）は拾わない。単一サブワード＝トークン全体は
        ``match()`` の担当なので対象外（複合トークンのみ）。

        返り値 ``(token_index, sub_start, sub_end, canonical, category)``。sub_start/end は
        **そのトークン内の文字位置**（呼び出し側がトークン先頭オフセットを足して全文位置にする）。
        """
        if not self._embed_map:
            return []
        out: list[tuple[int, int, int, str, str]] = []
        for i, surface in enumerate(token_surfaces):
            subs = _split_identifier(surface)
            if len(subs) <= 1:
                continue  # 複合語のみ（全体一致は match() が拾う）
            nsubs = [normalize(surface[a:b]) for a, b in subs]
            n = len(subs)
            j = 0
            while j < n:
                length = next(
                    (
                        ln
                        for ln in range(min(MAX_MATCH_TOKENS, n - j), 0, -1)
                        if "".join(nsubs[j : j + ln]) in self._embed_map
                    ),
                    0,
                )
                if length:
                    canonical, category = self._embed_map[
                        "".join(nsubs[j : j + length])
                    ]
                    out.append(
                        (i, subs[j][0], subs[j + length - 1][1], canonical, category)
                    )
                    j += length
                else:
                    j += 1
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
                    {
                        "category": category,
                        "canonical": item,
                        "aliases": [],
                        "mask": "",
                        "embed": False,
                    }
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
                        "embed": bool(item.get("embed")),
                    }
                )
    return entries


def sort_key(canonical: str) -> str:
    """辞書エントリの並び順キー（代表表記を NFKC+casefold で正規化した辞書順）。

    照合用 :func:`normalize` と同じ正規化（大小・全角半角・空白を無視）。件数が増えても
    探しやすいよう、各セクション内を代表表記でソートする。除外リスト側と方針を揃える。
    """
    return normalize(canonical)


def save_entries(path: str | Path, entries: list[dict]) -> None:
    """構造化エントリを YAML に書き出す（UI 保存用）。

    canonical が空のエントリは捨てる。別名・置換が無ければ文字列だけの簡潔形で書く。
    セクション順は 社名 → 商標 → 人名 →（その他）。各セクション内は代表表記の辞書順にソート。
    """
    sections: dict[str, list[tuple[str, dict | str]]] = {}
    for e in entries:
        canonical = (e.get("canonical") or "").strip()
        if not canonical:
            continue
        category = e.get("category") or "社名"
        section = _CATEGORY_SECTION.get(category, category)
        aliases = [a.strip() for a in (e.get("aliases") or []) if a.strip()]
        mask = (e.get("mask") or "").strip()
        embed = bool(e.get("embed"))
        item: dict | str
        if aliases or mask or embed:
            obj: dict = {"canonical": canonical}
            if aliases:
                obj["aliases"] = aliases
            if mask:
                obj["mask"] = mask
            if embed:
                obj["embed"] = True
            item = obj
        else:
            item = canonical
        sections.setdefault(section, []).append((canonical, item))

    # 各セクション内を代表表記でソートしてから item だけ取り出す。
    sorted_sections: dict[str, list] = {
        section: [item for _, item in sorted(rows, key=lambda r: sort_key(r[0]))]
        for section, rows in sections.items()
    }

    ordered = {
        s: sorted_sections[s] for s in ("社名", "商標", "人名") if s in sorted_sections
    }
    ordered.update({s: v for s, v in sorted_sections.items() if s not in ordered})
    Path(path).write_text(
        yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )
