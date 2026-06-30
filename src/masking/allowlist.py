"""除外リスト（allowlist）。マスク候補から恒久的に外す語の名簿。

マスク辞書（[dictionary.py]）の対。NER の誤検出（社内コード・変数名・汎用語など、人名/社名でない語）を
人が「これは機密でない」と判断したら登録し、以後**どの文書でも**候補を「除外」に落とす（1 文書で
登録→他文書でも効く）。

recall 安全のための重要な制約（適用は :func:`MaskingEngine.analyze` 側）:
- 除外できるのは**検出由来（強/中/弱/微弱）のみ**。**辞書一致・連絡先 regex（＝確定）は上書きしない**
  （名簿や決定的検出を誤って外して漏らさないため）。

照合はトークン単位でなく**正規化文字列**で行う（:func:`dictionary.normalize` と同じ NFKC+casefold）。

YAML 形式（``data/mask_allowlist.yaml``。フラットなリスト）::

    除外:
      - Em_NoYes
      - Reject
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from src.masking.dictionary import normalize

# YAML のセクション名（除外語のフラットリスト）
_SECTION = "除外"

# 連続する空白（半角/全角/タブ/改行）。照合前に 1 個へ畳む。
_WHITESPACE = re.compile(r"\s+")


def _match_key(surface: str) -> str:
    """除外照合のキー：前後 strip ＋連続空白を 1 個に畳んでから正規化（NFKC+casefold）。

    複数トークンにまたがる実体の表層は ``text[start:end]``＝原文のトークン間スペースを含むため、
    複数スペースや特殊スペースが混じることがある（HTML 表示では 1 個に潰れて見え、人が打つ
    半角 1 スペースの除外語と完全一致しない）。空白を畳んで取りこぼしを防ぐ。
    """
    return normalize(_WHITESPACE.sub(" ", surface).strip())


class MaskAllowlist:
    """除外語の集合（照合用に正規化済み）。``surface in allowlist`` で判定する。"""

    def __init__(self, surfaces: Iterable[str]) -> None:
        self._norm = {_match_key(s) for s in surfaces if s and s.strip()}

    @classmethod
    def empty(cls) -> MaskAllowlist:
        return cls([])

    @classmethod
    def load(cls, path: str | Path) -> MaskAllowlist:
        """YAML を読み込んで除外リストを作る。ファイルが無ければ空。"""
        p = Path(path)
        if not p.exists():
            return cls.empty()
        return cls(load_allowlist_entries(p))

    def __contains__(self, surface: str) -> bool:
        return _match_key(surface) in self._norm

    def __bool__(self) -> bool:
        return bool(self._norm)

    def __len__(self) -> int:
        return len(self._norm)


def load_allowlist_entries(path: str | Path) -> list[str]:
    """YAML を**構造のまま**読み込む（UI 編集・round-trip 用）。除外語の文字列リストを返す。

    旧 ``"nan"`` 等の空相当は捨てる。重複・空白のみは除く。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = raw.get(_SECTION) or []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s or s.lower() == "nan" or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def sort_key(surface: str) -> str:
    """除外語の並び順キー（正規化＝NFKC+casefold で大小・全角半角を無視した辞書順）。

    照合キー（:func:`_match_key`）と同じ正規化を使う。件数が増えても探しやすいよう、
    保存・表示の双方で同じ順序にするため共有する。
    """
    return _match_key(surface)


def save_allowlist_entries(path: str | Path, surfaces: Iterable[str]) -> None:
    """除外語リストを YAML に書き出す（UI 保存用）。空白除去・重複排除・正規化辞書順にソート。"""
    kept: list[str] = []
    seen: set[str] = set()
    for s in surfaces:
        s = str(s).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        kept.append(s)
    kept.sort(key=sort_key)
    Path(path).write_text(
        yaml.safe_dump({_SECTION: kept}, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )
