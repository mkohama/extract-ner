"""除外リスト（allowlist）。マスク候補から恒久的に外す語の名簿。

マスク辞書（[dictionary.py]）の対。NER の誤検出（社内コード・変数名・汎用語など、人名/社名でない語）を
人が「これは機密でない」と判断したら登録し、以後**どの文書でも**候補を「除外」に落とす（1 文書で
登録→他文書でも効く）。

recall 安全のための重要な制約（適用は :func:`MaskingEngine.analyze` 側）:
- 除外できるのは**検出由来（強/中/弱/微弱）のみ**。**辞書一致・連絡先 regex（＝確定）は上書きしない**
  （名簿や決定的検出を誤って外して漏らさないため）。＝確定 ＞ 除外。

照合は既定で**正規化文字列の完全一致**（:func:`dictionary.normalize` と同じ NFKC+casefold）。
``embed: true`` を付けた語は、辞書の ``embed`` と**対称**に、複合語の**境界で内包照合**する
（例 ``FB`` → ``GetFBData`` の ``FB``、``補正`` → ``用補正値`` の ``補正``）。境界照合なので
``FBI`` の ``FB``（より長い連続の一部）や ``補正`` の一部 ``補`` は拾わない。命中した候補は
**丸ごと除外**する（辞書 embed は部分マスクだが、除外は候補単位＝スパンは割らない）。

**境界の供給源＝2 段**（:meth:`_contains_embedded_tokens`）:
- **トークン間＝形態素境界**（SudachiPy トークン。``補正`` → ``用/補正/値``）＝日本語対応。
  ``apply_allowlist`` が候補を覆うトークン列を渡したときだけ効く。
- **トークン内＝``_split_identifier`` のサブワード境界**（camelCase・区切り。``GetFBData`` → ``Get/FB/Data``）
  ＝ASCII/識別子系。トークン無しでも表層文字列から効く（後方互換）。

英語（ASCII）は綴りから境界を導けるので昔から効いていたが、日本語は綴り上の境界が無く形態素解析が要る
＝これが日本語 embed が効かなかった理由。実装は英語版に**境界供給源としてトークナイザを足した**だけ。

YAML 形式（``data/mask_allowlist.yaml``）::

    除外:
      - Em_NoYes                 # 文字列だけ＝完全一致で除外
      - Reject
      - surface: FB              # embed:true＝FB を含む複合語（GetFBData 等）も丸ごと除外
        embed: true
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from src.masking.dictionary import MAX_MATCH_TOKENS, _split_identifier, normalize

# YAML のセクション名（除外語のリスト。文字列 or {surface, embed} を混在可）
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
    """除外語の集合。既定は**完全一致**（:meth:`matches`）。``embed`` 語はサブワード内包も照合する。"""

    def __init__(self, entries: Iterable[str | dict]) -> None:
        self._norm: set[str] = set()  # 完全一致キー（全エントリ）
        self._embed: set[str] = set()  # 内包照合キー（embed:true のみ）
        for e in entries:
            if isinstance(e, dict):
                s = str(e.get("surface") or "").strip()
                embed = bool(e.get("embed"))
            else:
                s = str(e).strip()
                embed = False
            if not s:
                continue
            self._norm.add(_match_key(s))
            if embed:
                self._embed.add(normalize(s))

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

    def matches(
        self, surface: str, token_surfaces: Sequence[str] | None = None
    ) -> bool:
        """除外対象か：**完全一致** or （``embed`` 語の）**境界内包一致**。

        ``token_surfaces``（候補を覆う SudachiPy トークンの表層列）を渡すと、embed 照合が
        **形態素境界**でも効く（日本語対応。``補正`` embed → ``用補正値``）。渡さないと従来どおり
        ``surface`` 文字列を ``_split_identifier`` で割る ASCII/識別子系だけ（後方互換）。
        """
        if _match_key(surface) in self._norm:
            return True
        if token_surfaces:
            return self._contains_embedded_tokens(token_surfaces)
        return self._contains_embedded(surface)

    def __contains__(self, surface: str) -> bool:
        """完全一致のみ（後方互換）。内包も含めた判定は :meth:`matches`。"""
        return _match_key(surface) in self._norm

    def _embed_join_hit(self, norm_units: Sequence[str]) -> bool:
        """正規化済みの単位列で、**連続する部分列の連結**が ``embed`` キーに一致するか。

        単位はトークン（形態素境界）でもサブワード（``_split_identifier`` の camelCase 境界）でも良い。
        境界照合＝単位をまたぐ連結のみ見るので、単位の**途中**では一致しない（``FB`` は 1 単位
        ``FBI`` を割れない・``補`` は 1 形態素 ``補正`` を割れない）。
        """
        n = len(norm_units)
        for i in range(n):
            for length in range(min(MAX_MATCH_TOKENS, n - i), 0, -1):
                if "".join(norm_units[i : i + length]) in self._embed:
                    return True
        return False

    def _contains_embedded(self, surface: str) -> bool:
        """surface が ``embed`` 語を**サブワード境界**で内包するか（トークン無しの後方互換経路）。

        ``_split_identifier`` で ASCII/識別子をサブワードに割り、連続サブワードの正規化連結が
        ``embed`` キーに一致すれば True（``GetFBData`` の ``FB``）。``FBI`` は 1 サブワード＝
        ``FB`` 単独では一致しない（境界照合）。日本語は割れないので **要 token_surfaces**。
        """
        if not self._embed:
            return False
        subs = _split_identifier(surface)
        return self._embed_join_hit([normalize(surface[a:b]) for a, b in subs])

    def _contains_embedded_tokens(self, token_surfaces: Sequence[str]) -> bool:
        """候補を覆うトークン列が ``embed`` 語を**境界**で内包するか（辞書 embed の実質的な対称）。

        2 段の境界を見る：**トークン間＝形態素境界**（``補正`` → ``用/補正/値`` の ``補正``。日本語対応）と、
        **トークン内＝``_split_identifier`` のサブワード境界**（``GetFBData`` の ``FB``。英 camelCase 維持）。
        どちらも単位の途中では一致しない（境界照合）。英語で表層文字列に対してやっていたことの、
        供給源にトークナイザ（形態素境界）を足した日本語版。
        """
        if not self._embed:
            return False
        # トークン間（形態素境界）＝日本語
        if self._embed_join_hit([normalize(t) for t in token_surfaces]):
            return True
        # 各トークン内のサブワード境界＝英 camelCase（GetFBData 等の融合トークン）
        return any(
            self._embed_join_hit([normalize(t[a:b]) for a, b in _split_identifier(t)])
            for t in token_surfaces
        )

    def __bool__(self) -> bool:
        return bool(self._norm)

    def __len__(self) -> int:
        return len(self._norm)


def load_allowlist_entries(path: str | Path) -> list[dict]:
    """YAML を**構造のまま**読み込む（UI 編集・round-trip 用）。

    返り値は ``{"surface": str, "embed": bool}`` の列。文字列だけの項目は ``embed=False``。
    旧 ``"nan"`` 等の空相当は捨てる。重複（surface 単位）・空白のみは除く。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = raw.get(_SECTION) or []
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            s = str(item.get("surface") or "").strip()
            embed = bool(item.get("embed"))
        else:
            s = str(item).strip()
            embed = False
        if not s or s.lower() == "nan" or s in seen:
            continue
        seen.add(s)
        out.append({"surface": s, "embed": embed})
    return out


def sort_key(surface: str) -> str:
    """除外語の並び順キー（正規化＝NFKC+casefold で大小・全角半角を無視した辞書順）。

    照合キー（:func:`_match_key`）と同じ正規化を使う。件数が増えても探しやすいよう、
    保存・表示の双方で同じ順序にするため共有する。
    """
    return _match_key(surface)


def save_allowlist_entries(path: str | Path, entries: Iterable[str | dict]) -> None:
    """除外語リストを YAML に書き出す（UI 保存用）。空白除去・重複排除・正規化辞書順にソート。

    ``embed`` が無ければ**文字列だけ**の簡潔形、``embed:true`` なら ``{surface, embed}`` 形で書く
    （辞書 :func:`dictionary.save_entries` と対称）。``entries`` は文字列 or ``{surface, embed}`` を混在可。
    """
    kept: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for e in entries:
        if isinstance(e, dict):
            s = str(e.get("surface") or "").strip()
            embed = bool(e.get("embed"))
        else:
            s = str(e).strip()
            embed = False
        if not s or s in seen:
            continue
        seen.add(s)
        kept.append((s, embed))
    kept.sort(key=lambda t: sort_key(t[0]))
    items: list[dict | str] = [
        {"surface": s, "embed": True} if embed else s for s, embed in kept
    ]
    Path(path).write_text(
        yaml.safe_dump(
            {_SECTION: items}, allow_unicode=True, sort_keys=False, indent=2
        ),
        encoding="utf-8",
    )
