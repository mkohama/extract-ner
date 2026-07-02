# -*- coding: utf-8 -*-
"""除外リスト（allowlist）の単体テスト。

完全一致 ＋ ``embed`` のサブワード境界内包照合（辞書 embed と対称）、YAML round-trip、
:func:`apply_allowlist` の確定ガード（辞書票は除外しない）を固定する。GiNZA 非依存。
"""

from __future__ import annotations

from pathlib import Path

from src.masking.allowlist import (
    MaskAllowlist,
    load_allowlist_entries,
    save_allowlist_entries,
)
from src.masking.engine import Candidate, apply_allowlist
from src.ner.engine import AnalyzedToken


def _al() -> MaskAllowlist:
    return MaskAllowlist(
        [
            "Em_NoYes",
            {"surface": "FB", "embed": True},
            {"surface": "NSR", "embed": True},
        ]
    )


def test_exact_match() -> None:
    al = _al()
    assert al.matches("Em_NoYes")  # 完全一致
    assert al.matches("ｅｍ＿ｎｏｙｅｓ")  # NFKC+casefold で全角も一致
    assert not al.matches("Reject")  # 未登録


def test_embed_matches_at_subword_boundary() -> None:
    al = _al()
    # camelCase / 区切り / 日本語隣接のサブワード境界で内包
    assert al.matches("GetFBData")
    assert al.matches("FBData")
    assert al.matches("FB_DATA")
    assert al.matches("NSR用補正ファイル")
    assert al.matches("NSR Wafer Alignment")


def test_embed_is_boundary_safe() -> None:
    al = _al()
    # より長い大文字連続の一部（FBI）や、camel のコブでちぎれる（getFBdata）は拾わない
    assert not al.matches("FBI")
    assert not al.matches("GetFBIData")
    assert not al.matches("getFBdata")


def test_embed_matches_at_morpheme_boundary() -> None:
    """日本語 embed はトークン（形態素）境界で内包一致（`補正` → `用/補正/値`）。tokens 必須。"""
    al = MaskAllowlist([{"surface": "補正", "embed": True}])
    toks = ["用", "補正", "値"]
    assert al.matches("用補正値", toks)  # 形態素境界で内包
    assert not al.matches("用補正値")  # tokens 無しは日本語を割れない＝一致しない
    # 形態素の連結（`用補正` = 用+補正）も拾う
    assert MaskAllowlist([{"surface": "用補正", "embed": True}]).matches(
        "用補正値", toks
    )


def test_embed_japanese_is_boundary_safe() -> None:
    """形態素の**途中**（`補` は 1 形態素 `補正` を割れない）は拾わない＝境界照合。"""
    al = MaskAllowlist([{"surface": "補", "embed": True}])
    assert not al.matches("用補正値", ["用", "補正", "値"])


def test_contains_is_exact_only() -> None:
    """``in`` は完全一致のみ（後方互換）。内包は matches が担う。"""
    al = _al()
    assert "FB" in al
    assert "GetFBData" not in al  # __contains__ は内包しない
    assert al.matches("GetFBData")  # matches は内包する


def test_yaml_roundtrip_and_sort(tmp_path: Path) -> None:
    p = tmp_path / "al.yaml"
    save_allowlist_entries(
        p, [{"surface": "NSR", "embed": True}, "Zebra", "apple", "apple"]
    )
    entries = load_allowlist_entries(p)
    # 重複排除・正規化辞書順（apple < Zebra < NSR? 正規化順）＋ embed 保持
    by = {e["surface"]: e["embed"] for e in entries}
    assert by == {"NSR": True, "Zebra": False, "apple": False}
    # embed は {surface, embed} 形、非 embed は文字列で書かれる
    text = p.read_text(encoding="utf-8")
    assert "surface: NSR" in text and "embed: true" in text
    assert "- apple" in text  # 文字列だけの簡潔形


def test_plain_string_entries_load_as_non_embed(tmp_path: Path) -> None:
    p = tmp_path / "al.yaml"
    p.write_text("除外:\n  - Em_NoYes\n  - Reject\n", encoding="utf-8")
    entries = load_allowlist_entries(p)
    assert all(e["embed"] is False for e in entries)
    assert {e["surface"] for e in entries} == {"Em_NoYes", "Reject"}


def _cand(surface: str, votes: tuple) -> Candidate:
    return Candidate(0, len(surface), surface, "商標", "中", votes)


def test_apply_allowlist_embed_excludes_and_guards_dict() -> None:
    al = MaskAllowlist([{"surface": "FB", "embed": True}])
    detected = _cand("GetFBData", (("llm", "Trademark"),))  # 検出由来
    dict_backed = _cand("GetFBData", (("dict", "商標(辞書)"),))  # 辞書票＝確定ガード
    out = apply_allowlist([detected, dict_backed], al)
    assert out[0].confidence == "除外"  # 検出由来の内包一致→除外
    assert out[1].confidence == "中"  # 辞書票は上書きしない（確定＞除外）


def test_apply_allowlist_japanese_embed_needs_tokens() -> None:
    """日本語 embed は `apply_allowlist` に tokens を渡したときだけ形態素境界で効く。"""
    al = MaskAllowlist([{"surface": "補正", "embed": True}])
    cand = Candidate(0, 4, "用補正値", "商標", "中", (("llm", "Trademark"),))
    # 用[0,1] 補正[1,3] 値[3,4]（tag/pos はダミー）
    toks = (
        AnalyzedToken(0, 1, "用", "", ""),
        AnalyzedToken(1, 3, "補正", "", ""),
        AnalyzedToken(3, 4, "値", "", ""),
    )
    assert apply_allowlist([cand], al, toks)[0].confidence == "除外"  # 形態素境界で除外
    assert apply_allowlist([cand], al)[0].confidence == "中"  # tokens 無しは効かない
