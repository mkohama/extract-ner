"""engine 側の LLM 票合流（J2）の単体テスト（GiNZA 不要・純ロジック）。

- ``vote_category`` の ``llm`` 分岐＝ENE type → 6 カテゴリ写像（§7-④）。
- ``_llm_raw``＝LlmDetection の各スパン → ``("llm", ene_type)`` 票。
- ``_cluster`` の確信度＝LLM 単独→中／LLM＋NER 相乗り→強／確定にはしない（§7-②）。
- ``_demote_code_like``＝コードらしき中/弱は微弱へ。ただし LLM 識別子は免除（§7-④）。
"""

from __future__ import annotations

from src.llm.schema import LlmDetection, LlmSpan
from src.masking.engine import (
    Candidate,
    _cluster,
    _demote_code_like,
    _has_llm_identifier_vote,
    _llm_raw,
    vote_category,
)


def test_vote_category_llm_branch() -> None:
    assert vote_category("llm", "Person") == "人名"
    assert vote_category("llm", "Company") == "社名"
    assert vote_category("llm", "Department") == "社名"
    assert vote_category("llm", "City") == "地名"
    assert vote_category("llm", "Email") == "連絡先"
    assert vote_category("llm", "Trademark") == "商標"
    assert vote_category("llm", "Employee_ID") == "その他"
    assert vote_category("llm", "Unknown_Type") is None


def test_llm_raw_makes_llm_votes() -> None:
    text = "田中商事に勤務"
    det = LlmDetection(
        spans=(LlmSpan(0, 2, "Person", "姓", "exact"),),
        not_found=(),
        model="gpt-4.1-mini",
        detector_version="v1",
    )
    raws = _llm_raw(det, text)
    assert len(raws) == 1
    c = raws[0]
    assert (c.start, c.end, c.surface) == (0, 2, "田中")
    assert c.votes == (("llm", "Person"),)
    assert c.category == "人名"  # 写像値（保険。最終カテゴリは tally が決める）


def test_llm_alone_is_chu_not_auto_mask() -> None:
    """LLM 単独（人名）は中（1 チャネル）。確定にも強にもならない。"""
    text = "田中商事"
    cands = _llm_raw(
        LlmDetection(
            spans=(LlmSpan(0, 2, "Person", None, "exact"),),
            not_found=(),
            model="m",
            detector_version="v",
        ),
        text,
    )
    [c] = _cluster(text, cands)
    assert c.category == "人名"
    assert c.confidence == "中"


def test_llm_plus_ner_is_strong() -> None:
    """LLM ＋ NER1 モデルが同じ人名に投票＝2 チャネル＝強（自動マスク）。"""
    text = "田中商事"
    cands = [
        Candidate(0, 2, "田中", "人名", "", (("llm", "Person"),)),
        Candidate(0, 2, "田中", "人名", "", (("ja_ginza_electra", "PERSON"),)),
    ]
    [c] = _cluster(text, cands)
    assert c.category == "人名"
    assert c.confidence == "強"


def test_llm_identifier_exempt_from_weak_demotion() -> None:
    """LLM 識別子（Employee_ID 等）はコードらしくても微弱に落とさない（弱のまま＝レビュー可視）。"""
    # その他＝弱。surface "7-410" はコードらしい（名前文字なし）。
    weak_id = Candidate(0, 5, "7-410", "その他", "弱", (("llm", "Employee_ID"),))
    assert _has_llm_identifier_vote(weak_id)
    [out] = _demote_code_like([weak_id])
    assert out.confidence == "弱"  # 免除＝微弱に落ちない


def test_non_llm_code_like_is_demoted() -> None:
    """LLM 識別子票が無いコードらしき中/弱は従来どおり微弱へ。"""
    weak_code = Candidate(
        0, 5, "7-410", "その他", "弱", (("sudachi", "名詞-固有名詞-一般"),)
    )
    assert not _has_llm_identifier_vote(weak_code)
    [out] = _demote_code_like([weak_code])
    assert out.confidence == "微弱"


# --- 2系統合議（A1）: NER の内部チャネル数は確信度に効かない（NERとLLMは対等な2系統） --- #


def test_ner_only_multichannel_is_chu_not_strong() -> None:
    """NER 系統だけなら、2モデル＋Sudachi が同じ人名でも **中**（小坂のケース）。

    旧フラット集計では「2チャネル＝強」だったが、2系統合議では NER は何チャネル一致しても
    1 系統＝中。LLM が併走して初めて 2系統＝強になる（§13）。
    """
    text = "小坂"
    cands = [
        Candidate(0, 2, "小坂", "人名", "", (("ja_ginza", "PERSON"),)),
        Candidate(0, 2, "小坂", "人名", "", (("ja_ginza_electra", "PERSON"),)),
        Candidate(0, 2, "小坂", "人名", "", (("sudachi", "名詞-固有名詞-人名-姓"),)),
    ]
    [c] = _cluster(text, cands)
    assert c.category == "人名"
    assert c.confidence == "中"


def test_llm_person_beats_ner_chimei_rescued_to_chu() -> None:
    """小浜：NER=地名（City＋Sudachi 地名）/ LLM=人名 → **人名・中**。

    特別（人名）が地名に勝ち、地名・弱に埋もれさせない＝LLM の人名票の死票回収。
    人名と言ったのは LLM 1系統のみなので確信度は中。
    """
    text = "小浜"
    cands = [
        Candidate(0, 2, "小浜", "地名", "", (("ja_ginza_electra", "CITY"),)),
        Candidate(0, 2, "小浜", "地名", "", (("sudachi", "名詞-固有名詞-地名-一般"),)),
        Candidate(0, 2, "小浜", "人名", "", (("llm", "Person"),)),
    ]
    [c] = _cluster(text, cands)
    assert c.category == "人名"
    assert c.confidence == "中"


def test_two_systems_different_special_is_strong() -> None:
    """2系統がともに特別（NER=社名 / LLM=人名）→ **強**。種別が違ってもよい（重い人名を採用）。"""
    text = "アクメ"
    cands = [
        Candidate(0, 3, "アクメ", "社名", "", (("ja_ginza_electra", "COMPANY"),)),
        Candidate(0, 3, "アクメ", "人名", "", (("llm", "Person"),)),
    ]
    [c] = _cluster(text, cands)
    assert c.confidence == "強"
    assert c.category == "人名"  # 人名 > 社名（_CAT_PRIORITY）


def test_both_systems_chimei_is_weak() -> None:
    """長崎：両系統とも地名（特別なし）→ **弱**。"""
    text = "長崎"
    cands = [
        Candidate(0, 2, "長崎", "地名", "", (("ja_ginza_electra", "CITY"),)),
        Candidate(0, 2, "長崎", "地名", "", (("llm", "City"),)),
    ]
    [c] = _cluster(text, cands)
    assert c.category == "地名"
    assert c.confidence == "弱"
