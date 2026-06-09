"""マスキング検出エンジン（UI 非依存）。

候補生成（NER 両モデル ∪ Sudachi 品詞 ∪ マスク辞書）→ 確信度づけ → 選択された候補で
マスク適用（文書内収集で全出現に展開）。設計の背景は docs-dev/対策仮説.md を参照。

確信度（カテゴリごとの「票数」で決める。票＝そのカテゴリへ投票した別チャネル数。
チャネル＝辞書/Sudachi/NER(ja_ginza)/NER(electra)。同一チャネルの複数形態素は 1 票）：
- カテゴリ … by-cat 最多票のカテゴリ（同票のみ _CAT_PRIORITY でタイブレーク）。辞書一致は無条件で確定。
- 確定 … 辞書一致
- 強  … 解決カテゴリへ 2 チャネル以上（人名/社名/商標/連絡先）
- 中  … 解決カテゴリへ 1 チャネルのみ（同上）
- 弱  … カテゴリが 地名/その他（票数問わず。誤分類で人物が紛れるので必ずレビュー）
- 折衷 … Sudachi 固有名詞-一般(その他) は、同スパンに NER の社名票があるとき社名票として数える
         （Sony 等の有名企業名を NER 単独でも強で拾う。長崎=地名は sudachi が地名なので無関係＝過信は起きない）。

使い方：
    analysis = engine.analyze(chunks)          # 全候補（確信度・各票の判定つき）
    selected = [c for c in analysis.candidates if c.confidence in ("確定", "強")]
    result = engine.apply(analysis, selected)  # 選んだ候補でマスク（全出現に展開）
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.masking.dictionary import MAX_MATCH_TOKENS, MaskDictionary, normalize
from src.ner import AVAILABLE_MODELS, AnalyzedToken, NerEngine

# クラスタ代表カテゴリの選択優先度（地名・その他は低い）
_CAT_PRIORITY = ["人名", "社名", "商標", "連絡先", "地名", "その他"]
# 確信度の強さ順（集約時に最良を選ぶ）
_CONF_RANK = {"確定": 3, "強": 2, "中": 1, "弱": 0}
# 自動マスク（初期チェック ON）にする確信度
AUTO_MASK_CONFIDENCE = ("確定", "強")
# プレースホルダ接頭辞
_PLACEHOLDER_PREFIX = {
    "人名": "人物",
    "社名": "社",
    "商標": "商標",
    "地名": "地名",
    "連絡先": "連絡先",
    "その他": "語",
}

# NER ラベル（大文字）→ カテゴリ。Product_Other 系（その他）は**ノイズ過多のため候補にしない**。
# 注: 「N_*」（N_Person=人数 等）は関根体系では**数値表現（個数）**であって固有名詞ではない。
# 人名・地名と取り違えないよう **含めない**（PERSON と N_PERSON は別物）。
_NER_LABEL_CATEGORY: dict[str, str] = {
    "PERSON": "人名",
    "COMPANY": "社名",
    "COMPANY_GROUP": "社名",
    "CORPORATION_OTHER": "社名",
    "SHOW_ORGANIZATION": "社名",
    "INTERNATIONAL_ORGANIZATION": "社名",
    "POLITICAL_ORGANIZATION_OTHER": "社名",
    "GOVERNMENT": "社名",
    "POLITICAL_PARTY": "社名",
    "SPORTS_ORGANIZATION_OTHER": "社名",
    # Nationality(国籍)・Ethnic_Group_Other(民族)・Family(一族) は会社/組織ではないので含めない。
    "CITY": "地名",
    "PROVINCE": "地名",
    "COUNTRY": "地名",
    "GPE_OTHER": "地名",
    "ADDRESS": "地名",
    "POSTAL_ADDRESS": "地名",
    "GEOLOGICAL_REGION_OTHER": "地名",
    "LOCATION_OTHER": "地名",
    "DOMESTIC_REGION_OTHER": "地名",
    "CONTINENTAL_REGION_OTHER": "地名",
    "FACILITY_OTHER": "地名",
    "FACILITY_PART": "地名",
    "STATION": "地名",
    "AIRPORT": "地名",
    "PHONE_NUMBER": "連絡先",
    "EMAIL": "連絡先",
    "URL": "連絡先",
}


@dataclass(frozen=True)
class Candidate:
    """マスク候補スパン（全文オフセット）。"""

    start: int
    end: int
    surface: str
    category: str  # 人名 / 社名 / 商標 / 地名 / 連絡先 / その他
    confidence: str  # 確定 / 強 / 中 / 弱
    # 各票の判定 (channel, label)。channel = dict / sudachi / <model_name>
    votes: tuple[tuple[str, str], ...]

    def vote_label(self, channel: str) -> str:
        """指定チャネルの判定ラベルを返す（無ければ空文字）。"""
        for ch, label in self.votes:
            if ch == channel:
                return label
        return ""


@dataclass(frozen=True)
class CandidateGroup:
    """同一実体（表層）の候補をまとめたもの。マスクは実体ごとに行うため UI/CLI はこれを使う。

    マスクは「出現ごと」でなく「実体ごと」（一か所選べば文書内の全出現に展開される）。
    """

    surface: str  # 代表表記（辞書 canonical があればそれ、なければ表層）
    category: str
    confidence: str  # 出現の中で最良の確信度
    votes: tuple[tuple[str, str], ...]  # 全出現でついた票の和集合
    members: tuple[Candidate, ...]  # この実体の全出現

    @property
    def count(self) -> int:
        return len(self.members)

    def vote_label(self, channel: str) -> str:
        for ch, label in self.votes:
            if ch == channel:
                return label
        return ""


@dataclass(frozen=True)
class MaskEntry:
    """プレースホルダ 1 件（復元用の対応表）。"""

    placeholder: str
    category: str
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class MaskAnalysis:
    """解析結果（マスクはまだ適用していない。候補を選ぶ前段）。

    ``text``/``tokens``/``candidates`` は検出に使う**平坦化後テキスト基準**。
    マスクは ``original_text``（平坦化前の `|` 入り原文）へ ``offset_map`` で写して当てる。
    平坦化しない場合は ``original_text == text``・``offset_map`` は恒等写像。
    """

    text: str
    tokens: tuple[AnalyzedToken, ...]
    candidates: tuple[Candidate, ...]
    original_text: str = ""
    offset_map: tuple[int, ...] = ()


@dataclass(frozen=True)
class MaskResult:
    """選んだ候補でマスクを適用した結果。"""

    text: str  # 元テキスト（チャンク連結後）
    masked_text: str  # マスク済みテキスト
    masked: tuple[Candidate, ...]  # 実際にマスクしたスパン（選択＋文書内展開）
    mapping: tuple[MaskEntry, ...]  # プレースホルダ ↔ 原語


def _sudachi_category(tag: str) -> str | None:
    """SudachiPy 品詞 → カテゴリ。対象外なら None。"""
    if tag.startswith("名詞-固有名詞-人名"):  # 姓/名/一般
        return "人名"
    if tag.startswith("名詞-固有名詞-地名"):
        return "地名"
    if tag.startswith("名詞-固有名詞-一般"):
        return "その他"
    return None


def vote_category(channel: str, label: str) -> str | None:
    """監査用：1 票 (channel, label) がどのカテゴリに投票したかを引く。

    候補生成（analyze）でカテゴリを決めるのと同じ対応関係を使う（重複ロジックを作らない）。
    辞書票のラベルは ``"社名(辞書)"`` 形式なので接頭のカテゴリを取り出す。
    """
    if channel == "dict":
        return label.split("(", 1)[0] or None
    if channel == "sudachi":
        return _sudachi_category(label)
    return _NER_LABEL_CATEGORY.get(label.upper())


def tally_votes(votes: Iterable[tuple[str, str]]) -> Counter[str]:
    """票集合 → カテゴリ別の「チャネル数」。確信度・カテゴリ決定はこの集計で行う。

    票＝チャネル（同一チャネルの複数形態素＝姓+名 等は 1 票に畳む）。
    折衷ルール：Sudachi 固有名詞-一般(その他) は、同スパンに NER の社名票があるとき
    社名票として数える（Sony 等の有名企業名を NER 単独でも拾う。長崎=地名は sudachi が
    地名なので無関係＝過信にはならない）。
    """
    votes = list(votes)
    has_ner_company = any(
        ch not in ("dict", "sudachi") and vote_category(ch, label) == "社名"
        for ch, label in votes
    )
    channels_by_cat: dict[str, set[str]] = defaultdict(set)
    for ch, label in votes:
        cat = vote_category(ch, label)
        if cat is None:
            continue
        if cat == "その他" and ch == "sudachi" and has_ner_company:
            cat = "社名"
        channels_by_cat[cat].add(ch)
    return Counter({cat: len(chs) for cat, chs in channels_by_cat.items()})


def _top_category(counts: Counter[str]) -> str:
    """カテゴリ別票数から代表カテゴリを選ぶ（最多票。同票のみ _CAT_PRIORITY でタイブレーク）。"""
    best = max(counts.values())
    tied = [c for c, n in counts.items() if n == best]
    return next((c for c in _CAT_PRIORITY if c in tied), tied[0])


class MaskingEngine:
    """マスキング検出エンジン。NER 両モデル ∪ Sudachi 品詞 ∪ マスク辞書で候補を作る。"""

    def __init__(
        self,
        dictionary: MaskDictionary | None = None,
        models: Sequence[str] | None = None,
    ) -> None:
        self.dictionary = dictionary or MaskDictionary.empty()
        self.engines = [NerEngine(m) for m in (models or AVAILABLE_MODELS)]

    def analyze(
        self, chunks: Iterable[str], *, flatten_tables: bool = False
    ) -> MaskAnalysis:
        """全候補（確信度・各票の判定つき）を作る。マスクはまだ適用しない。"""
        chunks = list(chunks)
        per_model = [
            (e.model_name, e.analyze_chunks(chunks, flatten_tables=flatten_tables))
            for e in self.engines
        ]
        base = per_model[0][1]
        text = base.text
        tokens = base.tokens

        raw: list[Candidate] = []

        # ① マスク辞書（dict 票）
        surfaces = [t.surface for t in tokens]
        for m in self.dictionary.match(surfaces):
            start = tokens[m.start_token].start
            end = tokens[m.end_token - 1].end
            raw.append(
                _raw(start, end, text, m.category, ("dict", f"{m.category}(辞書)"))
            )

        # ② Sudachi 品詞（sudachi 票）
        for t in tokens:
            category = _sudachi_category(t.tag)
            if category is not None:
                raw.append(_raw(t.start, t.end, text, category, ("sudachi", t.tag)))

        # ③ NER エンティティ（モデルごとに別票）。
        #    Product_Other 系（その他）は変数名・コード・融合ジャーゴンが多くノイズ過多のため**除外**。
        #    → NER は 人名/社名/地名/連絡先 のラベルのみ候補化。埋没社名・商標は辞書で拾う。
        for model_name, analysis in per_model:
            for ent in analysis.entities:
                category = _NER_LABEL_CATEGORY.get(ent.label.upper())
                if category is None:
                    continue
                raw.append(
                    _raw(ent.start, ent.end, text, category, (model_name, ent.label))
                )

        clusters = _cluster(text, raw)
        return MaskAnalysis(
            text=text,
            tokens=tokens,
            candidates=tuple(clusters),
            original_text=base.original_text or text,
            offset_map=base.offset_map or tuple(range(len(text))),
        )

    def apply(
        self,
        analysis: MaskAnalysis,
        selected: Iterable[Candidate],
        *,
        expand: bool = True,
    ) -> MaskResult:
        """選択された候補でマスクを適用する。

        expand=True（既定）: 実体ごと。選んだ表層を文書内の**全出現**に展開してマスク。
        expand=False: 出現ごと。選んだスパン**だけ**をマスク（同形異義語の個別制御用）。

        検出・展開は平坦化テキスト座標で行い、最後に :func:`_to_original_spans` で
        原文座標へ写してから原文を置換する（`|` 入りの原文を保ったままマスクする）。
        """
        selected = list(selected)
        if expand:
            collected: dict[str, str] = {}
            for c in selected:
                collected.setdefault(normalize(c.surface), c.category)
            spans = selected + _expand(
                analysis.text, analysis.tokens, collected, selected
            )
        else:
            spans = selected

        original_text = analysis.original_text or analysis.text
        offset_map = analysis.offset_map or tuple(range(len(analysis.text)))
        orig_spans = _to_original_spans(spans, original_text, offset_map)

        mapping, span_placeholder = _assign_placeholders(orig_spans, self.dictionary)
        masked_text = _apply_mask(original_text, orig_spans, span_placeholder)
        return MaskResult(
            text=original_text,
            masked_text=masked_text,
            masked=tuple(orig_spans),
            mapping=mapping,
        )

    def group_candidates(
        self, candidates: Iterable[Candidate]
    ) -> list[CandidateGroup]:
        """候補を実体（カテゴリ×代表表記）ごとにまとめる。

        出現ごとの候補を 1 実体 1 行にする（confidence は最良、votes は和集合）。
        マスクは実体ごとなので、UI/CLI はこの単位で「選ぶ/見せる」のが正しい。
        """
        groups: dict[tuple[str, str], list[Candidate]] = {}
        order: list[tuple[str, str]] = []
        for c in candidates:
            canonical = self.dictionary.canonical_of(c.surface)
            key = (c.category, normalize(canonical) if canonical else normalize(c.surface))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(c)

        result: list[CandidateGroup] = []
        for key in order:
            members = groups[key]
            best = max(members, key=lambda m: _CONF_RANK.get(m.confidence, 0))
            votes = tuple(dict.fromkeys(v for m in members for v in m.votes))
            canonical = self.dictionary.canonical_of(best.surface)
            result.append(
                CandidateGroup(
                    surface=canonical or best.surface,
                    category=best.category,
                    confidence=best.confidence,
                    votes=votes,
                    members=tuple(members),
                )
            )
        return result

    def mask_chunks(
        self, chunks: Iterable[str], *, flatten_tables: bool = False
    ) -> tuple[MaskAnalysis, MaskResult]:
        """解析＋既定選択（確定/強）でのマスク適用をまとめて行う。"""
        analysis = self.analyze(chunks, flatten_tables=flatten_tables)
        selected = [
            c for c in analysis.candidates if c.confidence in AUTO_MASK_CONFIDENCE
        ]
        return analysis, self.apply(analysis, selected)


# --------------------------------------------------------------------------- #
# ヘルパー
# --------------------------------------------------------------------------- #
def _raw(
    start: int, end: int, text: str, category: str, vote: tuple[str, str]
) -> Candidate:
    """確信度未確定の生候補（confidence はクラスタ時に決める）。"""
    return Candidate(start, end, text[start:end], category, "", (vote,))


def _confidence(category: str, counts: Counter[str]) -> str:
    """解決カテゴリと「カテゴリ別票数」から確信度を決める（辞書一致は _merge 側で確定扱い）。

    強/中は **解決カテゴリへ投票したチャネル数**で決める（他カテゴリの票では水増ししない）。
    地名/その他は誤分類で人物が紛れるため票数によらず弱（必ずレビュー）。
    """
    if category in ("地名", "その他"):
        return "弱"
    return "強" if counts.get(category, 0) >= 2 else "中"


def _cluster(text: str, cands: list[Candidate]) -> list[Candidate]:
    """重なる候補スパンを 1 つにまとめ、票数とカテゴリから確信度を決める。"""
    if not cands:
        return []
    ordered = sorted(cands, key=lambda c: (c.start, c.end))
    clusters: list[Candidate] = []
    start, end = ordered[0].start, ordered[0].end
    members = [ordered[0]]
    for c in ordered[1:]:
        if c.start < end:
            end = max(end, c.end)
            members.append(c)
        else:
            clusters.append(_merge(text, start, end, members))
            start, end, members = c.start, c.end, [c]
    clusters.append(_merge(text, start, end, members))
    return clusters


def _merge(text: str, start: int, end: int, members: list[Candidate]) -> Candidate:
    votes = tuple(dict.fromkeys(v for m in members for v in m.votes))
    # 辞書票があれば辞書のカテゴリで確定（NER/Sudachi の票数によらず最優先）。
    dict_cats = [
        cat
        for ch, label in votes
        if ch == "dict" and (cat := vote_category(ch, label)) is not None
    ]
    if dict_cats:
        return Candidate(start, end, text[start:end], dict_cats[0], "確定", votes)
    # 票が無ければ（理論上稀）カテゴリ未確定の中扱いで残す。
    counts = tally_votes(votes)
    if not counts:
        return Candidate(start, end, text[start:end], members[0].category, "中", votes)
    category = _top_category(counts)
    confidence = _confidence(category, counts)
    return Candidate(start, end, text[start:end], category, confidence, votes)


def _expand(
    text: str,
    tokens: tuple[AnalyzedToken, ...],
    collected: dict[str, str],
    existing: list[Candidate],
) -> list[Candidate]:
    """選んだ表層形を、文書内の全出現（トークン単位）に展開する。"""
    if not collected:
        return []
    norm = [normalize(t.surface) for t in tokens]
    ranges = [(c.start, c.end) for c in existing]
    out: list[Candidate] = []
    i = 0
    n = len(tokens)
    while i < n:
        hit: tuple[int, int, str] | None = None
        for length in range(min(MAX_MATCH_TOKENS, n - i), 0, -1):
            key = "".join(norm[i : i + length])
            if key in collected:
                hit = (i, i + length, collected[key])
                break
        if hit is not None:
            s, e_tok, category = hit
            start, end = tokens[s].start, tokens[e_tok - 1].end
            if not any(start < er and st < end for st, er in ranges):
                out.append(
                    Candidate(
                        start, end, text[start:end], category, "確定", (("collected", "展開"),)
                    )
                )
                ranges.append((start, end))
            i = e_tok
        else:
            i += 1
    return out


def _map_span(offset_map: tuple[int, ...], start: int, end: int) -> tuple[int, int] | None:
    """平坦化テキストのスパン [start, end) を原文座標へ写す。

    挿入文字（区切り `、`/句点 `。`/連結改行＝対応 -1）を読み飛ばし、原文側で実体を
    覆う連続範囲を返す。スパン内に実体が無ければ None。検出は `、` でセルを割っているため
    1 つのスパンはセル内に収まり、原文範囲に `|` は挟まらない。
    """
    n = min(end, len(offset_map))
    o_start = next((offset_map[i] for i in range(start, n) if offset_map[i] != -1), None)
    if o_start is None:
        return None
    o_end = next(
        (offset_map[i] + 1 for i in range(n - 1, start - 1, -1) if offset_map[i] != -1),
        None,
    )
    return (o_start, o_end if o_end is not None else o_start)


def _to_original_spans(
    spans: list[Candidate], original_text: str, offset_map: tuple[int, ...]
) -> list[Candidate]:
    """候補スパン（平坦化座標）を原文座標へ写し、原文の表層で作り直す。

    平坦化しない場合は offset_map が恒等なので原文 = 平坦化テキストで実質そのまま。
    """
    out: list[Candidate] = []
    for c in spans:
        mapped = _map_span(offset_map, c.start, c.end)
        if mapped is None:
            continue
        s, e = mapped
        out.append(
            Candidate(s, e, original_text[s:e], c.category, c.confidence, c.votes)
        )
    return out


def _assign_placeholders(
    spans: list[Candidate], dictionary: MaskDictionary
) -> tuple[tuple[MaskEntry, ...], dict[tuple[int, int], str]]:
    """(カテゴリ, 代表表記) ごとにプレースホルダを割り当てる。

    辞書にある語は canonical（代表表記）で束ね、表記ゆれ（英語表記↔カタカナ表記・略称・旧称）を
    同じプレースホルダに統一する。辞書外は正規化表層で束ねる。
    """
    groups: dict[tuple[str, str], list[Candidate]] = {}
    order: list[tuple[str, str]] = []
    for sp in spans:
        canonical = dictionary.canonical_of(sp.surface)
        key_str = normalize(canonical) if canonical else normalize(sp.surface)
        key = (sp.category, key_str)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(sp)

    counters: dict[str, int] = {}
    span_placeholder: dict[tuple[int, int], str] = {}
    mapping: list[MaskEntry] = []
    for key in order:
        category = key[0]
        counters[category] = counters.get(category, 0) + 1
        prefix = _PLACEHOLDER_PREFIX.get(category, "語")
        placeholder = f"[{prefix}{counters[category]}]"
        surfaces = tuple(dict.fromkeys(sp.surface for sp in groups[key]))
        mapping.append(MaskEntry(placeholder, category, surfaces))
        for sp in groups[key]:
            span_placeholder[(sp.start, sp.end)] = placeholder
    return tuple(mapping), span_placeholder


def _apply_mask(
    text: str, spans: list[Candidate], span_placeholder: dict[tuple[int, int], str]
) -> str:
    """マスク対象スパンを右から左へプレースホルダに置換する（重なりは右側優先で1回）。"""
    intervals = sorted(
        {(sp.start, sp.end) for sp in spans}, key=lambda x: x[0], reverse=True
    )
    result = text
    last_start = len(text) + 1
    for start, end in intervals:
        if end > last_start:  # 既に置換した右側と重なる → スキップ
            continue
        placeholder = span_placeholder.get((start, end), "[マスク]")
        result = result[:start] + placeholder + result[end:]
        last_start = start
    return result
