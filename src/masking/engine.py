"""マスキング検出エンジン（UI 非依存）。

候補生成（NER 両モデル ∪ Sudachi 品詞 ∪ マスク辞書）→ 確信度づけ → 選択された候補で
マスク適用（文書内収集で全出現に展開）。設計の背景は docs-dev/対策仮説.md を参照。

確信度（**NER×LLM の2系統合議**で決める＝§13）。NER と LLM は対等な2系統で、
NER の中の sudachi/electra/ja_ginza は「票」でなく系統内入力（NER は内部で何チャネル一致しても
外に出す意見は 1 つ）。フラット集計だと NER の3チャネルが LLM の1票を圧殺するため等分にする：
- カテゴリ … 各系統を 1 カテゴリへ畳み（特別=人名/社名/商標 が地名/その他に勝つ・重い特別が勝つ）、
  系統間でも同じく特別優先・重い方（_CAT_PRIORITY）。
- 確定 … **実辞書（名簿）一致のみ**。昇格（session）は確定にしない＝確定の出所は常に名簿。
- 強  … **2系統（NER∧LLM）がともに特別カテゴリを検出**（種別が違ってもよい）、**昇格（session）票**、
  または **連絡先の正規表現一致**（決定的だが誤検出あり得るので確定でなく強。除外リストで外せる）。
- 中  … **1系統のみ**が特別カテゴリを検出（floor：特別は必ず中以上＝弱に落とさない）。
- 弱  … どの系統も特別を出さない（地名/その他のみ。誤分類で人物が紛れるので必ずレビュー）
- 微弱 … 中/弱 のうち「コードらしき」表層（`_`/`::`/`@`/`~` を含む、数字・記号のみ、または漢字以外の
  1 文字＝社内コード/変数名/列挙子を NER が誤ラベルしたもの。例 `Em_NoYes` / `~C02` / `7-410` / `N`）。
  既定で非表示・自動マスク外（データには残す＝取りこぼさない）。確定/強（辞書・連絡先・2系統一致・
  昇格）は対象外。漢字 1 文字は実在姓（林・森 等）があるので保護＝微弱にしない。

昇格（session）＝ phase1 で強になった語を「確認済み」として phase2 に再注入したもの（§ analyze）。
実辞書（dict）とは別チャネルにし、**確定でなく強**に留める（確定＝名簿、強＝検出/確認済み、の区別を保つ）。
各票は自分のカテゴリにそのまま入れる（特例なし）。Sudachi 固有名詞-一般 は「その他」のまま扱い、
NER 社名票で社名へ昇格させたりしない（未知の英字トークン＝Reject 等の汎用語を強にしないため）。
単独モデルしか拾わない社名は中→レビュー（曖昧は人手）。重要な社名・商標は辞書が主役。

使い方：
    analysis = engine.analyze(chunks)          # 全候補（確信度・各票の判定つき）
    selected = [c for c in analysis.candidates if c.confidence in ("確定", "強")]
    result = engine.apply(analysis, selected)  # 選んだ候補でマスク（全出現に展開）
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

from src.masking.allowlist import MaskAllowlist
from src.masking.cache import NerCache, content_hash
from src.masking.dictionary import MAX_MATCH_TOKENS, MaskDictionary, normalize
from src.ner import AVAILABLE_MODELS, AnalyzedToken, NerEngine
from src.ner.engine import Analysis, ProgressCallback, sudachi_analyze_chunks

if TYPE_CHECKING:  # 型のみ（実行時 import しない＝engine は src.llm/IO に依存しない）
    from src.llm.schema import LlmDetection

# クラスタ代表カテゴリの選択優先度（地名・その他は低い）
_CAT_PRIORITY = ["人名", "社名", "商標", "連絡先", "地名", "その他"]
# カテゴリ → 優先度ランク（小さいほど優先）。同点時の代表カテゴリ選択に使う。
_CAT_RANK = {c: i for i, c in enumerate(_CAT_PRIORITY)}
# 確信度の強さ順（集約時に最良を選ぶ）。微弱＝コードらしき誤検出（既定で非表示・自動マスク外）。
# 除外＝allowlist で人が「機密でない」と判断した語（最弱・既定で非表示・自動マスク外）。
_CONF_RANK = {"確定": 4, "強": 3, "中": 2, "弱": 1, "微弱": 0, "除外": -1}
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
# 連絡先（Email/URL/Phone_Number）も **NER からは候補にしない**：`@TP` 等のコード式や数字に
# 過剰発火し、両モデル一致で「連絡先/強→自動マスク」に暴発するため（Product_Other と同じ理由）。
# 連絡先カテゴリ自体は残し、将来**正規表現で「形」から確定検出**する（メール/電話/郵便番号/型番）。
# そもそもの自動マスク対象は 人名・社名・商標（地名は弱＝レビュー）。
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
    # 連絡先（Phone_Number/Email/URL）は意図的に含めない（上のコメント参照。正規表現に委ねる）。
}

# LLM（pii-masker）の ENE type → data-redactor の6カテゴリ（§7-④。内部語彙は6カテゴリに畳む）。
# 生 ene_type は LlmSpan 側に温存し、出口1 で細分表示する（merge 語彙は汚さない）。
_ENE_TO_CATEGORY: dict[str, str] = {
    "Person": "人名",
    "Company": "社名",
    "Department": "社名",  # 内部組織名は recall 重要＝その他(弱)に落とさず社名側へ
    "Province": "地名",
    "City": "地名",
    "Country": "地名",
    "Address": "地名",
    "Facility": "地名",
    "Email": "連絡先",
    "Phone_Number": "連絡先",
    "Trademark": "商標",  # pii-masker への要望 type。主要カテゴリなので必須
    # 識別子は「その他」へ畳む（＝確信度は弱固定）。ただし LLM 由来は微弱降格を免除し弱のままレビューに残す（_LLM_IDENTIFIER_TYPES）。
    "Employee_ID": "その他",
    "Account": "その他",
    "IP_Address": "その他",
}

# 「その他」へ畳むので確信度は**弱固定**。ただし LLM が付けたこれらは _looks_like_code による
# 微弱降格を**免除**する（`7-410` 型の社員番号等が消えると recall 漏れ＝致命的）。
# ＝弱のままレビューに必ず残す（既定で自動マスクはされないが、候補から消えずデータにも残る。§7-④）。
_LLM_IDENTIFIER_TYPES = frozenset({"Employee_ID", "Account", "IP_Address"})

# 連絡先（category=連絡先）の正規表現。NER は @ や数字に過剰発火して不安定なので、
# 「形」が決まっている連絡先は決定的な正規表現で拾う（§ docs-dev/マスキング設計.md §10）。
# まずは Email のみ。URL・電話番号は同じ仕組みでここに追加する（すべて連絡先カテゴリ）。
# メールはドメインに辞書社名（exmotion 等）を内包するため、**1 件まるごと**を 1 候補にして
# 辞書による部分マスク（`x@[社名].co.jp` の体裁崩れ）を防ぐ（_contact_candidates で他候補を退ける）。
#
# パターンは WHATWG / Ruby URI::MailTo::EMAIL_REGEXP をベースに、用途に合わせて 3 点調整：
#  ① アンカー（^…$ / \A…\z）は付けない＝文中に埋め込まれたメールを finditer で拾うため。
#  ② ローカル部の許容文字から **`|` を除外**＝本ツールは `|` 区切りの表を扱い（flatten OFF 時は
#     原文に `|` が残る）、`|x@…|` の先頭 `|` まで飲み込んで表が壊れるのを防ぐ。実在メールで
#     `|` を含むものはほぼ無く recall 損失はない（WHATWG 原文は `{|}` を含む）。
#  ③ ドメインは **ドット付き＋英字 TLD（2 文字以上）を必須**にする。WHATWG/RFC は `local@host`
#     （`user@localhost` のような TLD 無しの社内ホスト）も許すが、業務文書の実在メールは必ず
#     ドット付きドメイン（`xxx@会社.co.jp`）。TLD 無しを許すと社内ジャーゴン `SmashMark@TP` や
#     `有効性@TP` 等を「メール」と誤検出する。TLD 無し/数字のみ TLD を捨てて誤検出を断つ
#     （`user@localhost` 等は落ちるが業務文書ではほぼ無い）。
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{}~-]+"
    r"@(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,}"
)
_CONTACT_PATTERNS: tuple[re.Pattern[str], ...] = (_EMAIL_RE,)

# 「コードらしさ」判定用。実在の人名・社名には現れない特徴を持つ語＝NER の誤ラベル（社内コード/
# 変数名/列挙子）とみなし、中/弱 の候補を **微弱**（既定で非表示・自動マスク外）へ落とす（§ analyze）。
# 確定/強（辞書・連絡先・2系統一致・昇格）は対象にしない＝確信度の根拠があるものは守る。
_CODE_MARKER_RE = re.compile(
    r"[_@~\[\]{}=!<>;|]|::"
)  # Em_NoYes / Em_OffOn::idOff / ピッチ@ / ~C02 / `37D]==0` / `a=b` 等（[ ] { } = ! < > ; | も）
# 「名前に使われる字」＝英字 or 日本語（かな U+3040-30FF・漢字 U+3400-9FFF）。これを 1 つも
# 含まない＝数字/記号のみ（例 7-410）。
_NAME_CHAR_RE = re.compile(r"[A-Za-z぀-ヿ㐀-鿿]")
# 漢字 1 文字（U+3400-9FFF）。1 文字語のうち漢字は実在姓（林・森・関・南 等）があるので保護する。
_KANJI_RE = re.compile(r"[㐀-鿿]")
# 英数字コード（16D / 1L / 37D）。ASCII のみ＋数字混在＝識別子。実在の人名・社名はまず数字を含まない
# （`7-Eleven`/`3M` 等は稀＝辞書登録で守る）。`-`/`.`/`&` は社名にあり得るのでマーカーには入れない。
_ASCII_DIGIT_CODE_RE = re.compile(r"^[\x21-\x7e]*[0-9][\x21-\x7e]*$")
# 全大文字ASCII（略語/ジャーゴン。FIARSL / EGPDPRY）。NER の **人名** 票はこれを実在人名と見なさない
# （実在の人名に全大文字ASCIIは無い）。社名は IBM/SAP/AWS 等があるので別扱い（_system_category の Stage 1）。
_ALLCAPS_ASCII_RE = re.compile(r"^[A-Z][A-Z0-9]*$")


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
        """指定チャネルの判定ラベルを返す（無ければ空文字。先頭1件のみ）。"""
        for ch, label in self.votes:
            if ch == channel:
                return label
        return ""

    def vote_labels(self, channel: str) -> str:
        """指定チャネルの全ラベルを ` / ` 連結で返す（重複は畳む）。

        Sudachi は形態素ごとに票が付くので、多トークンの実体（例 姓+名）では
        ``vote_label`` の先頭1件では足りない。表示用にこちらを使う。
        """
        labels = [label for ch, label in self.votes if ch == channel]
        return " / ".join(dict.fromkeys(labels))


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

    def vote_labels(self, channel: str) -> str:
        """指定チャネルの全ラベルを ` / ` 連結で返す（重複は畳む。多トークン Sudachi 用）。"""
        labels = [label for ch, label in self.votes if ch == channel]
        return " / ".join(dict.fromkeys(labels))


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
    # モデル別の解析時間 (model_name, 秒)。所要時間の可視化用（UI/CLI が表示）。
    timings: tuple[tuple[str, float], ...] = ()


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
    辞書票のラベルは ``"社名(辞書)"``、昇格票（session）は ``"社名(確認済)"`` 形式なので
    接頭のカテゴリを取り出す。
    """
    if channel in ("dict", "session"):
        return label.split("(", 1)[0] or None
    if channel == "regex":  # 連絡先（メール等）の決定的検出。label がそのままカテゴリ。
        return label or None
    if channel == "llm":  # LLM（pii-masker）票。label は ENE type。
        return _ENE_TO_CATEGORY.get(label)
    if channel == "sudachi":
        return _sudachi_category(label)
    return _NER_LABEL_CATEGORY.get(label.upper())


def tally_votes(votes: Iterable[tuple[str, str]]) -> Counter[str]:
    """票集合 → カテゴリ別の「チャネル数」（**監査用の集計**。cli の票分布表示に使う）。

    確信度・カテゴリの**決定**は :func:`_merge` の2系統合議（NER 系統 / LLM 系統）で行う＝
    この関数は決定には使わない。フラット集計だと NER の3チャネルが LLM の1票を圧殺するため
    （§13「NER と LLM は対等な2系統」。NER の中の sudachi/electra/ja_ginza は票でなく系統内入力）。
    票＝チャネル（同一チャネルの複数形態素＝姓+名 等は 1 票に畳む）。Sudachi 固有名詞-一般 は
    「その他」のまま（未知の英字トークンを社名に水増ししない＝Reject 等の汎用語を強にしない）。
    """
    channels_by_cat: dict[str, set[str]] = defaultdict(set)
    for ch, label in votes:
        cat = vote_category(ch, label)
        if cat is not None:
            channels_by_cat[cat].add(ch)
    return Counter({cat: len(chs) for cat, chs in channels_by_cat.items()})


# 確信度を数える系統（NER / LLM）から外す決定的チャネル。
#   dict=確定（名簿）／session=昇格（確認済＝強）／regex=連絡先（決定的＝強。実体は _merge を通らない）。
_DECISIVE_CHANNELS = frozenset({"dict", "session", "regex"})


def _system_category(
    votes: Iterable[tuple[str, str]], surface: str, *, llm: bool
) -> str | None:
    """1 系統の票を **1 カテゴリ**へ畳む（系統内合議）。``llm=True``＝LLM 系統 / ``False``＝NER 系統。

    NER 系統＝``dict``/``session``/``regex``/``llm`` 以外の全チャネル（sudachi・各 GiNZA モデル）。
    系統内では **特別（人名/社名/商標）が地名/その他に勝ち、重い特別が勝つ**＝``_CAT_PRIORITY``
    最上位を採る（NER が何チャネル一致しても外に出す意見は 1 つ＝§13）。票が無ければ None。

    **Stage 1（NER 限定・surface で特別票を弱める）**：ja_ginza 等が英字/コードを社名・人名へ
    誤爆するため、NER の特別票を surface で「その他」に落とし、**コードノイズが LLM と組んで
    「強」になる経路を断つ**（その後 1 系統＝中→ Stage 2 ``_demote_code_like`` で微弱化される）。
    カテゴリ非対称：
    - **人名**：全大文字ASCII（FIARSL）／コードらしき → その他（実在人名に全大文字ASCIIは無い）。
    - **社名**：コードらしき のみ → その他（``IBM``/``SAP`` 等の全大文字ASCII単体は正当ゆえ守る）。
    LLM 票は弱めない（文脈判定を尊重）。
    """
    cats: list[str] = []
    for ch, label in votes:
        if (ch == "llm") != llm or ch in _DECISIVE_CHANNELS:
            continue
        cat = vote_category(ch, label)
        if cat is None:
            continue
        if not llm:  # Stage 1: NER の特別票を surface で弱める（カテゴリ非対称）
            if cat == "人名" and (
                _looks_like_code(surface) or _ALLCAPS_ASCII_RE.match(surface)
            ):
                cat = "その他"
            elif cat == "社名" and _looks_like_code(surface):
                cat = "その他"
        cats.append(cat)
    return min(cats, key=lambda c: _CAT_RANK.get(c, 99)) if cats else None


def _representative(members: list[Candidate]) -> Candidate:
    """同一表層の出現群から代表を選ぶ：確信度が最良、同点はカテゴリ優先度が高い方。

    実体（表層）の代表カテゴリ・確信度を決めるのに使う（出現ごとに割れた種別を 1 つへ）。
    """
    return max(
        members,
        key=lambda m: (_CONF_RANK.get(m.confidence, 0), -_CAT_RANK.get(m.category, 99)),
    )


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
        self,
        chunks: Iterable[str],
        *,
        flatten_tables: bool = False,
        promote: bool = True,
        extra_terms: Iterable[tuple[str, str]] = (),
        allowlist: MaskAllowlist | None = None,
        ner_cache: NerCache | None = None,
        progress: ProgressCallback | None = None,
        llm_detection: LlmDetection | None = None,
        run_ner: bool = True,
    ) -> MaskAnalysis:
        """全候補（確信度・各票の判定つき）を作る。マスクはまだ適用しない。

        2 フェーズ：① 実辞書で確信度づけ → ② 確定/強の clean な実体（＋``extra_terms`` で
        渡された選択語）を**辞書と同等の語**としてセッション辞書に昇格し、辞書マッチと
        クラスタ分割を再実行する。これで「ある箇所で確定した語＝全出現で確定」「塊を確定語の
        境界で分割」が辞書機構の自然な帰結になる（per_entity→per_occurrence の一本化）。
        ``promote=False`` で昇格を切る（出現ごとモードの同形異義語制御用）。
        ``extra_terms`` は ``(surface, category)`` の列（UI/CLI の選択語）。
        ``progress`` はステージコールバック：各段階の開始時に (段階index, 全段階数, ラベル) を受ける。
        段階＝各モデルの解析（重い・モデルごと）＋ 候補の集約。1 モデルの解析中はサブ進捗を出さない
        （最速の既定バッチで処理するため。小バッチ化は本末転倒）。「どの段階か」を示すのが目的。
        """
        chunks = list(chunks)
        per_model: list[tuple[str, Analysis]] = []
        timings: list[tuple[str, float]] = []
        other_raw: list[Candidate] = []

        if run_ner:
            # NER 経路（NLP 処理＝GiNZA 2モデル。激重）。(content_hash, model, flatten) でキャッシュ。
            #   ヒットすれば GiNZA をスキップ。下の候補生成（辞書/除外依存）は毎回再計算する。
            n_models = len(self.engines)
            n_stages = n_models + 1  # 各モデル ＋ 候補集約
            chash = content_hash(chunks) if ner_cache is not None else ""
            for idx, e in enumerate(self.engines):
                cached = (
                    ner_cache.get(chash, e.model_name, flatten_tables)
                    if ner_cache is not None
                    else None
                )
                if (
                    progress is not None
                ):  # 重い解析の前にこの段階を表示（実行中に見える）
                    label = f"{e.model_name} を解析" + (
                        "（キャッシュ）" if cached else "中"
                    )
                    progress(idx, n_stages, label)
                t0 = perf_counter()
                if cached is not None:
                    a = cached
                else:
                    a = e.analyze_chunks(chunks, flatten_tables=flatten_tables)
                    if (
                        ner_cache is not None
                    ):  # 未確定でも解析過程として保存（再解析を一瞬に）
                        ner_cache.put(chash, e.model_name, flatten_tables, a)
                timings.append((e.model_name, perf_counter() - t0))
                per_model.append((e.model_name, a))
            if progress is not None:
                progress(n_models, n_stages, "候補を集約・確信度づけ中")
            base = per_model[0][1]
            # NLP（NER）チャネルの票＝Sudachi 品詞 ∪ GiNZA NER（§13。A 案: Sudachi 票は NER 実行時のみ）。
            other_raw = _sudachi_raw(base.tokens, base.text) + _ner_raw(
                per_model, base.text
            )
        else:
            # 軽量経路（§13 ③）：GiNZA を回さず SudachiPy 単体でトークンのみ（辞書照合用）。
            #   Sudachi 品詞票・NER 票は出さない（A 案）。ルールベース＋LLM のみで候補を作る。
            if progress is not None:
                progress(0, 1, "候補を集約・確信度づけ中（NER なし）")
            base = sudachi_analyze_chunks(chunks, flatten_tables=flatten_tables)

        text = base.text
        tokens = base.tokens
        surfaces = [t.surface for t in tokens]

        # LLM（pii-masker）検出を `llm` チャネルの票として合流（J2）。新カテゴリ・新確信度は作らず、
        #   2系統合議に **LLM 系統**として流すだけ＝単独→中／NER 系統と相乗り→強／確定は名簿のみ（§7-②）。
        if llm_detection is not None:
            other_raw.extend(_llm_raw(llm_detection, text))

        # 実辞書の票（確定の根拠）＋ `embed: true` のサブワード内包照合。両フェーズで共通。
        dict_raw = _dict_matches_raw(
            self.dictionary, tokens, surfaces, text, "dict", "(辞書)"
        )
        embed_raw = _embed_raw(self.dictionary, tokens, surfaces, text)

        # フェーズ① 実辞書で確信度づけ
        clusters = _cluster(text, dict_raw + embed_raw + other_raw)

        # フェーズ② 強の clean な語＋選択語を「確認済み」として昇格し、分割と確信度づけを再実行。
        #   昇格票は実辞書とは**別チャネル `session`**（→ 確定ではなく強）。確定は実辞書のみ。
        if promote:
            promoted = _promoted_dictionary(self.dictionary, clusters, extra_terms)
            if promoted is not None:
                session_raw = _dict_matches_raw(
                    promoted, tokens, surfaces, text, "session", "(確認済)"
                )
                clusters = _cluster(
                    text, dict_raw + embed_raw + session_raw + other_raw
                )

        # 連絡先（メール等）を正規表現で確定検出し、重なる他候補を退けて 1 件まるごとにする。
        #   例：辞書社名 exmotion を内包する `x@exmotion.co.jp` を、社名で割らずメール 1 候補に。
        contacts = _contact_candidates(text)
        if contacts:
            spans = [(c.start, c.end) for c in contacts]
            clusters = [
                c
                for c in clusters
                if not any(c.start < e and s < c.end for s, e in spans)
            ]
            clusters = sorted(clusters + contacts, key=lambda c: (c.start, c.end))

        # コードらしき誤検出（中/弱）を微弱へ落とす（既定で非表示・自動マスク外。データは残す）。
        clusters = _demote_code_like(clusters)

        # 除外リスト(allowlist)：人が「機密でない」と登録した語を「除外」へ落とす
        #   （辞書＝名簿は守る。連絡先 regex の誤検出メール等は検出由来なので除外可）。
        if allowlist is not None:
            clusters = apply_allowlist(clusters, allowlist)

        return MaskAnalysis(
            text=text,
            tokens=tokens,
            candidates=tuple(clusters),
            original_text=base.original_text or text,
            offset_map=base.offset_map or tuple(range(len(text))),
            timings=tuple(timings),
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
            # 表層ごとに代表カテゴリを 1 つに（出現ごとに割れた種別を実体単位へ統一）。
            by_surface: dict[str, list[Candidate]] = {}
            for c in selected:
                by_surface.setdefault(normalize(c.surface), []).append(c)
            collected = {k: _representative(v).category for k, v in by_surface.items()}
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

    def group_candidates(self, candidates: Iterable[Candidate]) -> list[CandidateGroup]:
        """候補を実体（**表層**）ごとにまとめる。

        **同じ表層は 1 実体に集約**する（マスクは表層＝トークン単位で効くため、カテゴリが
        出現ごとに割れていても 1 つにまとめる）。実体のカテゴリ・確信度は代表（確信度最良・
        同点はカテゴリ優先）を採る。confidence は最良、votes は和集合。
        出現ごとの個別判断は :data:`MaskAnalysis.candidates`（出現ごとモード）が担う。
        """
        groups: dict[str, list[Candidate]] = {}
        order: list[str] = []
        for c in candidates:
            canonical = self.dictionary.canonical_of(c.surface)
            key = normalize(canonical) if canonical else normalize(c.surface)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(c)

        result: list[CandidateGroup] = []
        for key in order:
            members = groups[key]
            best = _representative(members)
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


def _llm_raw(detection: LlmDetection, text: str) -> list[Candidate]:
    """LLM 検出（``LlmDetection``）の各スパンを ``llm`` チャネルの生票に変換する（Stage B）。

    票のラベルは生 ENE type（``vote_category`` が ``_ENE_TO_CATEGORY`` で6カテゴリへ写す）。
    生成段階のカテゴリは仮（最終カテゴリは _merge の2系統合議が決める）が、未投票時の保険として写像値を入れる。
    """
    out: list[Candidate] = []
    for sp in detection.spans:
        category = _ENE_TO_CATEGORY.get(sp.ene_type, "その他")
        out.append(_raw(sp.start, sp.end, text, category, ("llm", sp.ene_type)))
    return out


def _has_llm_identifier_vote(c: Candidate) -> bool:
    """LLM が識別子（社員番号/アカウント/IP）と判定した票を持つか（微弱降格の免除判定）。"""
    return any(ch == "llm" and label in _LLM_IDENTIFIER_TYPES for ch, label in c.votes)


def _demote_code_like(candidates: list[Candidate]) -> list[Candidate]:
    """中/弱 の「コードらしき」誤検出を微弱へ落とす（既定で非表示・自動マスク外。データは残す）。

    確定/強（辞書・連絡先・2系統一致・昇格）は守る＝中/弱 のみ対象。例外：LLM が識別子
    （社員番号/アカウント/IP）と判定したものは免除＝レビューに残す（§7-④。`7-410` 型でも消さない）。
    """
    return [
        (
            replace(c, confidence="微弱")
            if c.confidence in ("中", "弱")
            and _looks_like_code(c.surface)
            and not _has_llm_identifier_vote(c)
            else c
        )
        for c in candidates
    ]


# NER スパンを割る文字＝実在の人名・社名に決して現れない区切り（平文化が注入する `、`/`。`、
# 改行＝セル/文/チャンクの境界）。これらを含むスパンは「またいで融合した塊」なので片へ分ける。
_SPAN_SPLIT_CHARS = frozenset("、。\n")


def _split_on_separators(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """スパン [start,end) を ``_SPAN_SPLIT_CHARS`` で分割し、非空な片の (start,end) を返す。

    区切りを含まなければ元の 1 片をそのまま返す（無回帰）。区切り文字自身は片に含めない。
    """
    spans: list[tuple[int, int]] = []
    seg_start: int | None = None
    for i in range(start, end):
        if text[i] in _SPAN_SPLIT_CHARS:
            if seg_start is not None:
                spans.append((seg_start, i))
                seg_start = None
        elif seg_start is None:
            seg_start = i
    if seg_start is not None:
        spans.append((seg_start, end))
    return spans


# --------------------------------------------------------------------------- #
# チャネル別の票生成（§13 のチャネル分離）。各関数は「生候補（confidence 未確定）」の列を返す。
# 集約（_cluster/_merge の2系統合議）はチャネル非依存なので、ここを足し引きするだけで
# 「走ったチャネルだけ集計」が表現できる（NER 任意化＝④ の土台）。
# --------------------------------------------------------------------------- #
def _dict_matches_raw(
    dictionary: MaskDictionary,
    tokens: tuple[AnalyzedToken, ...],
    surfaces: list[str],
    text: str,
    channel: str,
    suffix: str,
) -> list[Candidate]:
    """辞書のトークン照合を ``channel`` 票（生候補）にする（dict→確定 / session→強 の根拠）。"""
    out: list[Candidate] = []
    for m in dictionary.match(surfaces):
        start = tokens[m.start_token].start
        end = tokens[m.end_token - 1].end
        out.append(
            _raw(start, end, text, m.category, (channel, f"{m.category}{suffix}"))
        )
    return out


def _embed_raw(
    dictionary: MaskDictionary,
    tokens: tuple[AnalyzedToken, ...],
    surfaces: list[str],
    text: str,
) -> list[Candidate]:
    """``embed: true`` 辞書語のサブワード境界内包照合を dict 票（生候補）にする。

    命中はトークン全体でなく**一致したサブワード部分だけ**を span にする（`SmashMark`→`Smash` 等）。
    """
    out: list[Candidate] = []
    for ti, sub_s, sub_e, _canon, category in dictionary.embedded_matches(surfaces):
        es = tokens[ti].start + sub_s
        ee = tokens[ti].start + sub_e
        out.append(_raw(es, ee, text, category, ("dict", f"{category}(辞書·内包)")))
    return out


def _sudachi_raw(tokens: tuple[AnalyzedToken, ...], text: str) -> list[Candidate]:
    """Sudachi 品詞（固有名詞）を ``sudachi`` 票（生候補）にする（NLP/NER チャネルの一部）。"""
    out: list[Candidate] = []
    for t in tokens:
        category = _sudachi_category(t.tag)
        if category is not None:
            out.append(_raw(t.start, t.end, text, category, ("sudachi", t.tag)))
    return out


def _ner_raw(per_model: list[tuple[str, Analysis]], text: str) -> list[Candidate]:
    """GiNZA NER エンティティを各モデル名チャネルの票（生候補）にする。

    Product_Other 系（その他）はノイズ過多のため除外。NER スパンは `、`/`。`/改行で分割
    （セル/文/チャンクをまたいだ融合の人工物を片へ割る。融合でも実体を取りこぼさない）。
    """
    out: list[Candidate] = []
    for model_name, analysis in per_model:
        for ent in analysis.entities:
            category = _NER_LABEL_CATEGORY.get(ent.label.upper())
            if category is None:
                continue
            for sp_s, sp_e in _split_on_separators(text, ent.start, ent.end):
                out.append(_raw(sp_s, sp_e, text, category, (model_name, ent.label)))
    return out


def _looks_like_code(surface: str) -> bool:
    """社内コード/変数名らしき表層か（実在の人名・社名には現れない特徴）。

    いずれかに該当：
    - 記号マーカー ``_`` / ``::`` / ``@`` / ``~`` / ``[ ] { } = ! < > ; |`` を含む
      （例 ``Em_NoYes`` / ``Em_OffOn::idOff`` / ``ピッチ@`` / ``~C02`` / ``37D]==0``）。
    - 英字も日本語（かな・漢字）も含まない＝数字・記号のみ（例 ``7-410``）。
    - **ASCII のみ＋数字を含む**（例 ``16D`` / ``1L`` / ``37D``）。実在の人名・社名はまず数字を含まない。
    - **1 文字語（漢字を除く）**＝ASCII 英字・かな・数字・記号 1 文字（例 ``N`` / ``D``）。実在名では
      まず無い。ただし**漢字 1 文字は実在姓**（林・森・関 等）があるので保護＝対象外。

    NER（electra/ja_ginza）がこれらを社名/人名に誤ラベルし中/弱で大量に湧くため、
    :meth:`MaskingEngine.analyze` で **中/弱 のみ** を **微弱**（既定で非表示・自動マスク外）へ落とす（Stage 2）。
    確定/強（辞書・連絡先・2系統一致・昇格）は対象にしない＝根拠があるものは守る。
    NER の票そのものを弱める Stage 1 は :func:`_system_category` を参照。
    """
    if _CODE_MARKER_RE.search(surface):
        return True
    if not _NAME_CHAR_RE.search(surface):
        return True
    if _ASCII_DIGIT_CODE_RE.match(surface):
        return True
    return len(surface) == 1 and not _KANJI_RE.match(surface)


def _contact_candidates(text: str) -> list[Candidate]:
    """連絡先（メール等）を正規表現で検出し、**1 件まるごと**の候補にする。

    finditer で全出現を個別に拾う（＝展開不要で全部マスクされる）。確信度は **強**（自動マスク）：
    正規表現は決定的だが誤検出があり得る（例 `20181210112500@MH01R2.sdf`）。**確定は名簿のみ**に
    予約し、連絡先は強に留める＝自動マスクしつつ、誤検出は除外リストで外せる。
    """
    out: list[Candidate] = []
    for pat in _CONTACT_PATTERNS:
        for m in pat.finditer(text):
            out.append(
                Candidate(
                    m.start(),
                    m.end(),
                    m.group(),
                    "連絡先",
                    "強",
                    (("regex", "連絡先"),),
                )
            )
    return out


def _cluster(text: str, cands: list[Candidate]) -> list[Candidate]:
    """重なる候補スパンをまとめ、票数とカテゴリから確信度を決める。

    辞書一致を含むクラスタは**辞書境界で分割**する（粗い NER スパンに辞書の実体を
    飲み込ませない）。例：`SONY・Nikon・Canon` を 1 つにせず SONY/Nikon/Canon の各辞書
    スパンへ、`小浜出身` を `小浜` だけに割る。辞書一致が無ければ従来どおり 1 スパンに統合。
    """
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
            clusters.extend(_resolve_cluster(text, start, end, members))
            start, end, members = c.start, c.end, [c]
    clusters.extend(_resolve_cluster(text, start, end, members))
    return clusters


def _has_dict_vote(c: Candidate) -> bool:
    """実辞書票を持つか（確定の判定用。session=昇格は含めない）。"""
    return any(ch == "dict" for ch, _ in c.votes)


def apply_allowlist(
    candidates: Iterable[Candidate], allowlist: MaskAllowlist
) -> list[Candidate]:
    """除外リスト一致の候補を「除外」へ落とす（**解析をやり直さず**候補だけ書き換える）。

    recall 安全：守るのは **辞書（人の名簿）一致のみ**（`_has_dict_vote`）。辞書語は人が意図して
    「必ずマスク」と登録したものなので、除外リストでは上書きしない。
    一方、**連絡先 regex（強）**（例 `20181210112500@MH01R2.sdf` の誤検出メール）は検出由来なので、
    人が明示的に除外できる（表層単位なので本物のメールには波及しない）。
    ``analyze`` 内でも、UI が確定済み解析へ後から除外を反映するときも、この同一ロジックを使う。
    """
    if not allowlist:
        return list(candidates)
    return [
        (
            replace(c, confidence="除外")
            if not _has_dict_vote(c) and c.surface in allowlist
            else c
        )
        for c in candidates
    ]


def apply_allowlist_to_analysis(
    analysis: MaskAnalysis, allowlist: MaskAllowlist
) -> MaskAnalysis:
    """解析結果（候補）に除外リストを後から適用した新しい :class:`MaskAnalysis` を返す。

    NER は再実行しない（候補の confidence を書き換えるだけ）＝除外の反映が即座で済む。
    """
    return replace(
        analysis, candidates=tuple(apply_allowlist(analysis.candidates, allowlist))
    )


def _has_anchor_vote(c: Candidate) -> bool:
    """分割の足場になる票を持つか＝実辞書 or 昇格（session）。確定済みの実体の境界で割る。"""
    return any(ch in ("dict", "session") for ch, _ in c.votes)


# 昇格対象から外す「区切り」を含む表層（塊＝複数実体）。中黒/スラッシュ/カンマ/空白など。
_SEPARATOR_CHARS = "・･/／,，、　 \t×"


def _has_separator(surface: str) -> bool:
    return any(ch in _SEPARATOR_CHARS for ch in surface)


def _promoted_dictionary(
    dictionary: MaskDictionary,
    clusters: list[Candidate],
    extra_terms: Iterable[tuple[str, str]],
) -> MaskDictionary | None:
    """強の clean な語＋選択語だけを集めた「確認済み」辞書を返す（実辞書とは別）。

    これは**確定ではなく強**として再注入するためのもの（実辞書 = 確定 と区別）。
    対象は phase1 で **強**になった語（＝実辞書に無い・NER/Sudachi で確認できた語）と
    選択語。確定（= 実辞書）は実辞書自身が全出現を拾うので昇格不要。
    昇格しないもの：地名/その他、区切りを含む表層（塊＝再び 1 つに固まる）。
    何も無ければ None。
    """
    additions: dict[str, tuple[str, str]] = {}
    for c in clusters:
        if c.confidence != "強":
            continue
        if c.category in ("地名", "その他") or _has_separator(c.surface):
            continue
        key = normalize(c.surface)
        if key:
            additions.setdefault(key, (c.surface, c.category))
    for surface, category in extra_terms:
        key = normalize(surface)
        if key and not _has_separator(surface):
            additions.setdefault(key, (surface, category))
    return MaskDictionary(additions) if additions else None


def _within_any(s: int, e: int, spans: list[tuple[int, int]]) -> bool:
    return any(ds <= s and e <= de for ds, de in spans)


def _spans_across(c: Candidate, dict_spans: list[tuple[int, int]]) -> bool:
    """c が少なくとも 1 つの辞書スパンを「またぐ」（より広く覆う）＝橋渡し。"""
    return any(
        c.start <= ds and de <= c.end and (c.start, c.end) != (ds, de)
        for ds, de in dict_spans
    )


def _segment_spans(members: list[Candidate]) -> list[tuple[int, int]]:
    """メンバー群を重なりでまとめた区間（min start, max end）の列にする。"""
    if not members:
        return []
    ordered = sorted(members, key=lambda m: (m.start, m.end))
    out: list[tuple[int, int]] = []
    s, e = ordered[0].start, ordered[0].end
    for m in ordered[1:]:
        if m.start < e:
            e = max(e, m.end)
        else:
            out.append((s, e))
            s, e = m.start, m.end
    out.append((s, e))
    return out


def _resolve_cluster(
    text: str, start: int, end: int, members: list[Candidate]
) -> list[Candidate]:
    """1 クラスタを確信度づけ。辞書一致があれば辞書境界で分割して返す。

    足場（辞書 or 昇格＝session）が無ければ 1 件に統合。足場があれば：
    - 各足場スパンを独立候補に（粗い NER スパンに飲み込ませない。例 `SONY・Nikon・Canon`→3分割）。
    - 足場スパンを「またがず・埋もれてもいない」その他メンバー（＝列挙に取り残された実体。例
      `SONY|Nikon|Canon` の昇格後の Nikon）も区間にまとめて出す。普通名詞のみの隙間（例 `小浜出身`
      の `出身`＝Sudachi 候補が無い）は出さない。
    - 各 emit スパンには**重なる全メンバーの票**を集める＝橋渡しの広い NER 票も確信度に効かせる。
    """
    anchor_members = [m for m in members if _has_anchor_vote(m)]
    if not anchor_members:
        return [_merge(text, start, end, members)]
    anchor_spans = sorted({(m.start, m.end) for m in anchor_members})
    leftover = [
        m
        for m in members
        if not _has_anchor_vote(m)
        and not _spans_across(m, anchor_spans)
        and not _within_any(m.start, m.end, anchor_spans)
    ]
    emit = sorted(set(anchor_spans) | set(_segment_spans(leftover)))
    out: list[Candidate] = []
    for s, e in emit:
        overlap = [m for m in members if m.start < e and s < m.end]
        out.append(_merge(text, s, e, overlap))
    return out


def _merge(text: str, start: int, end: int, members: list[Candidate]) -> Candidate:
    """重なる票群を 1 候補へ。種別・確信度は **NER×LLM の2系統合議**で決める（§13）。

    ① 種別＝特別（人名/社名/商標）が地名/その他に勝つ・両系統が別の特別なら重い方（_CAT_PRIORITY）。
    ② 確信度＝「特別（隠すべき）」と言った**系統数**：2系統=強／1系統=中（floor）／特別なし=弱。
       辞書=確定（最優先・名簿）／昇格(session)=強。NER の内部チャネル数は確信度に効かない。
    """
    votes = tuple(dict.fromkeys(v for m in members for v in m.votes))
    surface = text[start:end]
    # 辞書票があれば辞書のカテゴリで確定（系統の票数によらず最優先＝名簿が砦）。
    dict_cats = [
        cat
        for ch, label in votes
        if ch == "dict" and (cat := vote_category(ch, label)) is not None
    ]
    if dict_cats:
        return Candidate(start, end, surface, dict_cats[0], "確定", votes)
    # 各系統を 1 カテゴリへ畳む（系統内合議）。NER 系統 / LLM 系統。
    sys_cats = [
        c
        for c in (
            _system_category(votes, surface, llm=False),
            _system_category(votes, surface, llm=True),
        )
        if c is not None
    ]
    if not sys_cats:
        # 系統票が無い（session 昇格のみ等）＝カテゴリ未確定の中扱い（下の昇格で強になり得る）。
        category, confidence = members[0].category, "中"
    else:
        # ① 種別＝特別が地名/その他に勝つ・重い特別が勝つ＝_CAT_PRIORITY 最上位。
        category = min(sys_cats, key=lambda c: _CAT_RANK.get(c, 99))
        # ② 確信度＝特別（地名/その他でない）と言った系統数。2系統=強／1系統=中（floor）／0=弱。
        n_special = sum(1 for c in sys_cats if c not in ("地名", "その他"))
        confidence = "強" if n_special >= 2 else "中" if n_special >= 1 else "弱"
    # 昇格（session＝他箇所で確認済み）票がそのカテゴリにあれば最低でも強（決定的だが名簿でないので
    #   確定にはしない＝確定は実辞書だけ、という区別を保つ）。
    if category not in ("地名", "その他") and any(
        ch == "session" and vote_category(ch, label) == category for ch, label in votes
    ):
        confidence = "強"
    return Candidate(start, end, surface, category, confidence, votes)


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
                        start,
                        end,
                        text[start:end],
                        category,
                        "確定",
                        (("collected", "展開"),),
                    )
                )
                ranges.append((start, end))
            i = e_tok
        else:
            i += 1
    return out


def _map_span(
    offset_map: tuple[int, ...], start: int, end: int
) -> tuple[int, int] | None:
    """平坦化テキストのスパン [start, end) を原文座標へ写す。

    挿入文字（区切り `、`/句点 `。`/連結改行＝対応 -1）を読み飛ばし、原文側で実体を
    覆う連続範囲を返す。スパン内に実体が無ければ None。検出は `、` でセルを割っているため
    1 つのスパンはセル内に収まり、原文範囲に `|` は挟まらない。
    """
    n = min(end, len(offset_map))
    o_start = next(
        (offset_map[i] for i in range(start, n) if offset_map[i] != -1), None
    )
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

    **同じ表層（canonical）は 1 プレースホルダに統一**する（カテゴリが出現ごとに割れていても
    1 つにまとめる＝表層単位でマスク、と整合）。実体のカテゴリは代表（確信度最良・同点は優先）を採る。
    表記ゆれ（英語表記↔カタカナ表記・略称・旧称）も canonical で同じプレースホルダに寄る。
    辞書で**置換語（mask）が指定**された実体は、自動採番でなくその語を使う（未指定は自動採番）。
    """
    groups: dict[str, list[Candidate]] = {}
    order: list[str] = []
    group_canonical: dict[str, str | None] = {}
    for sp in spans:
        canonical = dictionary.canonical_of(sp.surface)
        key = normalize(canonical) if canonical else normalize(sp.surface)
        if key not in groups:
            groups[key] = []
            order.append(key)
            group_canonical[key] = canonical
        groups[key].append(sp)

    counters: dict[str, int] = {}
    span_placeholder: dict[tuple[int, int], str] = {}
    mapping: list[MaskEntry] = []
    for key in order:
        members = groups[key]
        category = _representative(members).category
        canonical = group_canonical[key]
        custom = dictionary.custom_placeholder(canonical) if canonical else None
        if custom:
            placeholder = custom
        else:
            counters[category] = counters.get(category, 0) + 1
            prefix = _PLACEHOLDER_PREFIX.get(category, "語")
            placeholder = f"[{prefix}{counters[category]}]"
        surfaces = tuple(dict.fromkeys(sp.surface for sp in members))
        mapping.append(MaskEntry(placeholder, category, surfaces))
        for sp in members:
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
