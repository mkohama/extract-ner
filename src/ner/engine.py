"""固有表現抽出エンジン（UI 非依存）。

テキストから固有表現を抽出する。特定のカテゴリ（ラベル）だけを抜き出す
フィルタリングにも対応する。Streamlit / CLI などの表示層からはこのエンジンを
呼び出すだけにし、エンジン自体は表示・IO に依存しない。

使用例::

    from src.ner import NerEngine

    engine = NerEngine("ja_ginza_electra")
    result = engine.extract("エクスモ社に勤める担当者", labels=["Company"])
    for ent in result.entities:
        print(ent.text, ent.label)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

import spacy

from src.ner.preprocess import prepare_for_ner_with_map

# 進捗（ステージ）コールバック型：progress(stage_index, stage_total, label)。各ステージ開始時に
# 1 回呼ぶ。UI 非依存（UI 側でステージ表示に使う）。
# 注: 1 モデルの nlp.pipe は最速の既定バッチ（全チャンクを 1 バッチ）で処理するため、その処理中の
# サブ進捗は出さない（小バッチにすると transformer が遅くなる＝本末転倒。electra 実測で +8〜29%）。
# 代わりに「どのモデル/段階を実行中か（何段/全何段）」を示す。
ProgressCallback = Callable[[int, int, str], None]

# ja_ginza_electra（torch/thinc/huggingface）系が出す deprecation 警告を抑制する。
# 推論のたびに大量に出る `torch.cuda.amp.autocast(...)` ほか、初回ロード時の
# huggingface_hub / transformers の deprecation などのノイズ（実害なし・我々のコード起因ではない）。
# サードパーティのモジュールに限定して抑制し、自前コードの警告は残す。
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)
for _noisy in ("thinc", "torch", "huggingface_hub", "transformers"):
    warnings.filterwarnings("ignore", category=FutureWarning, module=rf"{_noisy}.*")

# 利用可能な GiNZA モデル（先頭が既定）
AVAILABLE_MODELS: tuple[str, ...] = ("ja_ginza_electra", "ja_ginza")
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# GiNZA が内部で使う SudachiPy のトークナイズ上限（1 回の解析あたりのバイト数）。
# これを超えると `SudachiError: Input is too long` で落ちる。
SUDACHI_MAX_BYTES = 49149
# 上限に対する安全マージン。通常はチャンク分割（src.core.document.text_splitter）で
# 既に十分小さくなっているが、巨大な 1 チャンク/1 行が来ても確実に通すための保険。
SAFE_CHUNK_BYTES = 40000

# チャンク結合時の区切り（表示テキスト＝解析テキストの連結に使う）。
CHUNK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Entity:
    """抽出された 1 件の固有表現。"""

    text: str
    label: str
    start: int  # 解析対象テキスト中の開始文字位置
    end: int  # 同・終了文字位置


@dataclass(frozen=True)
class TokenInfo:
    """1 トークンの診断情報（recall の穴を実データで観察するためのデバッグ用）。

    マスキング目的では「GiNZA の NER が逃した固有名詞を、文脈非依存な
    SudachiPy の品詞（``tag``）で拾えるか」が要点。両者を並べて観察する。
    """

    text: str  # 表層形
    tag: str  # SudachiPy 品詞（例: 名詞-固有名詞-人名-姓）。文脈依存が小さい
    pos: str  # UD 品詞（例: PROPN）
    ent_type: str  # GiNZA の NER ラベル（無ければ ""）
    ent_iob: str  # B / I / O（エンティティ境界）
    is_oov: bool  # 語彙外フラグ（モデルのベクトル有無に依存。electra では参考値）
    norm: str  # 正規化表層形


@dataclass(frozen=True)
class AnalyzedToken:
    """全文オフセット付きの 1 トークン（マスキングのパイプラインが使う）。"""

    start: int  # 連結した全文中の開始文字位置
    end: int  # 同・終了文字位置
    surface: str
    tag: str  # SudachiPy 品詞
    pos: str  # UD 品詞


@dataclass(frozen=True)
class Analysis:
    """1 モデルでの解析結果（全文・トークン列・NER エンティティ）。

    トークンは SudachiPy のトークナイズ（品詞つき）、entities はこのモデルの
    NER ラベル。複数モデルを併用するときは entities を和集合する（トークンは
    同じ SudachiPy なのでどれか 1 つで足りる）。

    平坦化（``flatten_tables=True``）したときは、``text``/``tokens``/``entities`` は
    すべて**平坦化後テキスト基準**。一方 ``original_text`` は平坦化前の原文（`|` 入り）、
    ``offset_map[i]`` は ``text`` の i 文字目に対応する ``original_text`` の文字位置
    （挿入文字は -1）。検出は平坦化テキストで行い、マスクは原文へ写して当てるため
    （src.masking）に使う。平坦化しない場合は ``original_text == text``・恒等写像。
    """

    text: str
    tokens: tuple[AnalyzedToken, ...]
    entities: tuple[Entity, ...]
    original_text: str = ""
    offset_map: tuple[int, ...] = ()


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
        """1 つのテキストから固有表現を抽出する。

        長文（SudachiPy のトークナイズ上限超）でも落ちないよう、内部で
        バイト数安全なチャンクに分割してから解析する。ファイルや kb-mcp の
        ように元から複数チャンクに分かれている場合は :meth:`extract_chunks`
        を使う（kb-mcp と同じ分割単位で解析でき、結果も揃う）。

        Args:
            text: 解析対象のテキスト。
            labels: 残すカテゴリ（ラベル）。None なら全件。
            flatten_tables: True なら Markdown テーブルを平文化してから解析する。

        Returns:
            ExtractionResult（解析対象テキストと抽出結果）。
        """
        return self.extract_chunks([text], labels=labels, flatten_tables=flatten_tables)

    def extract_chunks(
        self,
        chunks: Iterable[str],
        *,
        labels: Iterable[str] | None = None,
        flatten_tables: bool = False,
    ) -> ExtractionResult:
        """複数チャンクから固有表現を抽出し、1 つの結果にマージする。

        各チャンクを個別に解析し、エンティティの文字位置を「全チャンクを
        :data:`CHUNK_SEPARATOR` で連結したテキスト」基準に補正してまとめる。
        これにより displaCy 表示（manual モード）がそのまま使える。

        チャンクが SudachiPy の上限（:data:`SUDACHI_MAX_BYTES`）を超える場合は、
        さらにバイト数安全な小片へ分割してから解析する（保険）。

        Args:
            chunks: 解析対象チャンクの列（kb-mcp / Splitter の出力など）。
            labels: 残すカテゴリ（ラベル）。None なら全件。
            flatten_tables: True なら各チャンクを平文化してから解析する。

        Returns:
            ExtractionResult（連結した解析テキストと、位置補正済みの抽出結果）。
        """
        # 解析する小片を確定（バイト数安全分割 → 平文化 → 空片除去）
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)

        # 小片ごとに NER（nlp.pipe でバッチ処理）し、全文基準にオフセット補正
        entities: list[Entity] = []
        offset = 0
        sep_len = len(CHUNK_SEPARATOR)
        for piece, doc in zip(pieces, self.nlp.pipe([p.flat for p in pieces])):
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char + offset,
                        end=ent.end_char + offset,
                    )
                )
            offset += len(piece.flat) + sep_len

        result = ExtractionResult(
            text=CHUNK_SEPARATOR.join(p.flat for p in pieces),
            entities=tuple(entities),
        )
        if labels is not None:
            result = result.filter(labels)
        return result

    def debug_tokens(
        self,
        chunks: Iterable[str],
        *,
        flatten_tables: bool = False,
    ) -> list[TokenInfo]:
        """各トークンの SudachiPy 品詞と GiNZA NER ラベルを並べて返す（デバッグ用）。

        :meth:`extract_chunks` と**同じ小片分割**（平文化 → バイト数安全分割）を
        通すため、ここで見えるトークンは実際に NER が解析する対象と一致する。

        マスキングの recall の穴（NER は逃すが SudachiPy は固有名詞・人名として
        割っている語など）を実データで観察するために使う。

        Args:
            chunks: 解析対象チャンクの列。
            flatten_tables: True なら各チャンクを平文化してから解析する。

        Returns:
            空白トークンを除いた :class:`TokenInfo` のリスト（出現順）。
        """
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)
        infos: list[TokenInfo] = []
        for doc in self.nlp.pipe([p.flat for p in pieces]):
            for tok in doc:
                if tok.is_space:
                    continue
                infos.append(
                    TokenInfo(
                        text=tok.text,
                        tag=tok.tag_,
                        pos=tok.pos_,
                        ent_type=tok.ent_type_,
                        ent_iob=tok.ent_iob_,
                        is_oov=tok.is_oov,
                        norm=tok.norm_,
                    )
                )
        return infos

    def analyze_chunks(
        self,
        chunks: Iterable[str],
        *,
        flatten_tables: bool = False,
    ) -> Analysis:
        """チャンク列を解析し、全文・オフセット付きトークン・NER エンティティを返す。

        :meth:`extract_chunks` と同じ小片分割・オフセット補正を使う。マスキングの
        パイプライン（src.masking）が、SudachiPy 品詞とスパンを使って候補を作るために用いる。
        """
        pieces = _prepare_pieces(chunks, flatten_tables=flatten_tables)
        tokens: list[AnalyzedToken] = []
        entities: list[Entity] = []
        omap: list[int] = []  # 平坦化テキストの各文字 → 原文の文字位置（挿入は -1）
        orig_parts: list[str] = []
        offset = 0  # 平坦化テキスト基準のオフセット（トークン/エンティティ用）
        orig_offset = 0  # 原文基準のオフセット
        sep_len = len(CHUNK_SEPARATOR)
        for idx, (piece, doc) in enumerate(
            zip(pieces, self.nlp.pipe([p.flat for p in pieces]))
        ):
            if idx > 0:  # 小片の区切り（CHUNK_SEPARATOR）は flat/orig 双方に入る
                omap.extend(orig_offset + k for k in range(sep_len))
                offset += sep_len
                orig_offset += sep_len
                orig_parts.append(CHUNK_SEPARATOR)
            for tok in doc:
                if tok.is_space:
                    continue
                start = tok.idx + offset
                tokens.append(
                    AnalyzedToken(
                        start=start,
                        end=start + len(tok.text),
                        surface=tok.text,
                        tag=tok.tag_,
                        pos=tok.pos_,
                    )
                )
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char + offset,
                        end=ent.end_char + offset,
                    )
                )
            omap.extend(orig_offset + c if c != -1 else -1 for c in piece.cmap)
            offset += len(piece.flat)
            orig_offset += len(piece.orig)
            orig_parts.append(piece.orig)
        return Analysis(
            text=CHUNK_SEPARATOR.join(p.flat for p in pieces),
            tokens=tuple(tokens),
            entities=tuple(entities),
            original_text="".join(orig_parts),
            offset_map=tuple(omap),
        )


@dataclass(frozen=True)
class _Piece:
    """NER に渡す 1 小片。平坦化したときは原文との対応表も持つ。"""

    flat: str  # NER へ渡す（平坦化後）テキスト
    orig: str  # 対応する原文（平坦化前。`|` 入り）
    cmap: tuple[int, ...]  # flat の各文字 → orig の文字位置（挿入文字は -1）


def _prepare_pieces(
    chunks: Iterable[str], *, flatten_tables: bool = False
) -> list[_Piece]:
    """チャンク列を、実際に NER へ渡す小片（バイト数安全・空片除去済み）に整える。

    :meth:`NerEngine.extract_chunks` と :meth:`NerEngine.debug_tokens` が同じ
    入力で解析するよう、分割処理をここに集約する。平坦化はバイト数安全分割の**後**に
    各小片へ適用し、原文（`|` 入り）と平坦化後の文字位置対応表（cmap）を保持する。
    """
    pieces: list[_Piece] = []
    for chunk in chunks:
        for orig in _byte_safe_pieces(chunk):
            # 括弧グルー対策の空白挿入は flatten の有無によらず常に適用する
            # （`姓A(社B)` の埋没＝辞書語の漏れを防ぐ）。flatten 時はテーブル平文化も。
            flat, cmap = prepare_for_ner_with_map(orig, flatten_tables=flatten_tables)
            if flat.strip():
                pieces.append(_Piece(flat, orig, tuple(cmap)))
    return pieces


def _byte_safe_pieces(text: str, max_bytes: int = SAFE_CHUNK_BYTES) -> list[str]:
    """テキストを UTF-8 で ``max_bytes`` 以下の小片に分割する（保険的フォールバック）。

    まず行（``\\n``）境界でまとめ、1 行で超える場合のみ文字単位で強制分割する。
    通常はチャンク分割で十分小さいため、ここはほぼ素通りする。
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    pieces: list[str] = []
    buf = ""
    for line in text.split("\n"):
        candidate = f"{buf}\n{line}" if buf else line
        if len(candidate.encode("utf-8")) <= max_bytes:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(line.encode("utf-8")) > max_bytes:
            pieces.extend(_hard_split_by_bytes(line, max_bytes))
        else:
            buf = line
    if buf:
        pieces.append(buf)
    return pieces


def _hard_split_by_bytes(text: str, max_bytes: int) -> list[str]:
    """1 行が上限を超える場合に、文字単位でバイト数上限まで詰めて分割する。"""
    pieces: list[str] = []
    buf = ""
    buf_bytes = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))
        if buf and buf_bytes + ch_bytes > max_bytes:
            pieces.append(buf)
            buf = ""
            buf_bytes = 0
        buf += ch
        buf_bytes += ch_bytes
    if buf:
        pieces.append(buf)
    return pieces
