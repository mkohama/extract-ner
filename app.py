"""GiNZA 固有表現抽出 Streamlit UI（薄い表示層）。

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力 UI・表示のみを行う。

起動:
    uv run streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.document.document_loader import DocumentLoader
from src.masking import (
    AUTO_MASK_CONFIDENCE,
    MaskAllowlist,
    MaskDictionary,
    MaskingEngine,
    NerCache,
    apply_allowlist_to_analysis,
    content_hash,
    load_allowlist_entries,
    load_entries,
    save_allowlist_entries,
    save_entries,
)
from src.llm import cached_detect
from src.llm.detect_layer import DEFAULT_MODEL as LLM_MODEL
from src.llm.schema import LlmDetection
from src.llm.windows import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS
from src.masking.engine import vote_category
from src.ner import (
    AVAILABLE_MODELS,
    DEFAULT_COLOR,
    MASKING_CATEGORY_COLORS,
    NerEngine,
    build_color_map,
    render_html,
    render_masking_html,
)
from src.ner.preprocess import build_body
from src.sources import SAMPLE_TEXT, load_chunks_from_file
from src.sources.kb_mcp import (
    DEFAULT_KB_MCP_URL,
    get_document_chunks_sync,
    list_documents_sync,
    suppress_async_generator_errors,
)

suppress_async_generator_errors()

# アップロード可能な拡張子 (DocumentLoader が対応する形式)
SUPPORTED_EXTENSIONS = sorted(e[1:] for e in DocumentLoader.SUPPORTED_EXTENSIONS)

# モデル選択肢（AVAILABLE_MODELS と同順）と説明
MODELS = list(AVAILABLE_MODELS)
MODEL_DESCRIPTIONS = {
    "ja_ginza_electra": "高精度・低速 (ELECTRA / Transformer ベース)",
    "ja_ginza": "軽量・高速 (CNN/Sudachi ベース)",
}


@st.cache_resource(show_spinner="GiNZA モデルを読み込み中 ...")
def get_engine(model_name: str) -> NerEngine:
    """モデルごとに NerEngine を生成・キャッシュする（モデルもここでロード）。"""
    engine = NerEngine(model_name)
    _ = engine.nlp  # ここでロードしておく（以降の解析を高速化）
    return engine


# マスク辞書・除外リスト・キャッシュの既定パス（ルート直下 data/）
_DEFAULT_DICT = Path(__file__).resolve().parent / "data" / "mask_dict.yaml"
_DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "data" / "mask_allowlist.yaml"
_DEFAULT_CACHE_DB = Path(__file__).resolve().parent / "data" / "cache.db"


@st.cache_resource(show_spinner=False)
def _ner_cache() -> NerCache:
    """NER 層キャッシュ（解析過程の高速化）。SQLite 接続は軽いがインスタンスは共有する。"""
    return NerCache(_DEFAULT_CACHE_DB)


def _load_allowlist(allowlist_path: str) -> MaskAllowlist:
    if allowlist_path and Path(allowlist_path).exists():
        return MaskAllowlist.load(allowlist_path)
    return MaskAllowlist.empty()


@st.cache_resource(show_spinner="モデルを読み込み中 ...")
def _masking_engine_for_models(models: tuple[str, ...]) -> MaskingEngine:
    """モデルだけを読み込んだマスキングエンジン（重いのでキャッシュ。辞書は都度差し替える）。"""
    engine = MaskingEngine(dictionary=MaskDictionary.empty(), models=list(models))
    for e in engine.engines:
        _ = e.nlp  # 先にロード
    return engine


def _load_dictionary(dict_path: str) -> MaskDictionary:
    if dict_path and Path(dict_path).exists():
        return MaskDictionary.load(dict_path)
    return MaskDictionary.empty()


def get_masking_engine(models: tuple[str, ...], dict_path: str) -> MaskingEngine:
    """マスキングエンジンを返す。モデルはキャッシュ、**辞書は毎回読み直して差し替える**。

    辞書編集 UI で保存した直後の再実行でも、モデルを再ロードせずに新しい辞書が反映される。
    """
    engine = _masking_engine_for_models(tuple(models))
    engine.dictionary = _load_dictionary(dict_path)
    return engine


@st.cache_data(show_spinner=False)
def fetch_kb_document_chunks(url: str, doc_id: str) -> list[str]:
    """kb-mcp から指定文書のチャンク本文を取得する (url+doc_id でキャッシュ)。"""
    return get_document_chunks_sync(doc_id, url)


def extract_chunks_from_upload(uploaded_file) -> list[str]:
    """アップロードされたファイルを一時保存し、チャンク化したテキストを返す。"""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        return load_chunks_from_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _short_models(models: tuple[str, ...]) -> str:
    """NER モデル名を短い別名に（一覧表示用）。"""
    names = {"ja_ginza_electra": "electra", "ja_ginza": "ginza"}
    return ", ".join(names.get(m, m) for m in models)


def _kb_doc_label(meta: dict) -> str:
    """kb-mcp 文書メタから表示名を作る。"""
    name = meta.get("title") or meta.get("file_name") or meta.get("id") or "?"
    path = meta.get("file_path") or ""
    return f"{name}　({path})" if path else str(name)


@st.fragment
def _select_table_fragment(
    df: pd.DataFrame,
    ids: list,
    *,
    key: str,
    sel_key: str,
    caption: str,
    column_config: dict | None = None,
) -> None:
    """1 行クリックで単一選択するテーブル（``st.fragment`` で再描画を局所化）。

    行をクリックすると **このテーブルだけ** 再実行され（画面全体は再描画しない＝チラつき/
    スクロール飛びを抑える）。選んだ行の識別子 ``ids[row]`` を ``st.session_state[sel_key]`` に
    書く（未選択は ``None``）。単一選択なので前の選択は自動で置き換わる（チェック残り無し）。
    位置選択だが、呼び出し側が安定順で渡し ``ids`` で解決するので並び替えの取り違えは起きない。
    列見出しクリックで表示の並べ替えも可能（選択には影響しない）。
    """
    st.caption(caption)
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
        key=key,
    )
    rows = event.selection.rows
    st.session_state[sel_key] = ids[rows[0]] if rows else None


def _llm_cache_status(content_hash: str, flatten: bool, has_llm: bool) -> str:
    """🗂 ピッカー用の LLM キャッシュ状態。``✓``=現行設定で有効 / ``⚠ 要更新``=旧版のみ / ``—``=無し。

    現在の ``(LLM_MODEL, flatten, _detector_version())`` に一致する行があれば「✓」（そのまま使える）。
    LLM 検出履歴はあるが現行設定に一致しなければ「⚠ 要更新」（窓ポリシー等が変わった＝再検出が要る）。
    実際のキャッシュヒット条件（has_llm）と同じ鍵で判定するので、表示と挙動がズレない。
    """
    if not has_llm:
        return "—"
    if _ner_cache().has_llm(content_hash, LLM_MODEL, flatten, _detector_version()):
        return "✓"
    return "⚠ 要更新"


@st.fragment
def _cache_picker_fragment(docs: list, flatten: bool) -> None:
    """🗂 キャッシュ選択の UI 一式（テーブル＋選択依存の操作）を 1 つの fragment にまとめる。

    行クリックは **この fragment だけ** 再実行され、画面全体は再描画しない。
    選択した content_hash を ``st.session_state["cache_sel"]`` に入れる（``render_input`` が読む）。
    ``docs`` は呼び出し側がソース名で安定ソート済み（並び替えで取り違えない）。``flatten`` は現在の平文化
    設定で、LLM 列の「現行設定で有効か（✓）/要更新（⚠）」判定に使う。
    NER のやり直し（キャッシュ無視）は読み込み時でなく **🔍 NER検出 タブ**で行う（パイプライン化に伴う移設）。
    """
    df = pd.DataFrame(
        [
            {
                "ソース": d.source_name,
                "種別": d.source_kind,
                "チャンク": d.chunk_count,
                "文字数": d.char_count,
                "NER": _short_models(d.models) or "—",
                "LLM": _llm_cache_status(d.content_hash, flatten, d.has_llm),
                "解析日時": d.created_at,
            }
            for d in docs
        ]
    )
    st.caption(
        f"キャッシュ済み: {len(docs)} 文書（行をクリックして選択）。"
        "LLM 列: **✓**=現在の設定で有効／**⚠ 要更新**=キャッシュはあるが窓ポリシー等が変わり再検出が必要／**—**=無し。"
    )
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="cache_pick",
    )
    rows = event.selection.rows
    sel_hash = docs[rows[0]].content_hash if rows else None
    st.session_state["cache_sel"] = sel_hash
    if sel_hash is None:
        st.caption("👆 行をクリックして文書を選択してください。")
        return
    if not _ner_cache().get_chunks(sel_hash):
        st.warning(
            "このキャッシュにはチャンク本文がありません（チャンク保存より前の古いエントリ）。"
            "一度ふつうに解析し直すと、以降ここから選べます。"
        )
        return
    st.caption(
        "保存チャンクを入力に使います。[📥 読み込む] 後、各タブ（NER検出 / LLM検出 / マージ&確信度）で実行します。"
        "NER のやり直し（キャッシュ無視）は NER検出 タブのチェックで行えます。"
    )


def render_input(
    input_mode: str,
    flatten_tables: bool,
) -> tuple[tuple | None, str, str, Callable[[], list[str]] | None]:
    """入力ウィジェットを描画し、解析に必要な情報を返す。

    ``flatten_tables`` は現在の平文化設定で、🗂 キャッシュ選択の LLM 列の有効/要更新判定に渡す。

    重いテキスト化／ダウンロードは**ここでは行わず**、``get_chunks`` 呼び出しに遅延させる
    （実際にチャンクを取り出すのは「解析する」ボタンが押されたときだけ）。

    戻り値 ``(input_id, input_kind, source_label, get_chunks)``：
      - ``input_id``  … 入力の同一性を表すハッシュ可能なタプル（署名に使う）。
                        入力未確定なら ``None``（解析不可）。
      - ``input_kind``… ``"text" / "file" / "kb"``（平文プレビューの要否判定に使う）。
      - ``source_label``… 結果見出しに出す表示名。
      - ``get_chunks``… 呼ぶとチャンク列を返す callable（未確定なら ``None``）。
    """
    if input_mode.startswith("📄"):
        uploaded_file = st.file_uploader(
            f"対応形式: {', '.join(SUPPORTED_EXTENSIONS)}",
            type=SUPPORTED_EXTENSIONS,
        )
        if uploaded_file is not None:
            input_id = ("file", uploaded_file.name, uploaded_file.size)

            def get_file_chunks(f=uploaded_file) -> list[str]:
                with st.spinner("ファイルをテキスト化・チャンク化中 ..."):
                    return extract_chunks_from_upload(f)

            return input_id, "file", uploaded_file.name, get_file_chunks
        return None, "file", "", None

    if input_mode.startswith("📚"):
        url = st.text_input(
            "kb-mcp サーバー URL",
            value=st.session_state.get("kb_url", DEFAULT_KB_MCP_URL),
        )
        st.session_state["kb_url"] = url
        if st.button("文書リストを取得"):
            try:
                with st.spinner("文書一覧を取得中 ..."):
                    st.session_state["kb_docs"] = list_documents_sync(url)
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("kb_docs", None)
                st.error(f"kb-mcp への接続/取得に失敗しました: {e}")

        docs = st.session_state.get("kb_docs")
        if docs is None:
            st.info(
                "kb-mcp サーバを起動し（`uv run kb-mcp-server --transport http --port 8000`）、"
                "[文書リストを取得] を押してください。"
            )
            return None, "kb", "", None
        if not docs:
            st.warning("kb-mcp に登録された文書がありません。")
            return None, "kb", "", None

        # 解析済み（＝NER キャッシュ済み）の kb 文書 → 「✓」を頭に付ける目安（名前一致）に使う。
        cached_kb = {
            d.source_name
            for d in _ner_cache().list_documents()
            if d.source_kind == "kb"
        }

        # 1 行クリックで選ぶテーブル（fragment で再描画局所化）。📦＝キャッシュ済みの目安。
        # チャンク数は kb-mcp の一覧メタ（knowledge://documents の chunk_count）をそのまま表示。
        kb_df = pd.DataFrame(
            [
                {
                    "📦": "✓" if _kb_doc_label(m) in cached_kb else "",
                    "名前": (
                        m.get("title") or m.get("file_name") or m.get("id") or "?"
                    ),
                    "チャンク": m.get("chunk_count", ""),
                    "パス": m.get("file_path") or "",
                }
                for m in docs
            ]
        )
        _select_table_fragment(
            kb_df,
            list(range(len(docs))),
            key="kb_pick",
            sel_key="kb_sel",
            caption=(
                f"kb-mcp 文書: {len(docs)} 件"
                "（行をクリックして選択。📦＝キャッシュ済みの目安・チャンク数は kb-mcp の登録値）"
            ),
            column_config={"📦": st.column_config.TextColumn("📦", width="small")},
        )
        sel = st.session_state.get("kb_sel")
        if sel is None or sel >= len(docs):
            return None, "kb", "", None
        meta = docs[sel]
        doc_id = meta.get("id") or meta.get("document_id")
        if not doc_id:
            return None, "kb", "", None

        label = _kb_doc_label(meta)

        def get_kb_chunks(u=url, d=doc_id) -> list[str]:
            with st.spinner("本文を取得中 ..."):
                return fetch_kb_document_chunks(u, d)

        return ("kb", url, doc_id), "kb", label, get_kb_chunks

    if input_mode.startswith("🗂"):
        docs = _ner_cache().list_documents()
        if not docs:
            st.info(
                "キャッシュがありません。テキスト/ファイル/kb-mcp を解析すると登録され、"
                "ここから入力元に選べるようになります。"
            )
            return None, "cache", "", None
        # 選択 UI 一式（テーブル＋強制再解析チェック＋案内）は 1 つの fragment 内で描く。
        # **ソース名で安定ソート**して渡すので、解析（created_at 更新）で行が動かず選択がズレない。
        docs = sorted(docs, key=lambda doc: doc.source_name)
        _cache_picker_fragment(docs, flatten_tables)

        # fragment が session_state に書いた選択・強制再解析を読んで get_chunks を組む（ここでは
        # 選択依存のウィジェットを描かない＝行クリックで再描画されない外側に widget を置かない）。
        sel_hash = st.session_state.get("cache_sel")
        matches = [x for x in docs if x.content_hash == sel_hash]
        if not matches:
            return None, "cache", "", None
        d = matches[0]
        cached_chunks = _ner_cache().get_chunks(d.content_hash)
        if not cached_chunks:  # 警告は fragment 内で表示済み（古いエントリ）
            return None, "cache", d.source_name, None

        # 読み込みは保存チャンクを返すだけ（NER は回さない）。NER のやり直し（キャッシュ無視）は
        # NER検出 タブのチェックで delete_ner する（パイプライン化に伴い読み込み時の強制再解析は廃止）。
        def get_cache_chunks(c=cached_chunks) -> list[str]:
            return c

        return (("cache", d.content_hash), "cache", d.source_name, get_cache_chunks)

    # テキスト入力（単一チャンクとして扱う。長文でもエンジン側で安全分割される）
    input_text = st.text_area("解析するテキスト", value=SAMPLE_TEXT, height=200)
    if input_text.strip():
        return ("text", input_text), "text", "入力テキスト", (lambda t=input_text: [t])
    return None, "text", "", None


def _dict_signature(dict_path: str) -> tuple[str, float | None]:
    """辞書ファイルの同一性（パス＋更新時刻）。保存で内容が変われば署名がズレる。"""
    p = Path(dict_path)
    try:
        return (str(p), p.stat().st_mtime) if p.exists() else (str(p), None)
    except OSError:
        return (str(p), None)


def _masking_settings_sig(
    models: list[str], flatten_tables: bool, dict_path: str, allowlist_path: str
) -> tuple:
    """マスキングの設定署名（モデル/平文化/辞書 mtime/除外リスト mtime）。

    再解析バナーの判定に使う。除外を「再解析なし」で反映したときに、この署名で stored を
    更新しておけばバナーが誤って出ない（main と同じ式を使うため共通化）。
    """
    return (
        "masking",
        tuple(models),
        flatten_tables,
        _dict_signature(dict_path),
        _dict_signature(allowlist_path),
    )


def _readable_text_block(
    text: str, *, placeholders: dict[str, str] | None = None, height: int = 400
) -> str:
    """読み取り専用テキストを、グレーアウトしない読める div にする HTML を返す。

    `st.text_area(disabled=True)` は背景・文字ともグレーで編集不可カーソルになり読みにくい。
    代わりに通常色・選択可・改行保持の div で表示する。``placeholders``（プレースホルダ→
    カテゴリ）を渡すと、マスク後の伏せ字をカテゴリ色で強調し「どこが変わったか」を見せる。
    """
    escaped = html.escape(text)
    if placeholders:
        pattern = re.compile(
            "|".join(re.escape(p) for p in sorted(placeholders, key=len, reverse=True))
        )

        def _repl(mo: re.Match) -> str:
            ph = mo.group(0)
            color = MASKING_CATEGORY_COLORS.get(placeholders.get(ph, ""), DEFAULT_COLOR)
            return (
                f'<mark style="background:{color}; color:#000; '
                f'padding:0 .15em; border-radius:3px;">{ph}</mark>'
            )

        escaped = pattern.sub(_repl, escaped)
    return (
        f'<div style="height:{height}px; overflow:auto; resize:vertical; '
        "white-space:pre-wrap; word-break:break-word; line-height:1.9; "
        'border:1px solid rgba(128,128,128,0.25); border-radius:6px; padding:0.6em;">'
        f"{escaped}</div>"
    )


def _render_extracted_text(chunks: list[str]) -> None:
    """テキスト化された平文（チャンク連結）を確認用に表示する。

    ファイルや kb-mcp は元がバイナリ/外部なので、何が抽出されたかを目視・ダウンロードできるようにする。
    チャンク境界は ``--- チャンク境界 ---`` で示す（解析単位の確認用）。
    """
    text = "\n\n".join(chunks)
    boundary = "\n\n----- チャンク境界 -----\n\n"
    shown = boundary.join(chunks)
    with st.expander(
        f"📄 テキスト化結果（平文 / {len(chunks)} チャンク・{len(text)} 文字）を確認",
        expanded=False,
    ):
        st.html(_readable_text_block(shown, height=300))
        st.download_button(
            "⬇ 平文をダウンロード",
            text,
            file_name="extracted.txt",
            mime="text/plain",
        )


def _stage_callback(status, n_chunks: int):
    """ステージ表示用コールバック。重い処理の前に「何段/全何段・何を実行中か」を出す。

    1 モデルの解析中はサブ進捗を出さない（最速の既定バッチで処理＝途中経過は取れない）。
    代わりにどの段階かを示す。Streamlit はブロッキング中も直前に積んだ表示を反映するので、
    各段階の開始時に更新すれば「実行中の段階」が見える。
    """

    def cb(idx: int, total: int, label: str) -> None:
        status.info(
            f"⏳ ステージ {idx + 1}/{total}: {label} ...（{n_chunks} チャンク）"
        )

    return cb


def _timing_caption(timings, total_seconds: float, n_chunks: int) -> str:
    """解析時間の説明文（合計・モデル別・チャンク当たり）を作る。"""
    per = total_seconds / n_chunks if n_chunks else 0.0
    if timings:  # モデル別の内訳（マスキングは 2 モデル）
        parts = " / ".join(f"{m}: {s:.1f}s" for m, s in timings)
        return f"⏱ 解析 合計 {total_seconds:.1f}s（{parts}）・ {per:.2f}s/チャンク（{n_chunks} 件）"
    return f"⏱ 解析 {total_seconds:.1f}s ・ {per:.2f}s/チャンク（{n_chunks} 件）"


def analyze_ner(chunks: list[str], model_name: str, flatten_tables: bool):
    """NER 解析（重い）。ボタン押下時のみ呼ぶ。戻り値は (結果, 所要秒)。"""
    engine = get_engine(model_name)
    start = time.perf_counter()
    with st.spinner(f"⏳ {model_name} で解析中 ...（{len(chunks)} チャンク）"):
        result = engine.extract_chunks(chunks, flatten_tables=flatten_tables)
    elapsed = time.perf_counter() - start
    return result, elapsed


def render_ner_result(stored: dict, *, view_height: int, font_size: float) -> None:
    """固有表現抽出（NER）の結果表示（保存済み結果から。再解析しない）。"""
    model_name = stored["model_name"]
    flatten_tables = stored["flatten"]
    source_label = stored["source_label"]
    result = stored["result"]
    chunks = stored["chunks"]

    engine = get_engine(model_name)
    colors = build_color_map(engine.available_labels())

    if stored.get("elapsed"):
        st.success(_timing_caption((), stored["elapsed"], len(chunks)), icon="✅")

    if source_label:
        st.subheader(f"解析結果: {source_label}")

    present_labels = result.labels
    selected_labels = st.multiselect(
        "表示するラベル（最初は全件。選択を外すと非表示）",
        options=present_labels,
        default=present_labels,
    )
    shown = result.filter(selected_labels)

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.caption(
            f"モデル: `{model_name}` / 平文化: {'ON' if flatten_tables else 'OFF'}"
        )
        st.metric(
            "表示中 / 全固有表現",
            f"{len(shown.entities)} / {len(result.entities)} 件",
        )
        st.metric("解析文字数", f"{len(result.text)} 文字")
        st.metric("チャンク数", f"{len(chunks)} 件")

        if result.entities:
            counts = pd.Series([e.label for e in result.entities]).value_counts()
            st.write("**ラベル別件数**")
            st.dataframe(
                counts.rename_axis("ラベル").reset_index(name="件数"),
                hide_index=True,
                width="stretch",
            )

    with col_main:
        html = render_html(shown, colors)
        st.html(
            f'<div style="height:{view_height}px; overflow:auto; resize:vertical; '
            "line-height:2.2; border:1px solid rgba(128,128,128,0.25); "
            f'border-radius:6px; padding:0.5em; font-size:{font_size}em;">{html}</div>'
        )

    st.subheader("固有表現の一覧")
    if shown.entities:
        rows = [
            {"テキスト": e.text, "ラベル": e.label, "開始": e.start, "終了": e.end}
            for e in shown.entities
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("表示対象の固有表現がありません。")


def _context(text: str, start: int, end: int, width: int = 20) -> str:
    """出現箇所の前後文脈スニペット（対象を《》で囲む）。"""
    s = max(0, start - width)
    e = min(len(text), end + width)
    head = "…" if s > 0 else ""
    tail = "…" if e < len(text) else ""
    return f"{head}{text[s:start]}《{text[start:end]}》{text[end:e]}{tail}"


# 確信度の並び順（確定→強→中→弱→微弱→除外）。文字列順だと崩れるので明示する。
_CONFIDENCE_ORDER = {"確定": 0, "強": 1, "中": 2, "弱": 3, "微弱": 4, "除外": 5}
# 確信度フィルタの選択肢と既定（微弱＝コードらしき誤検出・除外＝allowlist は既定で非表示）。
_CONFIDENCE_LEVELS = ["確定", "強", "中", "弱", "微弱", "除外"]
_CONFIDENCE_DEFAULT = ["確定", "強", "中", "弱"]


def _confidence_label(confidence: str) -> str:
    """並び順の番号を前置した表示用ラベル（例 '1 : 確定'）。

    列ヘッダで文字列ソートしても 確定→強→中→弱 の順になるようにする
    （番号なしだと文字コード順で「中→確定」になってしまう）。1=確定 … 4=弱。
    """
    return f"{_CONFIDENCE_ORDER.get(confidence, 9) + 1} : {confidence}"


def _sorted_by_confidence(items, *, key, mask_rank=None):
    """（あれば）マスク状態 → 確信度 降順（確定→強→中→弱）→ 第2キー（表層）昇順で並べる。

    items は ``.confidence`` を持つ候補 / 実体。``mask_rank`` は ``it -> int``（小さいほど上。
    例：マスク中=0 / 一部=1 / 未マスク=2）。指定すると**マスク中の行が先頭**に来る。
    """
    return sorted(
        items,
        key=lambda it: (
            mask_rank(it) if mask_rank is not None else 0,
            _CONFIDENCE_ORDER.get(it.confidence, 9),
            key(it),
        ),
    )


def _auto_mask_spans(analysis) -> set:
    """既定でマスクする出現の span 集合（確定/強）。共有選択 ``mask_sel`` の初期値（現状と同一）。"""
    return {
        (c.start, c.end)
        for c in analysis.candidates
        if c.confidence in AUTO_MASK_CONFIDENCE
    }


def _apply_selection(engine, analysis, sel: set):
    """共有選択 ``sel``（マスクする span 集合）から :class:`MaskResult` を作る。

    **結果はビューに依存しない**：``sel`` にある span（＝選択した出現）だけを正確にマスクする
    （展開しない）。実体ごと/出現ごとは「``sel`` をどう編集するか」の違いだけで、表示結果は常に
    ``sel`` と一致する＝**ビュー切替で広がらない**。実体ごとで語をチェックして[反映]すると、その語の
    全（検出）出現が ``sel`` に入る（＝実質その語を一括マスク）が、それは明示操作のときだけ。
    """
    selected = [c for c in analysis.candidates if (c.start, c.end) in sel]
    return engine.apply(analysis, selected, expand=False)


def _render_by_entity(engine, analysis, confidences, sel, ver, stored):
    """実体ごと：同じ語は文書内の全出現を一括マスク。``confidences`` で表示する確信度を絞る。"""

    def _group_mask_rank(g) -> int:
        spans = {(m.start, m.end) for m in g.members}
        if spans <= sel:  # 全出現が選択＝マスク（チェック ON）
            return 0
        return 1 if spans & sel else 2  # 一部選択 / 未選択

    all_groups = _sorted_by_confidence(
        engine.group_candidates(analysis.candidates),
        key=lambda g: g.surface,
        mask_rank=_group_mask_rank,
    )
    groups = [g for g in all_groups if g.confidence in confidences]
    hidden = len(all_groups) - len(groups)
    st.subheader(f"マスク候補（{len(groups)} 実体）— チェックで選択")
    cap = "チェックは**全出現が選択されているときだけ ON**。チェックした実体は文書内の全出現がマスクされます。"
    if hidden:
        cap += f"（確信度フィルタで {hidden} 実体を非表示）"
    cap += "　※`選択状況` が **⚠一部** の語は、出現ごとビューで一部だけ選択中です。"
    st.caption(cap)
    rows = []
    for g in groups:
        spans = [(m.start, m.end) for m in g.members]
        n_sel = sum(1 for s in spans if s in sel)
        if n_sel == 0:
            status = ""
        elif n_sel < g.count:
            status = f"⚠ 一部 {n_sel}/{g.count}"
        else:
            status = f"全 {g.count}"
        rows.append(
            {
                "マスク": n_sel == g.count,  # 全出現が選択済みのときだけ ON
                "選択状況": status,
                "除外": g.confidence == "除外",
                "確信度": _confidence_label(g.confidence),
                "カテゴリ": g.category,
                "表層": g.surface,
                "出現": g.count,
                "ja_ginza": g.vote_labels("ja_ginza"),
                "electra": g.vote_labels("ja_ginza_electra"),
                "Sudachi": g.vote_labels("sudachi"),
                "LLM": g.vote_labels("llm"),
                "辞書": "○" if g.vote_label("dict") else "",
            }
        )
    table = pd.DataFrame(rows)
    st.caption("チェックしてから **[✅ マスクを反映]** を押すと反映されます。")
    # st.form で囲む：チェックのたびに再実行せず、ボタン押下時だけまとめて適用する
    # （data_editor は編集ごとに rerun し画面が先頭へ飛ぶため。フォームで抑止）。
    # data_editor の鍵に ver を含める＝共有選択 sel が変わったら貼り直し、古い編集状態を残さない。
    with st.form("mask_entity_form"):
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            disabled=[c for c in table.columns if c not in ("マスク", "除外")],
            column_config={
                "マスク": st.column_config.CheckboxColumn("マスク"),
                "除外": st.column_config.CheckboxColumn(
                    "除外",
                    help="チェックして [🚫 選択を除外リストへ] を押すと候補外に。",
                ),
            },
            key=f"mask_entity_{ver}",
        )
        col_a, col_b = st.columns([1, 1])
        applied = col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    if applied:  # **変化したチェックだけ**反映（出現ごとの部分選択を壊さない）
        new_sel = set(sel)
        for g, on, ex in zip(groups, masks, excludes):
            spans = {(m.start, m.end) for m in g.members}
            was_on = (
                spans <= sel
            )  # 表示時のチェック状態（全出現が選択済み＝チェック ON）
            if ex or (was_on and not on):  # 除外 or チェックを外した → 全出現を削除
                new_sel -= spans
            elif on and not was_on:  # 新たにチェック → 全出現を追加
                new_sel |= spans
            # 変化なし（was_on == on）→ そのまま（部分選択を保持）
        stored["mask_sel"] = new_sel
        stored["mask_ver"] = ver + 1
        st.rerun()
    to_exclude = [g.surface for g, ex in zip(groups, excludes) if ex]
    return to_exclude, excl


def _render_by_occurrence(engine, analysis, confidences, sel, ver, stored):
    """出現ごと：各出現を個別にマスク。チェックは共有選択 sel を読み書きする。"""
    all_cands = _sorted_by_confidence(
        list(analysis.candidates),
        key=lambda c: c.surface,
        mask_rank=lambda c: 0 if (c.start, c.end) in sel else 1,  # マスク中を先頭に
    )
    cands = [c for c in all_cands if c.confidence in confidences]
    hidden = len(all_cands) - len(cands)
    st.subheader(f"マスク候補（{len(cands)} 出現）— 出現ごとに選択")
    cap = (
        "各出現を個別にマスク（フランク=人名 vs フランクに=気軽に、等を文脈で使い分け）。"
        "**選んだ出現だけ**マスクし、他の出現には広げません。"
    )
    if hidden:
        cap += f"（確信度フィルタで {hidden} 出現を非表示）"
    cap += " チェックしてから **[✅ マスクを反映]** を押すと反映されます。"
    st.caption(cap)
    table = pd.DataFrame(
        [
            {
                "マスク": (c.start, c.end) in sel,
                "除外": c.confidence == "除外",
                "確信度": _confidence_label(c.confidence),
                "カテゴリ": c.category,
                "表層": c.surface,
                "文脈": _context(analysis.text, c.start, c.end),
                "ja_ginza": c.vote_labels("ja_ginza"),
                "electra": c.vote_labels("ja_ginza_electra"),
                "Sudachi": c.vote_labels("sudachi"),
                "LLM": c.vote_labels("llm"),
                "辞書": "○" if c.vote_label("dict") else "",
            }
            for c in cands
        ]
    )
    with st.form("mask_occurrence_form"):
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            disabled=[c for c in table.columns if c not in ("マスク", "除外")],
            column_config={
                "マスク": st.column_config.CheckboxColumn("マスク"),
                "除外": st.column_config.CheckboxColumn(
                    "除外",
                    help="チェックして [🚫 選択を除外リストへ] を押すと候補外に。",
                ),
            },
            key=f"mask_occurrence_{ver}",
        )
        col_a, col_b = st.columns([1, 1])
        applied = col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    if applied:  # 表示中の出現について sel を更新（ON=その span を追加／OFF=削除）
        new_sel = set(sel)
        for c, on, ex in zip(cands, masks, excludes):
            span = (c.start, c.end)
            if on and not ex:
                new_sel.add(span)
            else:
                new_sel.discard(span)
        stored["mask_sel"] = new_sel
        stored["mask_ver"] = ver + 1
        st.rerun()
    to_exclude = [c.surface for c, ex in zip(cands, excludes) if ex]
    return to_exclude, excl


def render_dict_editor(dict_path: str) -> None:
    """マスク辞書の確認・追加・編集・保存 UI（独立タブ）。

    行を編集/追加/削除して保存すると `data/mask_dict.yaml`（dict_path）へ書き出す。
    「置換」列に値を入れると、その実体のマスク後の伏せ字を固定できる（空なら自動採番）。
    """
    st.caption(
        "カテゴリ / 代表表記 / 別名（カンマ区切り）/ 置換（任意。空なら `[社1]` 等を自動採番）。"
        "**保存先はローカルの辞書ファイル**（機密・git 管理外）。"
    )
    st.caption(
        "📝 **追加**＝一番下の空行に入力。"
        "🗑 **削除**＝左端のチェックを ON → キーボードの **Delete / Backspace** キー"
        "（または表右上のゴミ箱）。いずれも **[💾 辞書を保存] を押すまでファイルには反映されません**。"
    )
    path = Path(dict_path)
    entries = load_entries(path) if path.exists() else []
    rows = [
        {
            "カテゴリ": e["category"],
            "代表表記": e["canonical"],
            "別名": ", ".join(e["aliases"]),
            "置換": e["mask"],
            "内包": e["embed"],
        }
        for e in entries
    ]
    df = pd.DataFrame(rows, columns=["カテゴリ", "代表表記", "別名", "置換", "内包"])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        height=500,
        key="dict_editor",
        column_config={
            "カテゴリ": st.column_config.SelectboxColumn(
                "カテゴリ", options=["社名", "商標", "人名"], default="社名"
            ),
            "置換": st.column_config.TextColumn("置換", help="空なら自動採番"),
            "内包": st.column_config.CheckboxColumn(
                "内包",
                default=False,
                help="複合語の中（例 SmashMark の Smash、CBMark の CB）も伏字にする。"
                "境界一致なので ECBType の CB は拾わない。",
            ),
        },
    )
    if st.button("💾 辞書を保存", type="primary", key="dict_save"):

        def cell(value: object) -> str:
            # data_editor の空セルは NaN（float）。`nan or ""` は nan が truthy で
            # すり抜けて "nan" になるので、明示的に空文字へ落とす。
            return "" if pd.isna(value) else str(value).strip()

        new_entries = [
            {
                "category": cell(r["カテゴリ"]) or "社名",
                "canonical": cell(r["代表表記"]),
                "aliases": [a.strip() for a in cell(r["別名"]).split(",") if a.strip()],
                "mask": cell(r["置換"]),
                "embed": bool(r["内包"]) if not pd.isna(r["内包"]) else False,
            }
            for _, r in edited.iterrows()
        ]
        kept = [e for e in new_entries if e["canonical"]]
        save_entries(path, kept)
        st.success(f"保存しました: {path}（{len(kept)} 件）")


def render_cache_view() -> None:
    """キャッシュ済み文書の一覧・削除（🗂 キャッシュ モード）。"""
    cache = _ner_cache()
    docs = cache.list_documents()
    if not docs:
        st.info(
            "キャッシュはまだありません。マスキングで文書を解析すると、NER 結果が自動で"
            "登録され、次回以降の解析が高速になります。"
        )
        return

    st.caption(
        f"キャッシュ済み: {len(docs)} 文書　"
        "（**種別**はプルダウンで修正できます＝再解析で `cache` に潰れた行を元の出所へ。"
        "編集後は [💾 種別の変更を保存] を押す）"
        "　LLM 列: **✓**=現行 detector_version で有効／**⚠ 要更新**=旧版のキャッシュのみ（窓ポリシー等が変わった）／**—**=無し。"
    )
    current_ver = _detector_version()
    kind_options = ["text", "file", "kb", "cache"]

    def _llm_col(d) -> str:
        # 管理ビューは flatten 文脈を持たないので detector_version 一致のみで判定する。
        vers = cache.llm_versions(d.content_hash)
        if not vers:
            return "—"
        return "✓" if current_ver in vers else "⚠ 要更新"

    df = pd.DataFrame(
        [
            {
                "削除": False,
                "ソース": d.source_name,
                "種別": d.source_kind,
                "チャンク": d.chunk_count,
                "文字数": d.char_count,
                "NER": _short_models(d.models) or "—",
                "LLM": _llm_col(d),
                "解析日時": d.created_at,
                "hash": d.content_hash[:12],
            }
            for d in docs
        ]
    )
    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        disabled=[c for c in df.columns if c not in ("削除", "種別")],
        column_config={
            "削除": st.column_config.CheckboxColumn("削除"),
            "種別": st.column_config.SelectboxColumn(
                "種別", options=kind_options, required=True
            ),
        },
        key="cache_view",
    )

    # 種別の変更（出所の手動修正）を検出して保存する。
    kind_changes = [
        (d.content_hash, new_kind)
        for d, new_kind in zip(docs, edited["種別"].tolist())
        if new_kind != d.source_kind
    ]
    if kind_changes and st.button(
        f"💾 種別の変更（{len(kind_changes)} 件）を保存", type="primary"
    ):
        for content_hash_, new_kind in kind_changes:
            cache.set_source_kind(content_hash_, new_kind)
        st.success(f"種別を更新しました（{len(kind_changes)} 件）。")
        st.rerun()

    to_delete = [d for d, on in zip(docs, edited["削除"].tolist()) if on]
    if to_delete and st.button(f"🗑 選択した {len(to_delete)} 件のキャッシュを削除"):
        for d in to_delete:
            cache.delete(d.content_hash)
        st.success(f"{len(to_delete)} 件のキャッシュを削除しました。")
        st.rerun()


def render_allowlist_editor(allowlist_path: str) -> None:
    """除外リストの確認・追加・編集・保存 UI（独立タブ）。

    マスク辞書と同様、行を編集/追加/削除して保存すると `data/mask_allowlist.yaml` へ書き出す。
    1 列（除外語）だけのフラットなリスト。
    """
    st.caption(
        "📝 **追加**＝一番下の空行に語を入力。"
        "🗑 **削除**＝左端のチェックを ON → **Delete / Backspace**（または表右上のゴミ箱）。"
        "いずれも **[💾 除外リストを保存] を押すまでファイルには反映されません**。"
        "**保存先はローカルファイル**（機密・git 管理外）。"
    )
    path = Path(allowlist_path)
    surfaces = load_allowlist_entries(path) if path.exists() else []
    df = pd.DataFrame({"除外語": surfaces}, columns=["除外語"])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        height=500,
        key="allowlist_editor",
    )
    if st.button("💾 除外リストを保存", type="primary", key="allowlist_save"):
        kept = [
            s.strip()
            for s in edited["除外語"].tolist()
            if not pd.isna(s) and str(s).strip()
        ]
        save_allowlist_entries(path, kept)
        st.success(f"保存しました: {path}（{len(kept)} 件）")


def analyze_masking(
    chunks: list[str],
    models: list[str],
    flatten_tables: bool,
    dict_path: str,
    allowlist_path: str,
    llm_detection: LlmDetection | None = None,
    run_ner: bool = True,
):
    """マスキング検出。ボタン押下時のみ呼ぶ。

    ``llm_detection`` を渡すと LLM 検出を ``llm`` チャネルとして票に合流する（出口2）。
    ``run_ner=False`` で GiNZA を回さず、辞書＋regex（＋LLM）だけで集約する（§13・軽い）。
    """
    engine = get_masking_engine(tuple(models), dict_path)
    allowlist = _load_allowlist(allowlist_path)
    status = st.empty()
    analysis = engine.analyze(
        chunks,
        flatten_tables=flatten_tables,
        allowlist=allowlist,
        ner_cache=_ner_cache(),
        progress=_stage_callback(status, len(chunks)),
        llm_detection=llm_detection,
        run_ner=run_ner,
    )
    status.empty()
    return analysis


# detector_version の**静的部分**：pii-masker のコミット版（`pii-masker@<hash>`）だけ。
#   submodule 更新時に sync-pii-masker が自動で書き換える（この hash 文字列を正規表現で置換）。
# 窓ポリシー（win…）はここに書かず、実値（env or windows.py 既定）から _detector_version() が自動合成する
# ＝env で変えるだけで detector_version が変わりキャッシュ自動無効化（コード編集・手動バンプ不要）。
# 旧 ene-vN（type-map 版）は廃止：_ENE_TO_CATEGORY は解析時に毎回当たる後段変換で、LLM 検出キャッシュ
#   （生 ene_type のみ保存）に影響しない＝バンプ不要だったため。変更は次の解析で自動反映される。
_DETECTOR_STATIC = "pii-masker@9d9942e"


def _env_int(name: str, default: int) -> int:
    """環境変数を int で読む。未設定・不正値なら ``default``（チューニング用の安全側フォールバック）。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _window_policy() -> tuple[int, int]:
    """LLM 検出の窓ポリシー (max_tokens, overlap)。env で上書き可・既定は windows.py の定数。

    ``LLM_WINDOW_MAX_TOKENS`` / ``LLM_WINDOW_OVERLAP_TOKENS`` を .env に置くだけで調整でき、
    値を変えると _detector_version() 経由で **LLM 検出キャッシュが自動で無効化** される（再検出）。
    """
    return (
        _env_int("LLM_WINDOW_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        _env_int("LLM_WINDOW_OVERLAP_TOKENS", DEFAULT_OVERLAP_TOKENS),
    )


def _detector_version() -> str:
    """LLM 検出器の版 ``pii-masker@<hash>|win<max>ov<ov>``。

    win… は現在の窓ポリシー（env or 既定）から合成する。pii-masker@<hash> は _DETECTOR_STATIC（静的・
    sync-pii-masker が自動書換）。type-map（_ENE_TO_CATEGORY）は版に含めない（解析時の後段変換で、検出
    キャッシュに影響しないため。マップの変更は次の解析で自動反映される）。
    """
    max_tokens, overlap = _window_policy()
    return f"{_DETECTOR_STATIC}|win{max_tokens}ov{overlap}"


def run_llm_detection(
    chunks: list[str],
    flatten_tables: bool,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[str, LlmDetection]:
    """LLM 検出（Stage A）を実行（キャッシュ越し）。本文 ``text`` と ``LlmDetection`` を返す。

    pii-masker を呼ぶ（実機・Azure・``az login`` 前提）。キャッシュは
    ``(content_hash, model, flatten, detector_version)`` で、同一文書は再呼び出ししない。
    ``force=True`` でキャッシュ無視の再検出（LLM 層のみ。NER キャッシュには触れない）。
    ``progress(i, n)`` は窓ごとの進捗（キャッシュヒット時は呼ばれない）。
    窓ポリシー（max_tokens/overlap）は env で上書き可（_window_policy）。版にも反映されるので整合する。
    """
    max_tokens, overlap = _window_policy()
    body = build_body(chunks, flatten_tables=flatten_tables)
    detection = cached_detect(
        _ner_cache(),
        content_hash(chunks),
        body.text,
        flatten=flatten_tables,
        detector_version=_detector_version(),
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        progress=progress,
        force=force,
    )
    return body.text, detection


def render_llm_result(body_text: str, detection: LlmDetection) -> None:
    """LLM 単独ビュー（出口1）：LLM が拾った検出だけを displaCy＋表で表示する。"""
    st.caption(f"モデル `{detection.model}` / 版 `{detection.detector_version}`")
    col_main, col_side = st.columns([3, 1])
    with col_side:
        st.metric("検出（位置特定済み）", f"{len(detection.spans)} 件")
        st.metric("位置特定できなかった検出", f"{len(detection.not_found)} 件")
    with col_main:
        spans = [
            (s.start, s.end, vote_category("llm", s.ene_type) or "その他")
            for s in detection.spans
        ]
        html = render_masking_html(body_text, spans)
        st.html(
            '<div style="height:400px; overflow:auto; resize:vertical; '
            "line-height:2.2; border:1px solid rgba(128,128,128,0.25); "
            f'border-radius:6px; padding:0.5em;">{html}</div>'
        )

    if detection.spans:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "type(ENE)": s.ene_type,
                        "カテゴリ": vote_category("llm", s.ene_type) or "その他",
                        "テキスト": body_text[s.start : s.end],
                        "位置": f"{s.start}-{s.end}",
                        "一致": s.how,
                        "理由": s.reason or "",
                    }
                    for s in detection.spans
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("LLM の検出はありませんでした。")

    if detection.not_found:
        st.warning(
            f"本文に位置特定できなかった検出が {len(detection.not_found)} 件あります"
            "（綴りのゆれ等。要確認）。"
        )
        st.dataframe(
            pd.DataFrame(
                [{"type(ENE)": t, "テキスト": x} for t, x in detection.not_found]
            ),
            hide_index=True,
            width="stretch",
        )


def render_masking_result(stored: dict) -> None:
    """マスキングの結果表示（保存済み結果から。候補選択・表示切替は再解析しない）。"""
    models = stored["models"]
    dict_path = stored["dict_path"]
    allowlist_path = stored.get("allowlist_path", str(_DEFAULT_ALLOWLIST))
    source_label = stored["source_label"]
    analysis = stored["analysis"]
    chunks = stored["chunks"]

    engine = get_masking_engine(tuple(models), dict_path)

    if analysis.timings:
        total = sum(s for _, s in analysis.timings)
        st.success(_timing_caption(analysis.timings, total, len(chunks)), icon="✅")

    if source_label:
        st.subheader(f"結果: {source_label}")

    # --- マスク単位の切替 ---
    col_unit, col_conf = st.columns([1, 1])
    with col_unit:
        unit = st.radio(
            "マスク単位",
            ["実体ごと（推奨）", "出現ごと（個別に選ぶ）"],
            horizontal=True,
            help="実体ごと=同じ語は文書内の全出現を一括マスク。"
            "出現ごと=各出現を個別に選ぶ（同形異義語＝フランク等の使い分け用）。",
        )
    with col_conf:
        confidences = set(
            st.multiselect(
                "表示する確信度",
                options=_CONFIDENCE_LEVELS,
                default=_CONFIDENCE_DEFAULT,
                help="微弱＝コードらしき誤検出（`Em_NoYes`・`~C02`・`7-410` 等）。既定で非表示。"
                "見たいときは『微弱』を選択（取りこぼし確認用。データは保持されています）。",
            )
        )
    by_entity = unit.startswith("実体")

    # 共有選択（マスクする span 集合）＝ auto（確定/強）＋ 保存済みドラフト（手動の差分）。
    # ドラフトは content_hash 単位で DB に永続化するので、**再起動・再解析でも手動選択が消えない**。
    # 実体ごと/出現ごとの両ビューがこの 1 つの集合を読み書きするので、切替で選択が消えない。
    chash = content_hash(chunks)
    auto = _auto_mask_spans(analysis)
    if "mask_sel" not in stored:
        draft = _ner_cache().get_draft(chash)
        if draft is not None:
            added, removed = draft
            stored["mask_sel"] = (auto | added) - removed  # auto ∪ added − removed
        else:
            stored["mask_sel"] = set(auto)
        stored["mask_ver"] = 0
    sel = stored["mask_sel"]
    ver = stored["mask_ver"]

    if by_entity:
        to_exclude, excl_clicked = _render_by_entity(
            engine, analysis, confidences, sel, ver, stored
        )
    else:
        to_exclude, excl_clicked = _render_by_occurrence(
            engine, analysis, confidences, sel, ver, stored
        )

    # 「除外」チェックを除外リストへ追記し、**再解析なしで**この文書にも即反映する
    # （NER は再実行せず、保存済み解析の候補の confidence を書き換えるだけ）。
    if excl_clicked and to_exclude:
        path = Path(allowlist_path)
        current = load_allowlist_entries(path) if path.exists() else []
        merged = current + [s for s in to_exclude if s not in current]
        save_allowlist_entries(path, merged)
        added = len(merged) - len(current)
        # 現在の解析結果にその場で適用（再解析不要）。署名も更新してバナーを出さない。
        stored["analysis"] = apply_allowlist_to_analysis(
            analysis, MaskAllowlist.load(path)
        )
        # 除外（confidence="除外"）になった span を共有選択から外す（他の手動選択は保持）。
        excluded = {
            (c.start, c.end)
            for c in stored["analysis"].candidates
            if c.confidence == "除外"
        }
        stored["mask_sel"] = {s for s in stored["mask_sel"] if s not in excluded}
        stored["mask_ver"] = stored["mask_ver"] + 1
        stored["settings_sig"] = _masking_settings_sig(
            models, stored["flatten"], dict_path, allowlist_path
        )
        st.success(
            f"除外リストに {added} 件追加し、再解析なしで反映しました（計 {len(merged)} 件）。"
        )
        st.rerun()

    # 手動選択（auto からの差分）を文書単位で永続化（変化時のみ）。再起動/再解析で復元される。
    if stored.get("_draft_saved") != stored["mask_sel"]:
        _ner_cache().save_draft(
            chash, stored["mask_sel"] - auto, auto - stored["mask_sel"]
        )
        stored["_draft_saved"] = set(stored["mask_sel"])

    # 共有選択から結果を作る（ビュー非依存＝sel の span だけをマスク。切替で広がらない）。
    result = _apply_selection(engine, analysis, stored["mask_sel"])

    # --- 結果（色付き表示 / マスク済み / 元テキスト） ---
    col_main, col_side = st.columns([3, 1])
    with col_side:
        st.caption(f"モデル: {', '.join(models)} / 辞書: {len(engine.dictionary)} 表層")
        st.metric("マスク（選択中）", f"{len(result.mapping)} 種")
        st.metric("候補", f"{len(analysis.candidates)} 出現")
        st.metric("チャンク数", f"{len(chunks)} 件")

    with col_main:
        view = st.radio(
            "表示", ["色付き（元文）", "マスク済み", "元テキスト"], horizontal=True
        )
        if view.startswith("色付き"):
            html = render_masking_html(
                result.text, [(c.start, c.end, c.category) for c in result.masked]
            )
            st.html(
                '<div style="height:400px; overflow:auto; resize:vertical; '
                "line-height:2.2; border:1px solid rgba(128,128,128,0.25); "
                f'border-radius:6px; padding:0.5em;">{html}</div>'
            )
        elif view.startswith("マスク"):
            placeholders = {m.placeholder: m.category for m in result.mapping}
            st.html(_readable_text_block(result.masked_text, placeholders=placeholders))
        else:
            st.html(_readable_text_block(result.text))
        st.download_button(
            "⬇ マスク済みテキストをダウンロード",
            result.masked_text,
            file_name="masked.txt",
            mime="text/plain",
        )

    st.subheader(f"対応表（マスク {len(result.mapping)} 種）")
    if result.mapping:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "プレースホルダ": m.placeholder,
                        "カテゴリ": m.category,
                        "原語": " / ".join(m.surfaces),
                    }
                    for m in result.mapping
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("マスク対象が選択されていません。")


def _render_state_header(stored: dict) -> None:
    """選択ソースのパイプライン状態（冒頭設計図のミニ版）。

    ✅=本セッションで実行済み / 📂=キャッシュ有（表示は一瞬） / ⬜=未。保存物（session＋cache.db）から導出。
    """
    chash = content_hash(stored["chunks"])
    flatten = stored.get("flatten", False)
    cache = _ner_cache()

    def mark(in_session: bool, cached: bool) -> str:
        return "✅" if in_session else ("📂" if cached else "⬜")

    want = set(stored.get("models", []))
    cached_models = cache.cached_ner_models(chash, flatten)
    ner = mark("ner" in stored, bool(want) and want.issubset(cached_models))
    llm = mark(
        "llm" in stored, cache.has_llm(chash, LLM_MODEL, flatten, _detector_version())
    )
    draft = cache.get_draft(chash)
    merge = "✅" if "analysis" in stored else "⬜"
    if "analysis" in stored and draft and (draft[0] or draft[1]):
        merge += "（下書きあり）"
    st.caption(
        f"パイプライン状態:　平文 ✅　→　NER検出 {ner}　＋　LLM検出 {llm}　→　"
        f"マージ&確信度 {merge}　→　確定 ⬜"
        "　（✅=本セッション実行 / 📂=キャッシュ有 / ⬜=未）"
    )


def _render_ner_tab(stored: dict, flatten_tables: bool) -> None:
    """NER検出タブ（独立経路）：キャッシュ状態に応じて 実行 / 表示＋再実行 を出し分ける。"""
    chash = content_hash(stored["chunks"])
    cached_models = _ner_cache().cached_ner_models(chash, flatten_tables)
    want = set(stored["models"])
    ner_cached = bool(want) and want.issubset(cached_models)
    in_session = "ner" in stored

    def _run(force: bool) -> None:
        if force:
            _ner_cache().delete_ner(chash)
        with st.spinner("GiNZA で解析中 ...（ステージ進捗が出ます）"):
            stored["ner"] = analyze_masking(
                stored["chunks"],
                stored["models"],
                flatten_tables,
                stored["dict_path"],
                stored["allowlist_path"],
            )
        # マージは NER 票込みで作り直す必要があるので無効化（マージタブで再実行を促す）。
        stored.pop("analysis", None)
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)

    if in_session:
        st.caption("NER 状態: ✅ 実行済み（このセッション）")
        if st.button("🔄 再実行（キャッシュ無視）", key="ner_rerun"):
            _run(force=True)
    elif ner_cached:
        st.caption(
            f"NER 状態: 📂 キャッシュ済み（{_short_models(tuple(sorted(cached_models)))}）"
        )
        c1, c2 = st.columns(2)
        if c1.button("📂 キャッシュの結果を表示", type="primary", key="ner_show"):
            _run(force=False)
        if c2.button("🔄 再実行（キャッシュ無視）", key="ner_rerun"):
            _run(force=True)
    else:
        extra = (
            f"（一部のみキャッシュ: {_short_models(tuple(sorted(cached_models)))}）"
            if cached_models
            else ""
        )
        st.caption(f"NER 状態: ⬜ 未解析{extra}（GiNZA・重い）")
        if st.button("▶ NER 解析を実行", type="primary", key="ner_run"):
            _run(force=False)

    if "ner" not in stored:
        return

    analysis = stored["ner"]
    if analysis.timings:
        total = sum(s for _, s in analysis.timings)
        st.success(
            _timing_caption(analysis.timings, total, len(stored["chunks"])), icon="✅"
        )
    ner_channels = {"ja_ginza", "ja_ginza_electra"}
    cands = [
        c for c in analysis.candidates if any(ch in ner_channels for ch, _ in c.votes)
    ]
    st.caption(
        f"GiNZA(NER) 由来の候補 {len(cands)} 件（独立ビュー）。確信度の確定はマージ&確信度タブで。"
    )
    if not cands:
        st.write("NER 由来の候補はありません。")
        return
    html = render_masking_html(
        analysis.text, [(c.start, c.end, c.category) for c in cands]
    )
    st.html(
        '<div style="height:360px; overflow:auto; resize:vertical; line-height:2.2; '
        "border:1px solid rgba(128,128,128,0.25); border-radius:6px; "
        f'padding:0.5em;">{html}</div>'
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "テキスト": c.surface,
                    "カテゴリ": c.category,
                    "確信度": c.confidence,
                    "ja_ginza": c.vote_labels("ja_ginza"),
                    "electra": c.vote_labels("ja_ginza_electra"),
                }
                for c in cands
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _render_llm_tab(stored: dict, flatten_tables: bool) -> None:
    """LLM検出タブ（独立経路・出口1）：キャッシュ状態に応じて 実行 / 表示＋再実行 を出し分ける。"""
    chash = content_hash(stored["chunks"])
    llm_cached = _ner_cache().has_llm(
        chash, LLM_MODEL, flatten_tables, _detector_version()
    )
    in_session = "llm" in stored

    def _run(force: bool) -> None:
        status = st.empty()

        def _cb(
            i: int, n: int
        ) -> None:  # 窓ごとの進捗（キャッシュヒット時は呼ばれない）
            status.info(f"⏳ LLM 検出中 … 窓 {i + 1}/{n}（{LLM_MODEL} / pii-masker）")

        try:
            t0 = time.perf_counter()
            body_text, detection = run_llm_detection(
                stored["chunks"], flatten_tables, force=force, progress=_cb
            )
            elapsed = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            status.empty()
            st.error(
                f"LLM 検出に失敗しました: {e}\n"
                "（実機で `az login` 済みか、.env の RESOURCE_NAME_GPT41_MINI を確認）"
            )
            return
        status.empty()
        stored["llm"] = {
            "body_text": body_text,
            "detection": detection,
            "elapsed": elapsed,
        }
        # マージは LLM 票込みで作り直す必要があるので無効化（マージタブで再実行を促す）。
        stored.pop("analysis", None)
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)

    if in_session:
        st.caption("LLM 状態: ✅ 実行済み（このセッション）")
        if st.button("🔄 再実行（キャッシュ無視）", key="llm_rerun"):
            _run(force=True)
    elif llm_cached:
        st.caption("LLM 状態: 📂 キャッシュ済み")
        c1, c2 = st.columns(2)
        if c1.button("📂 キャッシュの結果を表示", type="primary", key="llm_show"):
            _run(force=False)
        if c2.button("🔄 再実行（キャッシュ無視）", key="llm_rerun"):
            _run(force=True)
    else:
        st.caption(f"LLM 状態: ⬜ 未実行（{LLM_MODEL} / Azure・要 `az login`）")
        if st.button("▶ LLM 検出を実行", type="primary", key="llm_run"):
            _run(force=False)

    llm = stored.get("llm")
    if not llm:
        return
    if llm.get("elapsed") is not None:
        n = len(llm["detection"].spans)
        st.success(f"⏱ LLM 検出 {llm['elapsed']:.1f}s（{n} 件）", icon="✅")
    render_llm_result(llm["body_text"], llm["detection"])


def _render_merge_tab(stored: dict, flatten_tables: bool) -> None:
    """マージ&確信度タブ（出口2）：辞書＋正規表現に、**実行済み or キャッシュ済み**の
    NER / LLM チャネルを自動合流して候補レビューする（§13・案A）。

    常に：辞書＋正規表現（決定的・軽い。辞書のため Sudachi トークナイズが内部で走る）。
    NER / LLM は「このセッションで実行済み」または「キャッシュ済み」なら**自動で合流**する
    （いずれもキャッシュ参照のみ＝GiNZA / Azure の再実行はしない）。未実行かつ未キャッシュなら
    合流しない（＝勝手に重い処理を走らせない）。状態は実行ボタンの手前に明示する。
    """
    chash = content_hash(stored["chunks"])
    cache = _ner_cache()

    # NER：セッション実行済み or 必要モデルが全てキャッシュ済みなら合流（どちらも GiNZA 再実行なし）。
    ner_in_session = "ner" in stored
    want = set(stored["models"])
    ner_cached = bool(want) and want.issubset(
        cache.cached_ner_models(chash, flatten_tables)
    )
    run_ner = ner_in_session or ner_cached

    # LLM：セッション実行済み or キャッシュ済みなら合流（キャッシュからは Azure 呼び出しなしで読む）。
    llm_in_session = "llm" in stored
    llm_cached = cache.has_llm(chash, LLM_MODEL, flatten_tables, _detector_version())
    has_llm = llm_in_session or llm_cached

    def _src(in_session: bool) -> str:
        return "実行済み" if in_session else "キャッシュ"

    if "analysis" not in stored:
        chans = ["辞書", "正規表現"]
        if run_ner:
            chans.append(f"NER（{_src(ner_in_session)}）")
        if has_llm:
            chans.append(f"LLM（{_src(llm_in_session)}）")
        # キャッシュも実行結果も無いチャネル＝そのままでは合流できない。各タブでの実行を促す。
        missing = []
        if not run_ner:
            missing.append("NER（🔍 NER検出 タブ）")
        if not has_llm:
            missing.append("LLM（🤖 LLM検出 タブ）")
        clicked = st.button("▶ マージ&確信度を実行", type="primary", key="run_merge")
        if not clicked:
            msg = f"『▶ マージ&確信度を実行』で **{' ＋ '.join(chans)}** の票を集約し確信度づけします。"
            if missing:
                msg += (
                    "\n\n" + "・".join(missing) + " はまだ結果がありません。"
                    "合流したい場合は各タブで実行してください（実行後は自動で合流します）。"
                )
            msg += (
                "\n\n確信度：単一チャネル→中／2チャネル以上→強／辞書→確定／"
                "正規表現パターン→強／地名・その他→弱（票数によらず）。"
            )
            st.info(msg)
            return
        # LLM 検出を用意：セッション優先、無ければキャッシュから読む（ヒット＝Azure を呼ばない）。
        det = (stored.get("llm") or {}).get("detection")
        if det is None and llm_cached:
            with st.spinner("LLM 検出をキャッシュから読み込み中 ..."):
                _, det = run_llm_detection(
                    stored["chunks"], flatten_tables, force=False
                )
        with st.spinner("候補を集約・確信度づけ中 ..."):
            stored["analysis"] = analyze_masking(
                stored["chunks"],
                stored["models"],
                flatten_tables,
                stored["dict_path"],
                stored["allowlist_path"],
                llm_detection=det,
                run_ner=run_ner,
            )
        # 実際に合流したチャネルを記録（後続の再描画でキャプションがブレないように）。
        stored["analysis_channels"] = {"ner": run_ner, "llm": has_llm}
        stored.pop("mask_sel", None)
        stored.pop("_draft_saved", None)
        stored["mask_ver"] = 0
    used = stored.get("analysis_channels", {"ner": run_ner, "llm": has_llm})
    st.caption(
        "合流したチャネル: 辞書＋正規表現"
        + ("＋NER" if used.get("ner") else "")
        + ("＋LLM" if used.get("llm") else "")
        + "　— 確信度＝チャネル投票数。LLM 単独は『🤖 LLM検出』タブ（出口1）。"
    )
    render_masking_result(stored)


def _render_pipeline(stored: dict, flatten_tables: bool) -> None:
    """1ソース＝1パイプライン：状態ヘッダー＋ステージ選択（§12）。

    ステージは **st.radio（key で選択状態を保持）** で切り替える。`st.tabs` は各ステージの実行ボタンや
    マスク反映が起こす rerun のたびに先頭タブへ戻ってしまうため使わない（選択が保持できる radio にする）。
    NER検出/LLM検出 は対等な独立経路、マージ&確信度 が合流。
    """
    _render_state_header(stored)
    stage = st.radio(
        "ステージ",
        ["📄 平文", "🔍 NER検出", "🤖 LLM検出", "🔒 マージ&確信度"],
        horizontal=True,
        key="pipeline_stage",
        label_visibility="collapsed",
    )
    st.divider()
    if stage.startswith("📄"):
        _render_extracted_text(stored["chunks"])
    elif stage.startswith("🔍"):
        _render_ner_tab(stored, flatten_tables)
    elif stage.startswith("🤖"):
        _render_llm_tab(stored, flatten_tables)
    else:
        _render_merge_tab(stored, flatten_tables)


def main() -> None:
    st.set_page_config(
        page_title="data-redactor — マスキング", page_icon="🔒", layout="wide"
    )

    # --- 参考ツール（サブ扱い）：NER ビューア ---
    # 本ツールの主機能はマスキング。NER は GiNZA の固有表現を確認する参考ツールとして
    # サイドバーのトグルで開く。トグルを**先に**読み、ON のときは上部の「モード」行を出さない
    # （NER に専念＝無関係なモード選択を見せない）。
    with st.sidebar:
        ner_tool = st.toggle(
            "🔍 NER ビューア（参考）",
            value=False,
            help="主機能はマスキング。これは GiNZA の固有表現を確認する参考ツール"
            "（OFF でマスキングに戻る）。",
        )

    dict_mode = allowlist_mode = cache_mode = False
    if ner_tool:
        masking_mode = False  # モード行は出さず、共通フローを NER 経路で通す
    else:
        mode = st.radio(
            "モード",
            ["🔒 マスキング", "📒 マスク辞書", "🚫 除外リスト", "🗂 キャッシュ"],
            horizontal=True,
        )
        masking_mode = mode.startswith("🔒")
        dict_mode = mode.startswith("📒")
        allowlist_mode = mode.startswith("🚫")
        cache_mode = mode.startswith("🗂")

        # --- キャッシュ一覧モード（解析済み文書の確認・削除） ---
        if cache_mode:
            with st.sidebar:
                st.header("⚙️ 設定")
            st.title("🗂 キャッシュ")
            st.caption(
                "解析（NER）をキャッシュ済みの文書一覧。再解析は NER をスキップして高速になります。"
                "削除すると次回はフル解析に戻ります。**ローカル専用**（`data/cache.db`・git 管理外）。"
            )
            render_cache_view()
            return
        # --- マスク辞書モード（文書入力なし。辞書の確認・編集・保存だけ） ---
        if dict_mode:
            with st.sidebar:
                st.header("⚙️ 設定")
                dict_path = st.text_input("マスク辞書 (YAML)", value=str(_DEFAULT_DICT))
            st.title("📒 マスク辞書")
            st.caption(
                "マスキングで確定マスクする社名・商標・社員名の名簿。"
                "確認・追加・編集・保存ができます。"
            )
            render_dict_editor(dict_path)
            return

        # --- 除外リストモード（文書入力なし。除外語の確認・編集・保存だけ） ---
        if allowlist_mode:
            with st.sidebar:
                st.header("⚙️ 設定")
                allowlist_path = st.text_input(
                    "除外リスト (YAML)", value=str(_DEFAULT_ALLOWLIST)
                )
            st.title("🚫 除外リスト")
            st.caption(
                "マスク**しない**語の名簿。NER の誤検出（社内コード・変数名・汎用語・誤検出メール"
                "など）をここに入れると、以後どの文書でも候補が「除外」へ落ちます。"
                "**辞書（名簿）は上書きしません**（recall 安全。連絡先 regex の誤検出は除外可）。"
            )
            render_allowlist_editor(allowlist_path)
            return

    # --- サイドバー（モード別の設定） ---
    with st.sidebar:
        st.header("⚙️ 設定")
        if masking_mode:
            models = st.multiselect(
                "モデル（併用推奨）",
                options=MODELS,
                default=MODELS,
                format_func=lambda m: f"{m}（{MODEL_DESCRIPTIONS.get(m, '')}）",
            )
            dict_path = st.text_input("マスク辞書 (YAML)", value=str(_DEFAULT_DICT))
            allowlist_path = st.text_input(
                "除外リスト (YAML)",
                value=str(_DEFAULT_ALLOWLIST),
                help="マスクしない語の名簿。一致した検出候補を「除外」へ落とす"
                "（辞書＝名簿は守る／連絡先の誤検出は除外可）。🚫 除外リスト タブで編集。",
            )
            flatten_tables = st.toggle(
                "テーブルを平文化して検出",
                value=True,
                help="表の `|` を句読点に直して**検出精度を上げる**処理（検出専用）。"
                "マスク結果は `|` を含む原文のまま＝セル内の語だけが伏せ字になり、"
                "`|` は区切りとして残ります（出力の体裁を保持）。既定 ON（表が無ければ無影響）。",
            )
        else:
            model_name = st.selectbox(
                "モデル",
                options=MODELS,
                format_func=lambda m: f"{m}（{MODEL_DESCRIPTIONS.get(m, '')}）",
            )
            flatten_tables = st.toggle(
                "テーブルを平文化する",
                value=False,
                help="Markdown テーブルの `|` を除いて平文に変換してから解析します。",
            )
            st.divider()
            st.subheader("🖥️ 表示")
            view_height = st.slider("表示エリアの高さ (px)", 300, 2000, 600, 50)
            font_size = st.slider("文字サイズ (em)", 0.8, 2.0, 1.05, 0.05)

    if masking_mode:
        st.title("🔒 機密情報マスキング")
        st.caption(
            "テキスト入力 / ファイルアップロード / kb-mcp から選択した文書の"
            "機密情報（人名・社名・商標など）を検出してマスクします。"
        )
    else:
        st.title("🔍 NER ビューア（参考）")
        st.caption(
            "**参考ツール**（本機能はマスキング）。テキスト入力 / ファイル / kb-mcp の文書を "
            "GiNZA で固有表現抽出し色付き表示します。サイドバーの『NER ビューア』を OFF で"
            "マスキングに戻ります。"
        )

    # --- 入力（両モード共通。ここでは描画だけ。解析はボタン押下時のみ） ---
    input_mode = st.radio(
        "入力方法",
        [
            "✏️ テキストを入力",
            "📄 ファイルをアップロード",
            "📚 kb-mcp から選択",
            "🗂 キャッシュから選択",
        ],
        horizontal=True,
    )
    input_id, input_kind, source_label, get_chunks = render_input(
        input_mode, flatten_tables
    )

    # 結果は (モード × 入力方法) ごとに別スロットへ保存する。これで入力方法を切り替えると
    # その方法の最後の結果（無ければ案内）が出て、別タブから戻れば元の結果が復元される
    # （テキストで解析→ファイルへ切替えてもテキストの結果が残り続ける、を防ぐ）。
    mode_key = "masking" if masking_mode else "ner"
    slot = f"{mode_key}:{input_kind}"

    # 再解析が必要かは「設定署名」と「入力署名」の 2 本で見る。
    #  - 設定署名（モデル/平文化/辞書 mtime）は**入力が無くても**算出できる。辞書を保存して
    #    別タブから戻ると file_uploader はファイルを失う（Streamlit が非描画ウィジェットの
    #    状態を捨てる）ので入力署名は不明になるが、設定署名は比較でき辞書変更を検知できる。
    #  - 入力署名（input_id）は入力が確定しているときだけ比較する。
    if masking_mode:
        settings_sig: tuple = _masking_settings_sig(
            models, flatten_tables, dict_path, allowlist_path
        )
    else:
        settings_sig = ("ner", model_name, flatten_tables)

    # --- 解析ボタン（テキスト/ファイル/kb-mcp 共通。押したときだけ重い解析が走る） ---
    if masking_mode and not models:
        st.warning("モデルを 1 つ以上選択してください。")
    stored = st.session_state.get(slot)

    # 新しい入力があればそれを解析する（can_fresh）。
    # stored フォールバック（テキスト化済み stored["chunks"] で再解析）は **ファイル入力専用**：
    #   file_uploader だけが別タブ往復で中身を失うため、辞書だけ変えた再解析等で上げ直さずに済む。
    # cache/kb は **選択を fragment 内で行う**ため、行クリックでは外側（このボタン）が再実行されず
    #   選択が反映されない。そこでボタンを選択に依存させず、モデルさえあれば押せるようにし、
    #   未選択のクリックは下のハンドラで案内する（stored への誤フォールバックはしない）。
    models_ok = not (masking_mode and not models)
    can_fresh = get_chunks is not None and models_ok
    can_reuse_stored = input_kind == "file" and stored is not None and models_ok
    # cache/kb は選択を fragment 内で行う＝行クリックでは外側（このボタン）が再実行されず選択が
    # 反映されない。ので選択に依存させず、モデルがあれば押せるようにする（クリック＝本体再実行で
    # 選択が解決される）。未選択のままのクリックは下のハンドラで案内する。
    can_select_list = input_kind in ("cache", "kb") and models_ok
    can_analyze = can_fresh or can_reuse_stored or can_select_list
    # マスキングは「読み込み」（チャンク確定のみ）→各タブで個別実行。NER ビューアは従来どおり即解析。
    action_label = "📥 読み込む" if masking_mode else "🔍 解析する"
    clicked = st.button(action_label, type="primary", disabled=not can_analyze)
    if not can_analyze:  # なぜ押せないかを明示（モデル未選択 / 入力未指定）
        if masking_mode and not models:
            st.caption("⚠ サイドバーでモデルを 1 つ以上選択してください。")
        else:
            st.caption("⚠ 入力（テキスト／ファイル／kb-mcp）を指定すると押せます。")

    # ボタン下の出力（案内 / スピナー / 結果）は 1 つの placeholder に集約する。
    # クリック時にここを描き替えてから解析に入るので、モデルロード等で処理が止まっても
    # 前フレームの「…を押してください」が裏に残って透ける現象が起きない（同一スロットを差し替え）。
    output = st.empty()

    if clicked:
        with (
            output.container()
        ):  # 旧フレームの内容を即座に置換（スピナーをこの位置に出す）
            src_label, in_kind, in_sig = source_label, input_kind, input_id
            if can_fresh:
                try:
                    chunks = get_chunks()  # type: ignore[misc]  # can_fresh で None 除外済み
                except Exception as e:  # noqa: BLE001
                    st.error(f"入力の取得に失敗しました: {e}")
                    chunks = None
            elif can_reuse_stored:
                # ファイル入力で file_uploader が空（別タブ往復でクリア）。同じファイルの
                # テキスト化済みチャンク（stored）を再解析する（file 限定＝別文書の誤解析を防ぐ）。
                src_label = stored["source_label"]  # type: ignore[index]
                in_kind = stored["input_kind"]  # type: ignore[index]
                in_sig = stored["input_sig"]  # type: ignore[index]
                chunks = stored["chunks"]  # type: ignore[index]
            else:
                # cache/kb で未選択のままクリック（選択は一覧の行クリックで行う）。
                st.warning("一覧から行をクリックして文書を選択してください。")
                chunks = None
            if chunks:
                base = {
                    "settings_sig": settings_sig,
                    "input_sig": in_sig,
                    "chunks": chunks,
                    "source_label": src_label,
                    "input_kind": in_kind,
                    "flatten": flatten_tables,
                }
                if masking_mode:
                    # パイプラインは「読み込み」＝チャンク確定のみ。NER/LLM/マージは各タブで個別に実行する。
                    st.session_state[slot] = {
                        **base,
                        "kind": "masking",
                        "models": models,
                        "dict_path": dict_path,
                        "allowlist_path": allowlist_path,
                    }
                    # 文書メタ＋チャンクを記録（「🗂 キャッシュから選択」で入力元に再利用できる）。
                    chash = content_hash(chunks)
                    rec_kind, rec_name = in_kind, src_label or "(無題)"
                    # キャッシュ入力での再解析は「入力方法」が cache なだけで、文書の出所は
                    # 元のまま（file/kb/text）。種別を "cache" で潰さないよう既存メタを保つ。
                    if in_kind == "cache":
                        existing = _ner_cache().get_source(chash)
                        if existing is not None:
                            rec_kind, rec_name = existing
                    _ner_cache().record_document(chash, rec_kind, rec_name, chunks)
                else:
                    result, elapsed = analyze_ner(chunks, model_name, flatten_tables)
                    st.session_state[slot] = {
                        **base,
                        "kind": "ner",
                        "result": result,
                        "model_name": model_name,
                        "elapsed": elapsed,
                    }

    stored = st.session_state.get(slot)  # クリックで更新された可能性があるので取り直す
    if not stored:
        # クリック時はハンドラ側が案内（未選択）やエラーを output に表示済み。上書きしない。
        if not clicked:
            output.info(f"入力を指定して [{action_label}] を押してください。")
        return

    # 解析結果は placeholder の中に描く（クリック時はスピナー表示を結果で置き換える）。
    with output.container():
        # 保存時から設定（辞書/モデル/平文化）か入力が変わっていれば、古い結果を残したまま
        # 再解析を促す。設定は入力が無くても比較できる（辞書保存→別タブ往復で検知できる）。
        # 入力が消えていても stored のチャンクで再解析できるので、ボタンは押せる前提でよい。
        settings_changed = stored.get("settings_sig") != settings_sig
        input_changed = input_id is not None and input_id != stored.get("input_sig")
        if settings_changed or input_changed:
            st.warning(
                f"⚠ 入力／設定が変更されています。最新にするには [{action_label}] を押し直してください"
                "（マスキングは再読み込みで各タブの結果がリセットされます）。"
            )

        if masking_mode:
            # 1ソース＝1パイプライン：平文/NER検出/LLM検出/マージ&確信度 のタブで見せる（§12）。
            #   平文はタブ内に置くので、ここでの inline 表示はしない。
            _render_pipeline(stored, flatten_tables)
        else:
            # NER ビューア（参考）：従来どおり。テキスト化平文を先に出してから結果表示。
            if stored["input_kind"] != "text":
                _render_extracted_text(stored["chunks"])
            render_ner_result(stored, view_height=view_height, font_size=font_size)


if __name__ == "__main__":
    main()
