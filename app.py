"""GiNZA 固有表現抽出 Streamlit UI（薄い表示層）。

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力 UI・表示のみを行う。

起動:
    uv run streamlit run app.py
"""

from __future__ import annotations

import html
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
from src.ner import (
    AVAILABLE_MODELS,
    DEFAULT_COLOR,
    MASKING_CATEGORY_COLORS,
    NerEngine,
    build_color_map,
    render_html,
    render_masking_html,
)
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


def _kb_doc_label(meta: dict) -> str:
    """kb-mcp 文書メタから表示名を作る。"""
    name = meta.get("title") or meta.get("file_name") or meta.get("id") or "?"
    path = meta.get("file_path") or ""
    return f"{name}　({path})" if path else str(name)


def render_input(
    input_mode: str,
) -> tuple[tuple | None, str, str, Callable[[], list[str]] | None]:
    """入力ウィジェットを描画し、解析に必要な情報を返す。

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

        idx = st.selectbox(
            "文書を選択",
            options=range(len(docs)),
            format_func=lambda i: _kb_doc_label(docs[i]),
        )
        meta = docs[idx]
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
        idx = st.selectbox(
            "キャッシュ済み文書を選択",
            options=range(len(docs)),
            format_func=lambda i: (
                f"{docs[i].source_name}（{docs[i].chunk_count}チャンク・"
                f"{docs[i].created_at}）"
            ),
        )
        d = docs[idx]
        # チャンク本文を先読み（軽い）。無ければ古いエントリ＝選べないので明示する。
        cached_chunks = _ner_cache().get_chunks(d.content_hash)
        if not cached_chunks:
            st.warning(
                "このキャッシュにはチャンク本文がありません（チャンク保存より前の古いエントリ）。"
                "一度ふつうに解析し直すと、以降ここから選べます。"
            )
            return None, "cache", d.source_name, None
        st.caption(
            "保存チャンクで再解析します。NER はキャッシュにヒットして高速"
            "（辞書・除外リストの変更は反映されます）。"
        )
        return (
            ("cache", d.content_hash),
            "cache",
            d.source_name,
            (lambda c=cached_chunks: c),
        )

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


def _sorted_by_confidence(items, *, key):
    """確信度 降順（確定→強→中→弱）→ 第2キー（表層など）昇順で並べる＝表の既定順。

    items は ``.confidence`` を持つ候補 / 実体。第2キーで同じ表層が隣接する
    （「出現ごと」では同形異義語の比較がしやすくなる）。
    """
    return sorted(
        items, key=lambda it: (_CONFIDENCE_ORDER.get(it.confidence, 9), key(it))
    )


def _render_by_entity(engine, analysis, confidences):
    """実体ごと：同じ語は文書内の全出現を一括マスク。``confidences`` で表示する確信度を絞る。"""
    all_groups = _sorted_by_confidence(
        engine.group_candidates(analysis.candidates), key=lambda g: g.surface
    )
    groups = [g for g in all_groups if g.confidence in confidences]
    hidden = len(all_groups) - len(groups)
    st.subheader(f"マスク候補（{len(groups)} 実体）— チェックで選択")
    cap = "確定/強は初期チェック ON。チェックした実体は**文書内の全出現**がマスクされます。"
    if hidden:
        cap += f"（確信度フィルタで {hidden} 実体を非表示）"
    st.caption(cap)
    table = pd.DataFrame(
        [
            {
                "マスク": g.confidence in AUTO_MASK_CONFIDENCE,
                "除外": g.confidence == "除外",
                "確信度": _confidence_label(g.confidence),
                "カテゴリ": g.category,
                "表層": g.surface,
                "出現": g.count,
                "ja_ginza": g.vote_labels("ja_ginza"),
                "electra": g.vote_labels("ja_ginza_electra"),
                "Sudachi": g.vote_labels("sudachi"),
                "辞書": "○" if g.vote_label("dict") else "",
            }
            for g in groups
        ]
    )
    st.caption(
        "チェックしてから **[✅ マスクを反映]** を押すと結果に反映されます"
        "（チェック中は再描画しません＝画面が先頭に飛びません）。"
    )
    # st.form で囲む：チェックのたびに再実行せず、ボタン押下時だけまとめて適用する
    # （data_editor は編集ごとに rerun し画面が先頭へ飛ぶため。フォームで抑止）。
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
            key="mask_entity",
        )
        col_a, col_b = st.columns([1, 1])
        col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    # 除外チェックした行はマスクしない（除外が優先）。
    selected = [
        m
        for g, on, ex in zip(groups, masks, excludes)
        if on and not ex
        for m in g.members
    ]
    to_exclude = [g.surface for g, ex in zip(groups, excludes) if ex]
    return engine.apply(analysis, selected, expand=True), to_exclude, excl


def _render_by_occurrence(engine, analysis, confidences):
    """出現ごと：各出現を個別にマスク。``confidences`` で表示する確信度を絞る。"""
    all_cands = _sorted_by_confidence(
        list(analysis.candidates), key=lambda c: c.surface
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
    cap += (
        " チェックしてから **[✅ マスクを反映]** を押すと反映されます"
        "（チェック中は再描画しません）。"
    )
    st.caption(cap)
    table = pd.DataFrame(
        [
            {
                "マスク": c.confidence in AUTO_MASK_CONFIDENCE,
                "除外": c.confidence == "除外",
                "確信度": _confidence_label(c.confidence),
                "カテゴリ": c.category,
                "表層": c.surface,
                "文脈": _context(analysis.text, c.start, c.end),
                "ja_ginza": c.vote_labels("ja_ginza"),
                "electra": c.vote_labels("ja_ginza_electra"),
                "Sudachi": c.vote_labels("sudachi"),
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
            key="mask_occurrence",
        )
        col_a, col_b = st.columns([1, 1])
        col_a.form_submit_button("✅ マスクを反映", type="primary")
        excl = col_b.form_submit_button("🚫 選択を除外リストへ")
    masks = edited["マスク"].tolist()
    excludes = edited["除外"].tolist()
    selected = [c for c, on, ex in zip(cands, masks, excludes) if on and not ex]
    to_exclude = [c.surface for c, ex in zip(cands, excludes) if ex]
    return engine.apply(analysis, selected, expand=False), to_exclude, excl


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
        }
        for e in entries
    ]
    df = pd.DataFrame(rows, columns=["カテゴリ", "代表表記", "別名", "置換"])
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

    def _short_models(models: tuple[str, ...]) -> str:
        names = {"ja_ginza_electra": "electra", "ja_ginza": "ginza"}
        return ", ".join(names.get(m, m) for m in models)

    st.caption(f"キャッシュ済み: {len(docs)} 文書")
    df = pd.DataFrame(
        [
            {
                "削除": False,
                "ソース": d.source_name,
                "種別": d.source_kind,
                "チャンク": d.chunk_count,
                "文字数": d.char_count,
                "モデル": _short_models(d.models),
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
        disabled=[c for c in df.columns if c != "削除"],
        column_config={"削除": st.column_config.CheckboxColumn("削除")},
        key="cache_view",
    )
    to_delete = [d for d, on in zip(docs, edited["削除"].tolist()) if on]
    if to_delete and st.button(
        f"🗑 選択した {len(to_delete)} 件のキャッシュを削除", type="primary"
    ):
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
):
    """マスキング検出（重い）。ボタン押下時のみ呼ぶ。"""
    engine = get_masking_engine(tuple(models), dict_path)
    allowlist = _load_allowlist(allowlist_path)
    status = st.empty()
    analysis = engine.analyze(
        chunks,
        flatten_tables=flatten_tables,
        allowlist=allowlist,
        ner_cache=_ner_cache(),
        progress=_stage_callback(status, len(chunks)),
    )
    status.empty()
    return analysis


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

    if by_entity:
        result, to_exclude, excl_clicked = _render_by_entity(
            engine, analysis, confidences
        )
    else:
        result, to_exclude, excl_clicked = _render_by_occurrence(
            engine, analysis, confidences
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
        stored["settings_sig"] = _masking_settings_sig(
            models, stored["flatten"], dict_path, allowlist_path
        )
        st.success(
            f"除外リストに {added} 件追加し、再解析なしで反映しました（計 {len(merged)} 件）。"
        )
        st.rerun()

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
    input_id, input_kind, source_label, get_chunks = render_input(input_mode)

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

    # 再解析はテキスト化済みチャンク（stored["chunks"]）を使い回せる＝辞書だけ変えたとき等は
    # ファイルを上げ直す必要がない。別タブ往復で file_uploader が中身を失っても、stored が
    # あれば押せる（その設定で再解析）。新しい入力があればそちらを優先する。
    models_ok = not (masking_mode and not models)
    can_fresh = get_chunks is not None and models_ok
    can_analyze = can_fresh or (stored is not None and models_ok)
    clicked = st.button("🔍 解析する", type="primary", disabled=not can_analyze)

    # ボタン下の出力（案内 / スピナー / 結果）は 1 つの placeholder に集約する。
    # クリック時にここを描き替えてから解析に入るので、モデルロード等で処理が止まっても
    # 前フレームの「…を押してください」が裏に残って透ける現象が起きない（同一スロットを差し替え）。
    output = st.empty()

    if clicked:
        with (
            output.container()
        ):  # 旧フレームの内容を即座に置換（スピナーをこの位置に出す）
            if can_fresh:
                src_label, in_kind, in_sig = source_label, input_kind, input_id
                try:
                    chunks = get_chunks()  # type: ignore[misc]  # can_fresh で None 除外済み
                except Exception as e:  # noqa: BLE001
                    st.error(f"入力の取得に失敗しました: {e}")
                    chunks = None
            else:
                # 入力ウィジェットが空（往復でクリア等）。テキスト化済みチャンクを再解析する。
                src_label = stored["source_label"]  # type: ignore[index]
                in_kind = stored["input_kind"]  # type: ignore[index]
                in_sig = stored["input_sig"]  # type: ignore[index]
                chunks = stored["chunks"]  # type: ignore[index]
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
                    analysis = analyze_masking(
                        chunks, models, flatten_tables, dict_path, allowlist_path
                    )
                    st.session_state[slot] = {
                        **base,
                        "kind": "masking",
                        "analysis": analysis,
                        "models": models,
                        "dict_path": dict_path,
                        "allowlist_path": allowlist_path,
                    }
                    # 文書メタ＋チャンクを記録（NER 層は engine 側で自動保存済み）。
                    # チャンクも保存＝「🗂 キャッシュから選択」で入力元に再利用できる。
                    _ner_cache().record_document(
                        content_hash(chunks),
                        in_kind,
                        src_label or "(無題)",
                        chunks,
                    )
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
        output.info("入力を指定して [🔍 解析する] を押してください。")
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
                "⚠ 入力／設定が変更されています。"
                "最新の結果にするには [🔍 解析する] を押してください。"
            )

        # ファイル/kb-mcp はテキスト化を経るので、解析した平文を確認できるようにする。
        if stored["input_kind"] != "text":
            _render_extracted_text(stored["chunks"])

        if masking_mode:
            render_masking_result(stored)
        else:
            render_ner_result(stored, view_height=view_height, font_size=font_size)


if __name__ == "__main__":
    main()
