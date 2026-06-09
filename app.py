"""GiNZA 固有表現抽出 Streamlit UI（薄い表示層）。

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力 UI・表示のみを行う。

起動:
    uv run streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.document.document_loader import DocumentLoader
from src.masking import AUTO_MASK_CONFIDENCE, MaskDictionary, MaskingEngine
from src.ner import (
    AVAILABLE_MODELS,
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


# マスク辞書の既定パス（ルート直下 data/mask_dict.yaml）
_DEFAULT_DICT = Path(__file__).resolve().parent / "data" / "mask_dict.yaml"


@st.cache_resource(show_spinner="モデル・マスク辞書を読み込み中 ...")
def get_masking_engine(models: tuple[str, ...], dict_path: str) -> MaskingEngine:
    """マスキングエンジンを生成・キャッシュする（モデル・辞書もここでロード）。"""
    dictionary = (
        MaskDictionary.load(dict_path)
        if dict_path and Path(dict_path).exists()
        else MaskDictionary.empty()
    )
    engine = MaskingEngine(dictionary=dictionary, models=list(models))
    for e in engine.engines:
        _ = e.nlp  # 先にロード
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


def get_input_chunks(input_mode: str) -> tuple[list[str] | None, str]:
    """入力方法に応じて解析対象チャンク（テキストのリスト）とソース名を返す。"""
    if input_mode.startswith("📄"):
        uploaded_file = st.file_uploader(
            f"対応形式: {', '.join(SUPPORTED_EXTENSIONS)}",
            type=SUPPORTED_EXTENSIONS,
        )
        if uploaded_file is not None:
            with st.spinner("ファイルをテキスト化・チャンク化中 ..."):
                try:
                    return (
                        extract_chunks_from_upload(uploaded_file),
                        uploaded_file.name,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"ファイルの読み込みに失敗しました: {e}")
        return None, ""

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
            return None, ""
        if not docs:
            st.warning("kb-mcp に登録された文書がありません。")
            return None, ""

        idx = st.selectbox(
            "文書を選択",
            options=range(len(docs)),
            format_func=lambda i: _kb_doc_label(docs[i]),
        )
        meta = docs[idx]
        doc_id = meta.get("id") or meta.get("document_id")

        # 明示的にボタンを押したときだけダウンロード＆解析する
        if st.button(
            "選択した文書をダウンロードして解析",
            type="primary",
            disabled=not doc_id,
        ):
            try:
                with st.spinner("本文を取得中 ..."):
                    st.session_state["kb_chunks"] = fetch_kb_document_chunks(
                        url, doc_id
                    )
                    st.session_state["kb_source"] = _kb_doc_label(meta)
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("kb_chunks", None)
                st.error(f"本文の取得に失敗しました: {e}")

        # 取得済みの本文があれば返す (ラベル絞り込み等の再実行でも保持される)
        if st.session_state.get("kb_chunks"):
            return st.session_state["kb_chunks"], st.session_state.get("kb_source", "")
        st.info("文書を選択して [ダウンロードして解析] を押してください。")
        return None, ""

    # テキスト入力（単一チャンクとして扱う。長文でもエンジン側で安全分割される）
    input_text = st.text_area("解析するテキスト", value=SAMPLE_TEXT, height=200)
    if input_text.strip():
        return [input_text], "入力テキスト"
    return None, ""


def render_ner(
    chunks: list[str],
    source_label: str,
    *,
    model_name: str,
    flatten_tables: bool,
    view_height: int,
    font_size: float,
) -> None:
    """固有表現抽出（NER）モードの表示。"""
    engine = get_engine(model_name)
    colors = build_color_map(engine.available_labels())

    with st.spinner(f"固有表現を抽出中 ...（{len(chunks)} チャンク）"):
        result = engine.extract_chunks(chunks, flatten_tables=flatten_tables)

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


# 確信度の並び順（確定→強→中→弱）。文字列順だと「中→確定」になるので明示する。
_CONFIDENCE_ORDER = {"確定": 0, "強": 1, "中": 2, "弱": 3}


def _confidence_label(confidence: str) -> str:
    """並び順の番号を前置した表示用ラベル（例 '1 確定'）。

    列ヘッダで文字列ソートしても 確定→強→中→弱 の順になるようにする
    （番号なしだと文字コード順で「中→確定」になってしまう）。1=確定 … 4=弱。
    """
    return f"{_CONFIDENCE_ORDER.get(confidence, 9) + 1} {confidence}"


def _sorted_by_confidence(items, *, key):
    """確信度 降順（確定→強→中→弱）→ 第2キー（表層など）昇順で並べる＝表の既定順。

    items は ``.confidence`` を持つ候補 / 実体。第2キーで同じ表層が隣接する
    （「出現ごと」では同形異義語の比較がしやすくなる）。
    """
    return sorted(
        items, key=lambda it: (_CONFIDENCE_ORDER.get(it.confidence, 9), key(it))
    )


def _render_by_entity(engine, analysis):
    """実体ごと：同じ語は文書内の全出現を一括マスク。"""
    groups = _sorted_by_confidence(
        engine.group_candidates(analysis.candidates), key=lambda g: g.surface
    )
    st.subheader(f"マスク候補（{len(groups)} 実体）— チェックで選択")
    st.caption(
        "確定/強は初期チェック ON。チェックした実体は**文書内の全出現**がマスクされます。"
    )
    table = pd.DataFrame(
        [
            {
                "マスク": g.confidence in AUTO_MASK_CONFIDENCE,
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
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[c for c in table.columns if c != "マスク"],
        column_config={"マスク": st.column_config.CheckboxColumn("マスク")},
        key="mask_entity",
    )
    selected = [
        m for g, on in zip(groups, edited["マスク"].tolist()) if on for m in g.members
    ]
    return engine.apply(analysis, selected, expand=True)


def _render_by_occurrence(engine, analysis):
    """出現ごと：各出現を個別にマスク（同形異義語の使い分け用）。展開せず選んだ箇所だけ。"""
    cands = _sorted_by_confidence(list(analysis.candidates), key=lambda c: c.surface)
    st.subheader(f"マスク候補（{len(cands)} 出現）— 出現ごとに選択")
    st.caption(
        "各出現を個別にマスク（フランク=人名 vs フランクに=気軽に、等を文脈で使い分け）。"
        "**選んだ出現だけ**マスクし、他の出現には広げません。"
    )
    table = pd.DataFrame(
        [
            {
                "マスク": c.confidence in AUTO_MASK_CONFIDENCE,
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
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[c for c in table.columns if c != "マスク"],
        column_config={"マスク": st.column_config.CheckboxColumn("マスク")},
        key="mask_occurrence",
    )
    selected = [c for c, on in zip(cands, edited["マスク"].tolist()) if on]
    return engine.apply(analysis, selected, expand=False)


def render_masking(
    chunks: list[str],
    source_label: str,
    *,
    models: list[str],
    flatten_tables: bool,
    dict_path: str,
) -> None:
    """マスキングモードの表示。候補をチェックで選び（確定/強は初期 ON）、選んだ分をマスクする。"""
    engine = get_masking_engine(tuple(models), dict_path)
    with st.spinner(f"検出中 ...（{len(chunks)} チャンク）"):
        analysis = engine.analyze(chunks, flatten_tables=flatten_tables)

    if source_label:
        st.subheader(f"結果: {source_label}")

    # --- マスク単位の切替 ---
    unit = st.radio(
        "マスク単位",
        ["実体ごと（推奨）", "出現ごと（個別に選ぶ）"],
        horizontal=True,
        help="実体ごと=同じ語は文書内の全出現を一括マスク。"
        "出現ごと=各出現を個別に選ぶ（同形異義語＝フランク等の使い分け用）。",
    )
    by_entity = unit.startswith("実体")

    if by_entity:
        result = _render_by_entity(engine, analysis)
    else:
        result = _render_by_occurrence(engine, analysis)

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
        else:
            shown = result.masked_text if view.startswith("マスク") else result.text
            st.text_area(
                "テキスト",
                value=shown,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )
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
        page_title="GiNZA NER / マスキング", page_icon="🔒", layout="wide"
    )

    mode = st.radio(
        "モード",
        ["🔍 固有表現抽出 (NER)", "🔒 マスキング"],
        horizontal=True,
    )
    masking_mode = mode.startswith("🔒")

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
        st.title("🔍 GiNZA 固有表現抽出 (NER)")
        st.caption(
            "テキスト入力 / ファイルアップロード / kb-mcp から選択すると、"
            "GiNZA で固有表現を抽出して色付きで表示します。"
        )

    # --- 入力（両モード共通） ---
    input_mode = st.radio(
        "入力方法",
        ["✏️ テキストを入力", "📄 ファイルをアップロード", "📚 kb-mcp から選択"],
        horizontal=True,
    )
    chunks, source_label = get_input_chunks(input_mode)
    if not chunks:
        return

    if masking_mode:
        if not models:
            st.warning("モデルを 1 つ以上選択してください。")
            return
        render_masking(
            chunks,
            source_label,
            models=models,
            flatten_tables=flatten_tables,
            dict_path=dict_path,
        )
    else:
        render_ner(
            chunks,
            source_label,
            model_name=model_name,
            flatten_tables=flatten_tables,
            view_height=view_height,
            font_size=font_size,
        )


if __name__ == "__main__":
    main()
