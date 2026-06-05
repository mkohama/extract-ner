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
from src.ner import AVAILABLE_MODELS, NerEngine, build_color_map, render_html
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


def main() -> None:
    st.set_page_config(page_title="GiNZA 固有表現抽出", page_icon="🔍", layout="wide")
    st.title("🔍 GiNZA 固有表現抽出 (NER)")
    st.caption(
        "ドキュメントをアップロード / テキスト入力 / kb-mcp から選択すると、"
        "GiNZA で固有表現を抽出して色付きで表示します。"
    )

    # --- モデル選択・前処理設定 ---
    with st.sidebar:
        st.header("⚙️ 設定")
        model_name = st.selectbox(
            "モデル",
            options=MODELS,
            format_func=lambda m: f"{m}（{MODEL_DESCRIPTIONS.get(m, '')}）",
        )
        flatten_tables = st.toggle(
            "テーブルを平文化する",
            value=False,
            help="Markdown テーブルの `|` を除いて平文に変換してから解析します。"
            "OFF（既定）では元のテキストをそのまま解析します。",
        )

        st.divider()
        st.subheader("🖥️ 表示")
        view_height = st.slider(
            "表示エリアの高さ (px)",
            min_value=300,
            max_value=2000,
            value=600,
            step=50,
            help="固有表現のハイライト表示エリアの高さ。枠の右下をドラッグしても伸縮できます。",
        )
        font_size = st.slider(
            "文字サイズ (em)",
            min_value=0.8,
            max_value=2.0,
            value=1.05,
            step=0.05,
        )

    engine = get_engine(model_name)
    colors = build_color_map(engine.available_labels())

    # --- 入力 ---
    # ラジオで入力方法を 1 つだけ有効にする（タブだと両方生きて結果が混ざるため）
    input_mode = st.radio(
        "入力方法",
        ["✏️ テキストを入力", "📄 ファイルをアップロード", "📚 kb-mcp から選択"],
        horizontal=True,
    )

    chunks, source_label = get_input_chunks(input_mode)

    if not chunks:
        return

    # --- 解析（エンジンに委譲。全カテゴリを抽出し、表示側で絞り込む） ---
    with st.spinner(f"固有表現を抽出中 ...（{len(chunks)} チャンク）"):
        result = engine.extract_chunks(chunks, flatten_tables=flatten_tables)

    if source_label:
        st.subheader(f"解析結果: {source_label}")

    # --- 表示するカテゴリ（ラベル）の選択 ---
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

        # ラベルごとの件数 (全件)
        if result.entities:
            counts = pd.Series([e.label for e in result.entities]).value_counts()
            st.write("**ラベル別件数**")
            st.dataframe(
                counts.rename_axis("ラベル").reset_index(name="件数"),
                hide_index=True,
                width="stretch",
            )

    with col_main:
        # ハイライト表示 (st.html で静的 HTML をインライン描画。高さ・文字サイズは
        # サイドバーのスライダーで可変。さらに resize:vertical で枠をドラッグ伸縮できる)
        html = render_html(shown, colors)
        st.html(
            f'<div style="height:{view_height}px; overflow:auto; resize:vertical; '
            "line-height:2.2; border:1px solid rgba(128,128,128,0.25); "
            f'border-radius:6px; padding:0.5em; font-size:{font_size}em;">{html}</div>'
        )

    # --- 抽出一覧 ---
    st.subheader("固有表現の一覧")
    if shown.entities:
        rows = [
            {
                "テキスト": ent.text,
                "ラベル": ent.label,
                "開始": ent.start,
                "終了": ent.end,
            }
            for ent in shown.entities
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("表示対象の固有表現がありません。")


if __name__ == "__main__":
    main()
