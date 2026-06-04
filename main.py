"""GiNZA 固有表現抽出 CLI（薄い表示層）。

実際の抽出は src.ner.NerEngine が担当する。本ファイルは入力の取得・引数処理・
コンソール出力・displaCy への受け渡しだけを行う。

使い方:
    uv run main.py                       # サンプルテキストを解析して ner.html を生成
    uv run main.py --text "解析したい文章"
    uv run main.py --input report.pdf    # 任意のファイルをテキスト化して解析
    uv run main.py --input data.xlsx --open
    uv run main.py --model ja_ginza      # 軽量モデルに切り替え (既定は ja_ginza_electra)
    uv run main.py --labels Person Company  # 指定カテゴリのみ抽出
    uv run main.py --flatten             # Markdown テーブルを平文化して解析
    uv run main.py --serve               # ブラウザで http://localhost:5000 を開いて表示
    uv run main.py --open                # 生成した HTML を既定ブラウザで開く
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from spacy import displacy

from src.ner import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    NerEngine,
    build_color_map,
    render_html,
    to_displacy_data,
)
from src.sources import SAMPLE_TEXT, load_text_from_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GiNZA で固有表現抽出を実行し displaCy で表示する"
    )
    parser.add_argument("--text", help="解析するテキストを直接指定")
    parser.add_argument(
        "--input",
        type=Path,
        help="解析するファイルのパス (.txt/.md/.pdf/.docx/.xlsx/.pptx/.html/.xml など)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("ner.html"), help="出力する HTML ファイル名"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(AVAILABLE_MODELS),
        help="使用する GiNZA モデル (既定: ja_ginza_electra は高精度・低速。ja_ginza は軽量・高速)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="抽出するカテゴリ（ラベル）を限定する。例: --labels Person Company",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Markdown テーブルを平文化してから解析する (既定: そのまま解析)",
    )
    parser.add_argument(
        "--serve", action="store_true", help="ブラウザ表示用のサーバーを起動する"
    )
    parser.add_argument(
        "--open", action="store_true", help="生成した HTML を既定ブラウザで開く"
    )
    args = parser.parse_args()

    # --- 入力テキストの取得 ---
    if args.text:
        text = args.text
    elif args.input:
        print(f"ファイルをテキスト化中: {args.input}")
        text = load_text_from_file(args.input)
    else:
        text = SAMPLE_TEXT

    # --- エンジンで抽出 ---
    print(f"GiNZA モデル ({args.model}) を読み込み中 ...")
    engine = NerEngine(args.model)
    result = engine.extract(text, labels=args.labels, flatten_tables=args.flatten)

    # --- コンソール出力 ---
    print(f"\n抽出された固有表現: {len(result.entities)} 件\n")
    print(f"{'テキスト':<16}{'ラベル':<22}{'開始':>5}{'終了':>5}")
    print("-" * 50)
    for ent in result.entities:
        print(f"{ent.text:<16}{ent.label:<22}{ent.start:>5}{ent.end:>5}")

    # --- displaCy 表示 ---
    # 色は（フィルタに関わらず安定させるため）モデルの全ラベルから作る
    colors = build_color_map(engine.available_labels())

    if args.serve:
        print("\nhttp://localhost:5000 で表示します (Ctrl+C で終了)")
        displacy.serve(
            to_displacy_data(result),
            style="ent",
            manual=True,
            options={"colors": colors},
            auto_select_port=True,
        )
        return

    html = render_html(result, colors, page=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"\nHTML を書き出しました: {args.output.resolve()}")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
