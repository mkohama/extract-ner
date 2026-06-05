"""後方互換のためのエントリ。実体は src.cli（統一サブコマンド CLI）。

推奨は ``uv run extract-ner <サブコマンド>``。本ファイル経由でも同じことができる::

    uv run main.py ui                 # Streamlit UI を起動
    uv run main.py ner <file> --open  # ファイル/テキストを NER → HTML 表示
    uv run main.py debug <file>       # トークンの品詞 / NER ラベルを観察
    uv run main.py check              # 品質ゲート（ruff + mypy）
"""

from __future__ import annotations

from src.cli import main

if __name__ == "__main__":
    main()
