"""ファイルを入力ソースとして扱うアダプタ。

kb-mcp から移植した DocumentLoader でファイルをテキスト化する。
"""

from __future__ import annotations

from pathlib import Path

from src.core.document.document_loader import DocumentLoader


def load_text_from_file(file_path: str | Path) -> str:
    """ファイルをテキスト化して返す。

    PDF / Excel / PowerPoint などは複数の Document に分割されて返るため、
    page_content を結合して 1 つのテキストにまとめる。
    """
    docs = DocumentLoader().load_document(file_path)
    return "\n\n".join(doc.page_content for doc in docs)
