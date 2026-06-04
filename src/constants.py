"""
定数定義

kb-mcp のテキスト変換部 (DocumentLoader) が参照する定数のみを移植。
"""

# ============================================================================
# サポートされるドキュメント拡張子
# ============================================================================

# Note: DocumentLoader.LOADER_MAPPING と同期を保つこと
SUPPORTED_DOCUMENT_EXTENSIONS: set[str] = {
    # Text
    ".txt",
    ".md",
    # PDF
    ".pdf",
    # Microsoft Excel
    ".xlsx",
    ".xlsm",
    ".xls",
    # Microsoft PowerPoint
    ".pptx",
    # Microsoft Word
    ".docx",
    # Web/Data formats
    ".html",
    ".xml",
}
