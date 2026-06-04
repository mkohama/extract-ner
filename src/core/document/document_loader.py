"""
Document Loader
様々な形式のドキュメントを読み込む
"""

from pathlib import Path
from typing import List, Dict, Callable

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
    # UnstructuredExcelLoader,
    # UnstructuredMarkdownLoader,
    # UnstructuredPowerPointLoader,
    # UnstructuredWordDocumentLoader,
)
from src.core.document.loaders.excel_loader import ExcelToMarkdownLoader
from src.core.document.loaders.powerpoint_loader import PowerPointLoader
from src.core.document.loaders.word_loader import WordToMarkdownLoader
from src.core.document.loaders.custom_text_loader import CustomTextLoader
from src.core.document.loaders.pdf_loader import PdfLoader
from src.constants import SUPPORTED_DOCUMENT_EXTENSIONS
from src.utils.text_utils import normalize_nfkc


class DocumentLoader:
    """ドキュメントの読み込みを担当"""

    # fmt: off
    LOADER_MAPPING = {
        # Text
        ".txt": CustomTextLoader,

        # Markdown
        # (UnstructuredMarkdownLoaderは使用せず、単純なテキストとして読んだ後、独自のチャンク分割処理を適用)
        # ".md": UnstructuredMarkdownLoader,
        ".md": CustomTextLoader,

        # PDF
        # (PyPDFLoaderはToUnicode CMapがない日本語PDFで文字化けするため、pdfminer.sixベースのローダーを使用)
        ".pdf": PdfLoader,

        # Microsoft Excel
        # (UnstructuredExcelLoaderは使用せず、独自の読み込み処理を適用)
        # ".xlsx": UnstructuredExcelLoader
        # ".xls": UnstructuredExcelLoader
        ".xlsx": ExcelToMarkdownLoader,
        ".xlsm": ExcelToMarkdownLoader,
        ".xls": ExcelToMarkdownLoader,

        # Microsoft PowerPoint
        # (UnstructuredPowerPointLoaderはWindows環境での依存関係の問題があるため、独自のローダーを使用)
        # (ppt を扱うには LibreOffice が必要となるが、依存したくないため、ppt には対応しない)
        ".pptx": PowerPointLoader,
        # ".ppt": UnstructuredPowerPointLoader,

        # Microsoft Word
        # (UnstructuredWordLoaderはWindows環境での依存関係の問題があるため、独自のローダーを使用)
        # (doc を扱うには LibreOffice が必要となるが、依存したくないため、doc には対応しない)
        ".docx": WordToMarkdownLoader,
        # ".doc": UnstructuredWordDocumentLoader,

        # Web/Data formats
        ".html": UnstructuredHTMLLoader,
        ".xml": UnstructuredXMLLoader,
    }
    # fmt: on

    # 軽量モジュールから参照（UI との共有用）
    SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS

    def __init__(self, loader_mapping: Dict[str, Callable] | None = None):
        self.loader_mapping = loader_mapping or self.LOADER_MAPPING.copy()

    def add_custom_loader(self, extension: str, loader: Callable) -> None:
        """カスタムローダーを追加"""
        self.loader_mapping[extension] = loader

    def load_document(self, file_path: str | Path) -> List[Document]:
        """単一ファイルを読み込む

        Args:
            file_path: 読み込むファイルのパス

        Returns:
            読み込まれたドキュメントのリスト

        Raises:
            ValueError: サポートされていないファイル形式の場合
            FileNotFoundError: ファイルが存在しない場合
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in self.loader_mapping:
            raise ValueError(
                f"サポートされていないファイル形式: {extension}\n"
                f"サポート形式: {list(self.loader_mapping.keys())}"
            )

        loader_class = self.loader_mapping[extension]
        loader = loader_class(str(file_path))
        docs = loader.load()

        for doc in docs:
            # NFKC正規化を適用（全フォーマット共通）
            doc.page_content = normalize_nfkc(doc.page_content)
            # 基本的なメタデータを追加
            doc.metadata.update(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": extension[1:],  # '.' を除く
                }
            )

        return docs
