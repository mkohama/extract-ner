"""
pdfminer.six を使用した PDF ローダー

[pypdf から pdfminer.six へ移行した理由]
日本語PDFの一部（特に古い形式）には、フォント内の文字ID（CID）から
Unicode への変換テーブル（ToUnicode CMap）が埋め込まれていないものがある。

pypdf はこの変換テーブルがないと CID をそのまま Unicode コードポイントとして
解釈するため、日本語がチベット文字やオリヤー文字などに化けてしまう。

pdfminer.six は Adobe-Japan1 用の CMap を内蔵しており、変換テーブルがない
PDF でも CID → Unicode 変換を正しく行えるため、文字化けが発生しない。
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.logger import log


class PdfLoader:
    """
    PDFファイルからテキストを抽出するローダー

    pdfminer.six を使用してページごとにテキストを抽出し、
    ページ単位の LangChain Document リストとして返す。
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Document]:
        """
        PDFファイルを読み込み、ページごとの Document リストとして返す

        Returns:
            Document のリスト（ページごとに1つ、page メタデータ付き）
        """
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer

            documents: List[Document] = []

            for i, page_layout in enumerate(
                extract_pages(str(self.file_path)), start=0
            ):
                texts = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        texts.append(element.get_text())
                text = "".join(texts).strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(self.file_path),
                                "file_type": "pdf",
                                "page": i,
                            },
                        )
                    )

            if not documents:
                log(f"PDF file {self.file_path} has no extractable text")

            return documents

        except ImportError:
            raise RuntimeError(
                "pdfminer.six がインストールされていません。"
                "'uv add pdfminer.six' を実行してください。"
            )
        except Exception as e:
            log(f"Error processing PDF file {self.file_path}: {e}")
            raise RuntimeError(f"Failed to process PDF file: {e}") from e
