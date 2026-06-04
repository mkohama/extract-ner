"""
Custom Text Loader
エンコーディング自動検出のタイムアウトを回避するため、優先順位付きでエンコーディングを試行するローダー。
"""

from typing import List
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from src.utils.text_utils import detect_encoding_and_read


class CustomTextLoader(BaseLoader):
    """
    エンコーディング判定を強化したテキストローダー

    src.utils.text_utils を使用して、エンコーディング自動検出のタイムアウトを回避する。
    優先順位: 指定エンコーディング -> 一般的なエンコーディング(UTF-8, Shift-JIS等) -> 自動検出
    """

    def __init__(self, file_path: str, encoding: str | None = None):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:

        try:
            # エンコーディングを判定
            # (detect_encoding_and_read は内容も読み込むが、TextLoaderを使うためにここではエンコーディング名だけ利用する)
            content, detected_encoding = detect_encoding_and_read(
                self.file_path, self.encoding
            )

            # 特殊なフォールバック(utf-8-replace)の場合は、TextLoaderが対応できない可能性があるため
            # 読み込み済みの content から直接 Document を生成する
            if detected_encoding == "utf-8-replace":
                metadata = {"source": self.file_path}
                return [Document(page_content=content, metadata=metadata)]

            # 標準的なエンコーディングの場合は、LangChain標準のTextLoaderに委譲する
            return TextLoader(self.file_path, encoding=detected_encoding).load()

        except Exception as e:
            raise RuntimeError(f"Failed to load text file {self.file_path}: {e}")
