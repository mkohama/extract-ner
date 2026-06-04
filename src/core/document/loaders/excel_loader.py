"""
Excel to Markdown Loader
1. ExcelFileによるシートごとの読み込み
2. 数値フォーマットの正規化 (:.g -> :.8g に修正)
3. マルチレベルヘッダー対応
4. クリーニング品質向上 (全角スペース対応)
5. メタデータキーを file_type から source_type に変更
6. ベクトルDB格納用にMarkdown圧縮機能を追加（約30-40%のサイズ削減）
"""

from typing import List, Union
import pandas as pd
from langchain_core.documents import Document
from src.logger import log


class ExcelToMarkdownLoader:
    """
    Excel ファイルを Markdown 形式で読み込むローダー
    各シートを個別の Document として扱い、大規模ファイル処理とデータ正規化に対応。
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Excel ファイルのパス
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Excel ファイルを読み込み、各シートを Markdown 化して Document のリストとして返す

        Returns:
            Document のリスト（各シートが1つの Document）
        """
        documents = []

        try:
            # 1. pd.ExcelFileを使用してメモリ効率良くシートを順次読み込む
            with pd.ExcelFile(self.file_path) as xls:
                sheet_names = xls.sheet_names

                for sheet_name in sheet_names:
                    # シートごとに読み込み
                    df = pd.read_excel(xls, sheet_name=sheet_name)

                    # DataFrameのクリーニングとデータ正規化
                    cleaned_df = self._clean_dataframe(df)

                    if cleaned_df.empty:
                        log(
                            f"Sheet '{sheet_name}' in {self.file_path} is empty after cleaning. Skipping."
                        )
                        continue

                    # Markdown 形式に変換
                    # tablefmt="pipe" は標準的。必要に応じて tablefmt="github" も検討可能
                    markdown_content = cleaned_df.to_markdown(
                        index=False, tablefmt="pipe"
                    )

                    # ベクトルDB格納用に空白を除去して圧縮
                    markdown_content = self._compact_markdown_table(markdown_content)

                    # Document として作成
                    doc = Document(
                        page_content=markdown_content,
                        metadata={
                            "source": self.file_path,
                            "sheet_name": sheet_name,
                            # "source_type": "xlsx_md",
                            # "row_count": len(cleaned_df),
                            # "column_count": len(cleaned_df.columns),
                            # "columns": list(cleaned_df.columns),
                        },
                    )

                    documents.append(doc)

        except FileNotFoundError:
            log(f"Error: File not found at {self.file_path}")
            raise  # 例外を再raise

        except Exception as e:
            log(f"Error processing Excel file {self.file_path}: {e}")
            raise RuntimeError(f"Failed to process Excel file: {e}") from e

        return documents

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame をクリーニングし、数値フォーマットを正規化
        """
        df = df.copy()

        # 1. 列名の処理 (マルチレベルヘッダー対応を含む)
        df.columns = self._clean_column_names(df.columns)

        # 2. NaN を空文字列に置換
        # この時点で fillna を行うことで、後続の map での NaN エラーを防ぐ
        df = df.fillna("")

        # 3. 数値フォーマットの正規化: 数値を含む可能性のある列を処理
        for col in df.columns:
            # 元のDtypeが数値系であれば正規化を試みる
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                df[col] = df[col].apply(
                    # 修正: :.g は桁数指定がないため、:.8g (有効数字8桁) に修正
                    lambda x: f"{x:.8g}" if pd.notna(x) and x != "" else ""
                )

        # 4. すべての値を文字列に変換し、前後の空白を削除
        # NFKC正規化は DocumentLoader.load_document() で一元的に適用される
        df = df.map(lambda x: str(x).strip())

        # 5. セル内の改行を空白に置き換え（Markdown表の行構造を維持）
        # pandas.to_markdown() はセル内の \n を複数行に分割するため、
        # 1つのセルとして保持するために改行を空白に置換する
        df = df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)

        # 6. 完全に空の行を削除（すべてのセルが空文字列('')の行を削除）
        df = df[df.apply(lambda row: any(cell != "" for cell in row), axis=1)]

        # 7. 完全に空の列を削除
        df = df.loc[:, df.apply(lambda col: any(cell != "" for cell in col), axis=0)]

        # 8. インデックスをリセット
        df = df.reset_index(drop=True)

        return df

    def _clean_column_names(self, columns: Union[pd.Index, pd.MultiIndex]) -> List[str]:
        """
        列名のクリーニングとマルチレベルヘッダーに対応
        """
        cleaned_names = []

        if isinstance(columns, pd.MultiIndex):
            # マルチレベルヘッダーの場合: タプルを ' > ' で結合
            for i, col_tuple in enumerate(columns):
                col_parts = [
                    str(c).strip()
                    for c in col_tuple
                    if pd.notna(c) and str(c).strip() != ""
                ]

                if col_parts:
                    cleaned_names.append(" > ".join(col_parts))
                else:
                    # 全て空の場合、列番号を使用
                    cleaned_names.append(f"Col_{i+1}")
        else:
            # シングルレベルヘッダーの場合
            for i, col in enumerate(columns):
                col_str = str(col).strip()

                # "Unnamed: X" パターンの処理
                if col_str.startswith("Unnamed:"):
                    cleaned_names.append(f"Col_{i+1}")
                else:
                    cleaned_names.append(col_str)

        return cleaned_names

    def _compact_markdown_table(self, markdown: str) -> str:
        """
        Markdownテーブルから不要な空白を除去して圧縮
        ベクトルデータベース格納時のサイズを削減するため、
        各セルの前後の空白パディングを除去する

        Args:
            markdown: 元のMarkdownテーブル

        Returns:
            圧縮されたMarkdownテーブル
        """
        lines = markdown.split("\n")
        compacted_lines = []

        for line in lines:
            if "|" in line:
                # パイプで分割してセルを取得
                cells = line.split("|")
                # 各セルの空白を除去
                stripped_cells = [cell.strip() for cell in cells]
                # パイプで再結合（空白なし）
                compacted_line = "|".join(stripped_cells)
                compacted_lines.append(compacted_line)
            else:
                compacted_lines.append(line)

        return "\n".join(compacted_lines)
