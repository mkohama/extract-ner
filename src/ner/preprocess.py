"""NER 解析前のテキスト整形（UI 非依存）。

現状は Markdown テーブルの平文化のみ。GiNZA は自然文で学習しているため、
`|` を含むテーブル記法のままだとセル内の語をほとんど抽出できない。
"""

from __future__ import annotations

import re

# テーブルのセルを連結する区切り文字。読点が抽出精度・可読性ともに無難。
TABLE_CELL_DELIMITER = "、"

_SEPARATOR_CELL = re.compile(r"^:?-{1,}:?$")


def _split_table_row(line: str) -> list[str]:
    """Markdown テーブルの 1 行をセルのリストに分解する。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(line: str) -> bool:
    """`| --- | --- |` のような区切り行か判定する。"""
    cells = [c for c in _split_table_row(line) if c != ""]
    return bool(cells) and all(_SEPARATOR_CELL.match(c) for c in cells)


def flatten_markdown_tables(text: str, delimiter: str = TABLE_CELL_DELIMITER) -> str:
    """Markdown テーブルの記法を取り除いて NER 向きの平文にする。

    行単位で
    - 区切り行 (`| --- | --- |`) は削除
    - データ行はセルを区切り文字で連結し、末尾に句点を付与
    する。ヘッダー行に依存しないため、テーブルの途中だけを含むチャンクに
    適用しても破綻しない（`| 由利` のような記号混じりの誤抽出を生まない）。

    なお GiNZA の NER は文脈依存が強く、短い語や曖昧な語（短い英字の社名など）は
    どの整形をしても抽出されないことがある点には注意。
    """
    out: list[str] = []
    for line in text.split("\n"):
        if "|" not in line:
            out.append(line)
            continue
        if _is_separator_row(line):
            continue  # 区切り行は捨てる
        cells = [c for c in _split_table_row(line) if c]
        out.append(delimiter.join(cells) + "。" if cells else "")
    return "\n".join(out)


def prepare_for_ner(text: str) -> str:
    """NER 解析前のテキスト整形（現状は Markdown テーブルの平文化のみ）。"""
    return flatten_markdown_tables(text)
