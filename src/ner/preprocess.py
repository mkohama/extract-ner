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


# 1 文字とその「原文での文字位置」（挿入文字は -1）の組。平坦化の対応表づくりに使う。
_Char = tuple[str, int]


def _cells_with_pos(line: str, line_start: int) -> list[list[_Char]]:
    """テーブル行を `|` で分割し、各セルを (文字, 原文インデックス) 列で返す（前後空白は除去）。

    `_split_table_row` の位置情報つき版。`|` 区切りで分け、各セルの前後空白を落とす。
    （`line.split("|")` 相当。先頭/末尾 `|` 由来の空セルは呼び出し側で除外する）
    """
    cells: list[list[_Char]] = [[]]
    for i, ch in enumerate(line):
        if ch == "|":
            cells.append([])
        else:
            cells[-1].append((ch, line_start + i))
    stripped: list[list[_Char]] = []
    for cell in cells:
        s, e = 0, len(cell)
        while s < e and cell[s][0].isspace():
            s += 1
        while e > s and cell[e - 1][0].isspace():
            e -= 1
        stripped.append(cell[s:e])
    return stripped


def _flatten_line_with_map(
    line: str, line_start: int, delimiter: str
) -> list[_Char] | None:
    """1 行を平坦化し (文字, 原文インデックス) 列で返す。区切り行は None（=削除）。"""
    if "|" not in line:
        return [(ch, line_start + i) for i, ch in enumerate(line)]
    if _is_separator_row(line):
        return None
    cells = [c for c in _cells_with_pos(line, line_start) if c]
    if not cells:
        return []
    seg: list[_Char] = []
    for ci, cell in enumerate(cells):
        if ci > 0:
            seg.append((delimiter, -1))  # 挿入した区切り（原文に対応なし）
        seg.extend(cell)
    seg.append(("。", -1))  # 挿入した句点（原文に対応なし）
    return seg


def flatten_markdown_tables_with_map(
    text: str, delimiter: str = TABLE_CELL_DELIMITER
) -> tuple[str, list[int]]:
    """:func:`flatten_markdown_tables` と同じ平坦化に、文字位置の対応表を付けて返す。

    返り値 ``(flat, cmap)``：``flat`` は平坦化後テキスト、``cmap[i]`` は ``flat`` の
    i 文字目に対応する**原文（引数 text）の文字位置**。挿入文字（区切り `、`/句点 `。`/
    行連結の改行）は ``-1``。検出（平坦化テキスト）で得たスパンを、`|` 入り原文へ
    逆写像してマスクするために使う（src.masking.apply）。
    """
    out_lines: list[list[_Char]] = []
    line_start = 0
    for line in text.split("\n"):
        seg = _flatten_line_with_map(line, line_start, delimiter)
        if seg is not None:
            out_lines.append(seg)
        line_start += len(line) + 1  # +1 は split で落ちた "\n" の分
    chars: list[_Char] = []
    for j, seg in enumerate(out_lines):
        if j > 0:
            chars.append(("\n", -1))  # 出力行を連結する改行（原文対応は付けない）
        chars.extend(seg)
    flat = "".join(ch for ch, _ in chars)
    cmap = [pos for _, pos in chars]
    return flat, cmap


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
    return flatten_markdown_tables_with_map(text, delimiter)[0]


def prepare_for_ner(text: str) -> str:
    """NER 解析前のテキスト整形（現状は Markdown テーブルの平文化のみ）。"""
    return flatten_markdown_tables(text)


def prepare_for_ner_with_map(text: str) -> tuple[str, list[int]]:
    """:func:`prepare_for_ner` の対応表つき版（平坦化後テキストと原文位置対応表）。"""
    return flatten_markdown_tables_with_map(text)
