"""解析キャッシュ（UI 非依存）。

解析は **NER 層（GiNZA 2モデル＝激重。400 チャンクで 7〜10 分）** と **マスキング層（辞書照合・確信度・
クラスタ・除外＝ミリ秒）** に分かれる。**キャッシュするのは NER 層だけ**にし、マスキング層は都度再計算する
（辞書・除外を変えても再 NER 不要。`MaskingEngine.analyze` が利用）。

- キー `(content_hash, model, flatten)` → per-model の :class:`~src.ner.engine.Analysis`。
- `content_hash` はチャンク列（解析対象テキスト）の sha256。内容が同じなら名前が違っても同一視。
- 格納は SQLite（`data/cache.db`）。件数が増えても O(1) 照合・一覧/削除が容易。

確定マスク（人手レビュー後の決定）の保存は別テーブル（将来）。本モジュールはまず NER 層を担う。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.ner.engine import AnalyzedToken, Analysis, Entity


def content_hash(chunks: Iterable[str]) -> str:
    """チャンク列（解析対象テキスト）の内容ハッシュ。区切りバイトで連結の曖昧性を避ける。"""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def analysis_to_dict(a: Analysis) -> dict:
    """:class:`Analysis` を JSON 化可能な dict にする（NER 層キャッシュの値）。"""
    return {
        "text": a.text,
        "original_text": a.original_text,
        "offset_map": list(a.offset_map),
        "tokens": [[t.start, t.end, t.surface, t.tag, t.pos] for t in a.tokens],
        "entities": [[e.text, e.label, e.start, e.end] for e in a.entities],
    }


def analysis_from_dict(d: dict) -> Analysis:
    """:func:`analysis_to_dict` の逆。"""
    return Analysis(
        text=d["text"],
        tokens=tuple(AnalyzedToken(*t) for t in d["tokens"]),
        entities=tuple(
            Entity(text=e[0], label=e[1], start=e[2], end=e[3]) for e in d["entities"]
        ),
        original_text=d.get("original_text", ""),
        offset_map=tuple(d.get("offset_map") or []),
    )


@dataclass(frozen=True)
class DocInfo:
    """キャッシュ済み文書 1 件の表示用メタ情報（キャッシュ一覧で使う）。"""

    content_hash: str
    source_kind: str  # text / file / kb
    source_name: str
    char_count: int
    chunk_count: int
    models: tuple[str, ...]  # NER キャッシュ済みのモデル
    created_at: str


class NerCache:
    """NER 層（per-model Analysis）の SQLite キャッシュ。キー＝(content_hash, model, flatten)。

    あわせて文書メタ（ソース名・チャンク数等）を ``documents`` に持ち、キャッシュ一覧の参照・
    削除に使う（NER 層は content_hash でしか引けないので、人が見て分かる情報を別に記録する）。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ner ("
                "content_hash TEXT, model TEXT, flatten INTEGER, "
                "analysis_json TEXT, created_at TEXT, "
                "PRIMARY KEY (content_hash, model, flatten))"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "content_hash TEXT PRIMARY KEY, source_kind TEXT, source_name TEXT, "
                "char_count INTEGER, chunk_count INTEGER, created_at TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def record_document(
        self,
        content_hash: str,
        source_kind: str,
        source_name: str,
        char_count: int,
        chunk_count: int,
    ) -> None:
        """文書メタを記録（一覧表示用）。同じ content_hash は上書き。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    source_kind,
                    source_name,
                    char_count,
                    chunk_count,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def list_documents(self) -> list[DocInfo]:
        """キャッシュ済み文書の一覧（新しい順）。各文書の NER キャッシュ済みモデルも付ける。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT d.content_hash, d.source_kind, d.source_name, d.char_count, "
                "d.chunk_count, d.created_at, "
                "(SELECT GROUP_CONCAT(DISTINCT model) FROM ner n "
                " WHERE n.content_hash = d.content_hash) "
                "FROM documents d ORDER BY d.created_at DESC"
            ).fetchall()
        return [
            DocInfo(
                content_hash=r[0],
                source_kind=r[1],
                source_name=r[2],
                char_count=r[3],
                chunk_count=r[4],
                created_at=r[5],
                models=tuple((r[6] or "").split(",")) if r[6] else (),
            )
            for r in rows
        ]

    def delete(self, content_hash: str) -> None:
        """1 文書のキャッシュ（NER 層＋文書メタ）を削除する。"""
        with self._conn() as c:
            c.execute("DELETE FROM ner WHERE content_hash = ?", (content_hash,))
            c.execute("DELETE FROM documents WHERE content_hash = ?", (content_hash,))

    def get(self, content_hash: str, model: str, flatten: bool) -> Analysis | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT analysis_json FROM ner "
                "WHERE content_hash=? AND model=? AND flatten=?",
                (content_hash, model, int(flatten)),
            ).fetchone()
        return analysis_from_dict(json.loads(row[0])) if row else None

    def put(
        self, content_hash: str, model: str, flatten: bool, analysis: Analysis
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO ner VALUES (?, ?, ?, ?, ?)",
                (
                    content_hash,
                    model,
                    int(flatten),
                    json.dumps(analysis_to_dict(analysis), ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
