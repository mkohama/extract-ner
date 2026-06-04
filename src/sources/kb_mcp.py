"""kb-mcp への Streamable HTTP 接続と Resource 読み取り.

term-variants の kb_mcp/client.py から、固有表現抽出に必要な
list_documents / get_document のみを移植したもの。

kb-mcp 側は次のように HTTP サーバを起動しておくこと:
    uv run kb-mcp-server --transport http --port 8000
既定の接続先は http://localhost:8000/mcp。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

# .env を読み込み、接続先 URL を環境変数 KB_MCP_URL から取得する
# (term-variants と同じ流儀。既存の環境変数は上書きしない)
load_dotenv(override=False)
DEFAULT_KB_MCP_URL = os.environ.get("KB_MCP_URL", "http://localhost:8000/mcp")


def suppress_async_generator_errors() -> None:
    """MCP SDK / anyio の非同期ジェネレータクローズ時エラーを stderr に出さない."""
    _original_hook = sys.unraisablehook

    def _quiet_hook(unraisable: sys.UnraisableHookArgs) -> None:
        if unraisable.err_msg and "asynchronous generator" in unraisable.err_msg:
            return
        _original_hook(unraisable)

    sys.unraisablehook = _quiet_hook


class KbMcpConnectionError(Exception):
    """kb-mcp サーバへの接続に失敗した場合の例外."""


def _extract_root_cause(exc: BaseException) -> str:
    """ExceptionGroup やネストされた例外から根本原因のメッセージを抽出する."""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            msg = _extract_root_cause(sub)
            if msg:
                return msg
    cause = exc.__cause__
    if cause is not None:
        return _extract_root_cause(cause)
    return str(exc)


class KbMcpClient:
    """kb-mcp サーバへの MCP クライアント.

    Usage::

        async with KbMcpClient("http://localhost:8000/mcp") as client:
            docs = await client.list_documents()
            detail = await client.get_document("doc_xxx")
    """

    def __init__(self, url: str = DEFAULT_KB_MCP_URL) -> None:
        self._url = url
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> KbMcpClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        """Streamable HTTP で kb-mcp に接続し、セッションを初期化する."""
        try:
            client_ctx = streamablehttp_client(self._url)
            streams = await self._exit_stack.enter_async_context(client_ctx)
            read_stream, write_stream = streams[0], streams[1]

            session_ctx = ClientSession(read_stream, write_stream)
            session = await self._exit_stack.enter_async_context(session_ctx)
            await session.initialize()
            self._session = session
        except (
            BaseExceptionGroup,
            OSError,
            asyncio.CancelledError,
            RuntimeError,
        ) as exc:
            root_cause = _extract_root_cause(exc)
            raise KbMcpConnectionError(
                f"kb-mcp サーバ ({self._url}) に接続できません。"
                f"サーバが起動しているか確認してください。\n"
                f"原因: {root_cause}"
            ) from None

    async def close(self) -> None:
        """接続を閉じる."""
        try:
            await self._exit_stack.aclose()
        except BaseException:
            # MCP SDK / anyio がクローズ時に投げる例外はすべて無視する。
            pass
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with KbMcpClient(url):'")
        return self._session

    async def _read_resource_json(self, uri: str) -> Any:
        """Resource を読んで JSON としてパースする."""
        result = await self.session.read_resource(AnyUrl(uri))
        if not result.contents:
            raise ValueError(f"Empty response for resource: {uri}")
        content = result.contents[0]
        text: str = getattr(content, "text", "")
        if not text:
            raise ValueError(f"No text in resource content: {uri}")
        return json.loads(text)

    async def list_documents(self) -> list[dict[str, Any]]:
        """kb-mcp に登録された全文書のメタ情報リストを返す."""
        data: Any = await self._read_resource_json("knowledge://documents")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("documents", "items", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError(f"Unexpected document list format: {type(data)}")

    async def get_document(self, doc_id: str) -> dict[str, Any]:
        """指定ドキュメントの全チャンク本文を含む詳細を返す."""
        data: Any = await self._read_resource_json(f"knowledge://documents/{doc_id}")
        return data


def list_documents_sync(url: str = DEFAULT_KB_MCP_URL) -> list[dict[str, Any]]:
    """同期版: 文書一覧を取得する (Streamlit から使う)."""

    async def _run() -> list[dict[str, Any]]:
        async with KbMcpClient(url) as client:
            return await client.list_documents()

    return asyncio.run(_run())


def get_document_text_sync(doc_id: str, url: str = DEFAULT_KB_MCP_URL) -> str:
    """同期版: 指定ドキュメントの全文テキストを取得する.

    kb-mcp の detail は `content` に全文を持つが、無い場合は chunks を結合する。
    """

    async def _run() -> dict[str, Any]:
        async with KbMcpClient(url) as client:
            return await client.get_document(doc_id)

    detail = asyncio.run(_run())
    content = detail.get("content")
    if isinstance(content, str) and content.strip():
        return content
    chunks = detail.get("chunks", [])
    return "\n\n".join(c.get("text", c.get("page_content", "")) for c in chunks)
