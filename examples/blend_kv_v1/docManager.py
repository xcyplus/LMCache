import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple


class RAG_Database:
    def __init__(self, rag_db_path: str = "rag_db.json") -> None:
        """从 JSON 文件加载 RAG 数据库。

        JSON 文件格式：
            {
                "http://example.com/doc1": "文档1的文本内容...",
                "http://example.com/doc2": "文档2的文本内容..."
            }

        :param rag_db_path: JSON 文件路径，默认 rag_db.json
        """
        self.rag_db_path = rag_db_path
        self.url_to_text: Dict[str, str] = {}

        if os.path.exists(rag_db_path):
            with open(rag_db_path, "r", encoding="utf-8") as f:
                self.url_to_text = json.load(f)
        else:
            print(f"[RAG_Database] Warning: '{rag_db_path}' not found, starting empty.")

    def get_text(self, url: str) -> Optional[str]:
        """根据 url 查询对应的文本内容。

        :param url: 要查询的文档 url
        :return: 对应的文本内容；url 不存在时返回 None
        """
        return self.url_to_text.get(url, None)

    def __contains__(self, url: str) -> bool:
        return url in self.url_to_text

    def __len__(self) -> int:
        return len(self.url_to_text)


class DocCacheManager:
    def __init__(
        self, rag_database: RAG_Database, *, ipc_endpoint: Optional[str] = None
    ):
        self.rag_database = rag_database
        self.ipc_endpoint = ipc_endpoint

    async def register_request(
        self, request_id: str, document_spans: List[Tuple[int, int, str]]
    ) -> None:
        """Register document token spans before submitting a model request."""
        await asyncio.to_thread(self._send_document_spans, request_id, document_spans)

    async def unregister_request(self, request_id: str) -> None:
        """Remove the worker's document spans when the request ends."""
        await asyncio.to_thread(self._send_document_spans, request_id, None)

    def _send_document_spans(
        self, request_id: str, document_spans: Optional[List[Tuple[int, int, str]]]
    ) -> None:
        if self.ipc_endpoint is None:
            raise RuntimeError("Configure an IPC endpoint to register document spans")

        import zmq
        from lmcache.v1.lookup_client.lmcache_lookup_client import (
            DOCUMENT_SPANS_MESSAGE,
        )

        payload = json.dumps([request_id, document_spans]).encode("utf-8")
        with zmq.Context() as context, context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            socket.setsockopt(zmq.SNDTIMEO, 5000)
            socket.connect(self.ipc_endpoint)
            socket.send_multipart([DOCUMENT_SPANS_MESSAGE, payload])
            reply = socket.recv()
        if reply != b"ok":
            raise RuntimeError(reply.decode("utf-8"))
