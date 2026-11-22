# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import json
import os
import threading

# Third Party
from vllm.utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache
from lmcache.v1.token_database import UrlSegmentTokenDatabase

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)
DOCUMENT_SPANS_ENDPOINT_ENV = "LMCACHE_DOCUMENT_SPANS_ENDPOINT"
DOCUMENT_SPANS_MESSAGE = b"document-spans-v1"


class LMCacheLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - create_lookup_server_only_on_worker_0:
        is a flag to control whether to create lookup server only on worker 0.
    """

    def __init__(self, vllm_config: "VllmConfig", config: LMCacheEngineConfig):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.create_lookup_server_only_on_worker_0 = (
            config.extra_config
            and config.extra_config.get("create_lookup_server_only_on_worker_0", True)
        )
        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0:
            ranks = 1
        for tp_rank in range(ranks):
            socket_path = get_zmq_rpc_path_lmcache(
                vllm_config, "lookup", rpc_port, tp_rank
            )
            if tp_rank == 0:
                self.socket = make_zmq_socket(
                    self.ctx,
                    socket_path,
                    zmq.REQ,  # type: ignore[attr-defined]
                    bind=False,
                )
            else:
                self.socket.connect(socket_path)

    def lookup(self, token_ids: torch.Tensor, request_id: Optional[str] = None) -> int:
        token_bufs = self.encoder.encode(token_ids)
        request_id_buf = request_id.encode("utf-8")
        ranks = self.tensor_parallel_size
        if self.create_lookup_server_only_on_worker_0:
            ranks = 1
        results = []
        for i in range(ranks):
            self.socket.send_multipart(token_bufs + [request_id_buf], copy=False)
            resp = self.socket.recv()
            result = int.from_bytes(resp, "big")
            results.append(result)
        if not all(x == results[0] for x in results):
            raise RuntimeError(
                f"Lookup results (number of hit tokens) differ "
                f"across tensor parallel ranks: {results}."
            )
        return results[0]

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, "lookup", rpc_port, vllm_config.parallel_config.rank
        )
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        document_spans_endpoint = os.getenv(DOCUMENT_SPANS_ENDPOINT_ENV)
        if document_spans_endpoint:
            if vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Document span registration requires one TP worker")
            self.socket.bind(document_spans_endpoint)
        self.running = True

        def process_request():
            while self.running:
                frames = self.socket.recv_multipart(copy=False)
                match self._message_type(frames, document_spans_endpoint):
                    case "document_spans":
                        response = self._handle_document_spans(frames[1].bytes)
                    case "lookup":
                        response = self._handle_lookup(frames)
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    @staticmethod
    def _message_type(frames: list[zmq.Frame], endpoint: Optional[str]) -> str:
        # DocCacheManager 的注册或撤销请求带有固定标记。
        if endpoint and len(frames) == 2 and frames[0].bytes == DOCUMENT_SPANS_MESSAGE:
            return "document_spans"
        return "lookup"

    def _handle_lookup(self, frames: list[zmq.Frame]) -> bytes:
        # Lookup client 把请求 ID 放在最后一帧。
        request_id = frames[-1].bytes.decode("utf-8")
        token_ids = self.decoder.decode(frames[:-1])
        result = self.lmcache_engine.lookup(token_ids, request_id=request_id, pin=True)
        return result.to_bytes(4, "big")

    def _handle_document_spans(self, payload: bytes) -> bytes:
        """Register the request's document spans or remove them when spans is null."""
        try:
            request_id, spans = json.loads(payload)
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("Expected a non-empty request ID")
            database = self.lmcache_engine.token_database
            if not isinstance(database, UrlSegmentTokenDatabase):
                raise ValueError("Document span registration requires URL-aware blending")
            if spans is None:
                database.remove_request_document_spans(request_id)
            else:
                if not isinstance(spans, list):
                    raise ValueError("Expected a list of document spans")
                database.set_request_document_spans(
                    request_id, [tuple(span) for span in spans]
                )
        except (TypeError, ValueError) as exc:
            return str(exc).encode("utf-8")
        return b"ok"

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
