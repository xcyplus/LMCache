# SPDX-License-Identifier: Apache-2.0
"""Persistent, independently encoded KV for complete RAG documents."""

import hashlib
import json
import os
from pathlib import Path
from pickle import UnpicklingError
from tempfile import NamedTemporaryFile

import torch

from lmcache.logging import init_logger

DOCUMENT_KV_DIR_ENV = "LMCACHE_DOCUMENT_KV_DIR"
logger = init_logger(__name__)


class DocumentKVStore:
    def __init__(
        self,
        directory: str,
        model_identity: str,
        num_layers: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_identity = model_identity
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dtype = dtype

    def path_for(self, url: str, tokens: list[int]) -> Path:
        identity = json.dumps(
            [
                1,
                self.model_identity,
                self.num_layers,
                self.hidden_size,
                str(self.dtype),
                url,
                tokens,
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.pt"

    def load(self, path: Path, num_tokens: int) -> list[torch.Tensor] | None:
        try:
            layers = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            shape = (2, num_tokens, self.hidden_size)
            if (
                not isinstance(layers, list)
                or len(layers) != self.num_layers
                or any(
                    not isinstance(layer, torch.Tensor)
                    or layer.shape != shape
                    or layer.dtype != self.dtype
                    or not layer.is_contiguous()
                    for layer in layers
                )
            ):
                raise ValueError("Invalid document KV layout")
            return layers
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
            logger.warning("Could not load document KV %s: %s", path, exc)
            return None

    def save(self, path: Path, layers: list[torch.Tensor]) -> None:
        # Publish a complete file in one rename. Readers never see partial layers.
        with NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as tmp:
            temporary_path = Path(tmp.name)
        try:
            torch.save(layers, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
