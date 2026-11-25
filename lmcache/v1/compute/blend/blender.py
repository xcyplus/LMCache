# SPDX-License-Identifier: Apache-2.0
# Standard
import os
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        self.layerwise_model = infer_model_from_vllm(vllm_model, self)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # Blend recompute ratio: read from env (LMCACHE_BLEND_RECOMPUTE_RATIO)
        # to support ablation (15% vs 20% etc.). Defaults to 0.15 (CacheBlend).
        _blend_ratio = float(os.environ.get("LMCACHE_BLEND_RECOMPUTE_RATIO", "0.15"))
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=[1],
            recomp_ratios=[_blend_ratio],
            thresholds=None,
        )
        logger.info(f"LMCBlender initialized with blend_recompute_ratio={_blend_ratio}")

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        self.document_reuse_mask: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def blend_documents(
        self, tokens: torch.Tensor, prefix_length: int, request_id: str, **kwargs
    ) -> None:
        """Assemble a complete scheduled prefix from on-demand document KV."""
        store = self.cache_engine.document_kv_store
        assert store is not None
        database = self.cache_engine.token_database
        spans = database.get_request_document_spans(request_id)
        if not spans:
            raise RuntimeError("Document spans disappeared before the worker loaded KV")
        if spans[-1][1] > len(tokens):
            raise ValueError("Document spans must be within the scheduled prompt")

        starts, ends, sources = [], [], []
        reuse_mask = torch.zeros(prefix_length, dtype=torch.bool)
        for start, end, url in spans:
            document_tokens = tokens[start:end]
            document_token_ids = document_tokens.tolist()
            path = store.path_for(url, document_token_ids)
            layers = store.load(path, len(document_tokens))
            if layers is None:
                layers = []

                def capture_kv(key, value):
                    layers.append(
                        torch.stack((key, value)).to(device="cpu", dtype=store.dtype)
                    )

                for _ in self.layerwise_model.compute_layer(
                    document_tokens, capture_kv=capture_kv
                ):
                    pass
                try:
                    store.save(path, layers)
                except (OSError, RuntimeError) as exc:
                    logger.warning("Could not persist KV for %s: %s", url, exc)
                else:
                    logger.info("Built independent document KV for %s", url)
            else:
                logger.info("Loaded independent document KV for %s", url)

            for offset in range(0, len(document_tokens), database.chunk_size):
                chunk_end = min(offset + database.chunk_size, len(document_tokens))
                chunk_tokens = document_token_ids[offset:chunk_end]
                database.registry.register_chunk(
                    url, database._hash_tokens(chunk_tokens)
                )
                if start + offset >= prefix_length:
                    continue
                chunk_end = min(chunk_end, prefix_length - start)
                starts.append(start + offset)
                ends.append(start + chunk_end)
                sources.append((layers, offset, chunk_end))
            reuse_mask[start : min(end, prefix_length)] = True

        logger.info(
            "Assembling request %s with %d document tokens and %d other prefix tokens",
            request_id,
            int(reuse_mask.sum()),
            prefix_length - int(reuse_mask.sum()),
        )
        self.document_reuse_mask = reuse_mask.to(kwargs["slot_mapping"].device)
        consumer = self.gpu_connector.batched_to_gpu(
            starts, ends, blend_token_range=(0, prefix_length), **kwargs
        )
        executor = self.layerwise_model.compute_layer(tokens[:prefix_length])
        try:
            next(consumer)
            for layer_id in range(self.num_layers):
                objects = []
                for layers, start, end in sources:
                    tensor = layers[layer_id][:, start:end].contiguous()
                    metadata = MemoryObjMetadata(
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        address=0,
                        phy_size=tensor.nbytes,
                        ref_count=1,
                        fmt=MemoryFormat.KV_2TD,
                    )
                    metadata.old_positions = torch.arange(start, end)
                    objects.append(TensorMemoryObj(tensor.flatten(), metadata))
                consumer.send(objects)
                next(executor)
            next(consumer)
        finally:
            executor.close()
            consumer.close()
            self.metadata.clean()
            self.document_reuse_mask = None

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")

        # If this layer is not in buffer_mapping (e.g., cache miss / empty
        # retrieve keys), skip blending and use vllm's freshly computed KV.
        # This happens when refresh_article removed stale chunks but no
        # cached chunks match the new document content.
        if layer_id not in self.gpu_connector.buffer_mapping:
            logger.debug(
                f"Layer {layer_id} not in buffer_mapping, skipping blend"
            )
            if attn_output is None:
                attn_output = torch.empty(
                    q.shape, dtype=q.dtype, device=q.device,
                )
            return q, k, v, residual, attn_output, attn_metadata

        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if layer_id in self.common_metadata.check_layers:
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            total_len = diff_k.shape[0]

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])

            if self.document_reuse_mask is None:
                top_indices = torch.topk(diff_k, k=topk_num).indices
            else:
                reusable = torch.where(self.document_reuse_mask)[0]
                required = torch.where(~self.document_reuse_mask)[0]
                count = int(len(reusable) * self.common_metadata.recomp_ratios[0])
                if len(required) == 0:
                    count = max(1, count)
                selected = reusable[torch.topk(diff_k[reusable], k=count).indices]
                top_indices = torch.cat((required, selected))
                topk_num = len(top_indices)
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Picking indices: {top_indices}")
            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.max_query_len = topk_num
            attn_metadata.query_start_loc = torch.tensor(
                [0, topk_num], dtype=torch.int32, device=q.device
            )

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            if self.document_reuse_mask is not None:
                # Before the selection layer, attention uses fresh projections
                # but only the gaps need to be copied into the paged KV buffer.
                required = ~self.document_reuse_mask
                old_k[required] = k[required]
                old_v[required] = v[required]
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """
        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
