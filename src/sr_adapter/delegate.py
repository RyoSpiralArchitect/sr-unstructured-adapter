# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM delegation used for low-confidence escalation."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .drivers.manager import DriverManager
from .escalation import (
    SelectionResult,
    get_escalation_logger,
    get_escalation_policy,
)
from .normalizer import LLMNormalizer
from .schema import Block, clone_model
from .recipe import load_recipe

logger = logging.getLogger(__name__)

_driver_manager: DriverManager | None = None
_normalizer = LLMNormalizer()


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)
    return int(number)


def _safe_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(default)


def _clamp_int(value: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    return max(min_value, min(max_value, int(value)))


def _render_block_texts(
    blocks: Sequence[Block],
    indices: Sequence[int],
    *,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        max_chars = 8_000
    budget = int(max_chars)
    parts: list[str] = []
    used = 0
    for idx in indices:
        if idx < 0 or idx >= len(blocks):
            continue
        text = (blocks[idx].text or "").strip()
        if not text:
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            parts.append(text[:remaining])
            used = budget
            break
        parts.append(text)
        used += len(text) + 2
    return "\n\n".join(parts)


def _select_related_context(
    blocks: Sequence[Block],
    *,
    target_indices: Sequence[int],
    top_k: int,
    neighbor_window: int,
    max_blocks: int,
    embed_dim: int,
    use_semantic_field: bool,
) -> list[int]:
    if not blocks:
        return []
    targets = [idx for idx in target_indices if 0 <= idx < len(blocks)]
    if not targets:
        return []

    top_k = _clamp_int(int(top_k), min_value=0, max_value=128)
    neighbor_window = _clamp_int(int(neighbor_window), min_value=0, max_value=64)
    max_blocks = _clamp_int(int(max_blocks), min_value=0, max_value=256)
    embed_dim = _clamp_int(int(embed_dim), min_value=32, max_value=4096)

    target_set = set(targets)
    forced: list[int] = []
    if neighbor_window > 0:
        for idx in targets:
            start = max(0, idx - neighbor_window)
            end = min(len(blocks), idx + neighbor_window + 1)
            for j in range(start, end):
                if j in target_set:
                    continue
                forced.append(j)
    forced = sorted(set(forced))

    scored: dict[int, float] = {}
    if top_k > 0:
        try:
            from .embedding import BlockEmbedder, EmbeddingIndex

            embedder = BlockEmbedder(
                dimensions=embed_dim,
                use_sentence_transformers=False,
            )
            vectors = embedder.embed_with_context(blocks) if use_semantic_field else embedder.embed(blocks)
            if not vectors:
                raise ValueError("No embeddings produced")
            index = EmbeddingIndex(len(vectors[0]))
            for vec in vectors:
                index.add(vec)

            for idx in targets:
                query = vectors[idx]
                hits = index.search(query, top_k=top_k + 1)
                for hit in hits:
                    j = int(hit.index)
                    if j in target_set or j == idx:
                        continue
                    score = float(hit.score)
                    if math.isnan(score) or math.isinf(score):
                        continue
                    scored[j] = max(scored.get(j, float("-inf")), score)
        except Exception:
            scored = {}

    related: list[int] = []
    for idx in forced:
        if idx not in related and idx not in target_set:
            related.append(idx)
            if max_blocks > 0 and len(related) >= max_blocks:
                return related

    if max_blocks > 0 and len(related) >= max_blocks:
        return related

    remaining = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    for idx, _ in remaining:
        if idx in target_set or idx in related:
            continue
        related.append(idx)
        if max_blocks > 0 and len(related) >= max_blocks:
            break

    related.sort()
    return related


def _get_driver_manager() -> DriverManager:
    global _driver_manager
    if _driver_manager is None:
        _driver_manager = DriverManager()
    return _driver_manager


def select_escalation_indices(
    blocks: Sequence[Block],
    *,
    max_confidence: Optional[float] = None,
    max_semantic_confidence: Optional[float] = None,
    allow_types: Sequence[str] | None = None,
    limit: Optional[int] = None,
) -> List[int]:
    """Return indices of blocks that satisfy the escalation policy."""

    if isinstance(limit, int) and limit <= 0:
        return []

    policy = get_escalation_policy()
    result = policy.evaluate(
        blocks,
        max_confidence=max_confidence,
        max_semantic_confidence=max_semantic_confidence,
        allow_types=allow_types,
        limit=limit,
    )
    return list(result.indices)


def escalate_low_conf(
    blocks: Iterable[Block],
    recipe_name: str,
    *,
    tenant: str | None = None,
    max_confidence: Optional[float] = None,
    max_semantic_confidence: Optional[float] = None,
    allow_types: Sequence[str] | None = None,
    limit: Optional[int] = None,
    selection: SelectionResult | None = None,
    context_overrides: Mapping[str, Any] | None = None,
) -> List[Block]:
    """Escalate low-confidence predictions via a configured LLM driver."""

    recipe = load_recipe(recipe_name)
    if not recipe.llm or not recipe.llm.get("enable"):
        return list(blocks)

    if max_confidence is None:
        recipe_threshold = recipe.llm.get("min_conf", recipe.llm.get("max_confidence"))
        if recipe_threshold is not None:
            try:
                max_confidence = float(recipe_threshold)
            except Exception:
                max_confidence = None

    original_blocks = list(blocks)
    if selection is not None:
        indices = list(selection.indices)
    else:
        policy_engine = get_escalation_policy()
        selection = policy_engine.evaluate(
            original_blocks,
            max_confidence=max_confidence,
            max_semantic_confidence=max_semantic_confidence,
            allow_types=allow_types,
            limit=limit,
        )
        indices = list(selection.indices)
    logger_instance = get_escalation_logger()
    if selection is not None:
        logger_instance.log_selection(
            recipe.name,
            selection,
            original_blocks,
            metadata={
                "max_confidence": max_confidence,
                "max_semantic_confidence": max_semantic_confidence,
                "allow_types": list(allow_types or ()),
                "limit": limit,
            },
        )
    if not indices:
        return original_blocks

    manager = _get_driver_manager()
    tenant_override = str(tenant).strip() if isinstance(tenant, str) and tenant.strip() else None
    tenant_name = str(
        tenant_override
        or recipe.llm.get("tenant")
        or manager.tenant_manager.get_default_tenant()
    )
    try:
        driver = manager.get_driver(tenant_name, recipe.llm)
    except Exception as exc:  # pragma: no cover - configuration failure path
        logger.warning(
            "LLM escalation skipped because driver could not be resolved for tenant '%s': %s",
            tenant_name,
            exc,
        )
        return original_blocks
    prompt_template = recipe.llm.get("prompt_template") or recipe.llm.get("prompt")
    target_blocks = [original_blocks[i] for i in indices]

    context_cfg: dict[str, Any] = {}
    raw_context_cfg = recipe.llm.get("context")
    if isinstance(raw_context_cfg, Mapping):
        context_cfg.update(raw_context_cfg)
    if isinstance(context_overrides, Mapping) and context_overrides:
        context_cfg.update(context_overrides)

    context_top_k = _safe_int(context_cfg.get("top_k", recipe.llm.get("context_top_k")), default=0)
    context_neighbor_window = _safe_int(
        context_cfg.get("neighbor_window", recipe.llm.get("context_neighbor_window")),
        default=0,
    )
    context_max_blocks = _safe_int(context_cfg.get("max_blocks", recipe.llm.get("context_max_blocks")), default=8)
    context_max_chars = _safe_int(context_cfg.get("max_chars", recipe.llm.get("context_max_chars")), default=8_000)
    context_embed_dim = _safe_int(context_cfg.get("embed_dim", recipe.llm.get("context_embed_dim")), default=64)
    context_use_semantic_field = _safe_bool(
        context_cfg.get("use_semantic_field", recipe.llm.get("context_use_semantic_field")),
        default=False,
    )

    context = "\n\n".join(block.text for block in target_blocks)
    context_related_indices = _select_related_context(
        original_blocks,
        target_indices=indices,
        top_k=context_top_k,
        neighbor_window=context_neighbor_window,
        max_blocks=context_max_blocks,
        embed_dim=context_embed_dim,
        use_semantic_field=context_use_semantic_field,
    )
    context_related = _render_block_texts(
        original_blocks,
        context_related_indices,
        max_chars=context_max_chars,
    )

    if prompt_template:
        rendered = str(prompt_template).strip()
        if "{context}" in rendered or "{recipe}" in rendered or "{context_related}" in rendered:
            try:
                prompt = rendered.format(
                    context=context,
                    context_related=context_related,
                    recipe=recipe.name,
                )
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Failed to render prompt template for recipe '%s'", recipe.name)
                prompt = context
        else:
            prompt = f"{rendered}\n\n{context}" if context else rendered
    else:
        prompt = context

    if context_related and "{context_related}" not in str(prompt_template or ""):
        prompt = f"{prompt}\n\n[RELATED CONTEXT]\n{context_related}"

    metadata = {
        "recipe": recipe.name,
        "tenant": tenant_name,
        "block_count": len(target_blocks),
        "indices": indices,
        "context_indices": context_related_indices,
    }
    try:
        raw_response = driver.generate(prompt, metadata=metadata)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning(
            "LLM escalation failed for tenant '%s' with driver '%s': %s",
            tenant_name,
            driver.name,
            exc,
        )
        if selection is not None:
            logger_instance.log_failure(
                recipe.name,
                reason=str(exc),
                selection=selection,
            )
        return original_blocks

    normalized = _normalizer.normalize(driver.name, raw_response, prompt=prompt)
    payload = asdict(normalized)
    payload.update(
        {
            "tenant": tenant_name,
            "driver": driver.name,
            "indices": indices,
            "context_indices": context_related_indices,
        }
    )

    escalated = list(original_blocks)
    for idx in indices:
        block = original_blocks[idx]
        attrs = dict(block.attrs)
        escalations = list(attrs.get("llm_escalations", []))
        enriched = copy.deepcopy(payload)
        enriched["target_index"] = idx
        escalations.append(enriched)
        attrs["llm_escalations"] = escalations
        candidate = selection.find(idx) if selection else None
        if candidate is not None:
            attrs.setdefault("llm_meta", {})
            meta = dict(attrs.get("llm_meta") or {})
            meta.update(
                {
                    "escalation_score": candidate.score,
                    "escalation_rank": candidate.rank,
                }
            )
            attrs["llm_meta"] = meta
        updated_block = clone_model(block, attrs=attrs)
        escalated[idx] = updated_block
        if candidate is not None:
            logger_instance.log_result(
                recipe.name,
                updated_block,
                index=idx,
                candidate_score=candidate.score,
                llm_result=normalized,
                rank=candidate.rank,
            )
    return escalated
