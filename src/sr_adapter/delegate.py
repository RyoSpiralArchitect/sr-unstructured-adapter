# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM delegation used for low-confidence escalation."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Iterable, List, Optional, Sequence

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


def _get_driver_manager() -> DriverManager:
    global _driver_manager
    if _driver_manager is None:
        _driver_manager = DriverManager()
    return _driver_manager


def select_escalation_indices(
    blocks: Sequence[Block],
    *,
    max_confidence: Optional[float] = None,
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
        allow_types=allow_types,
        limit=limit,
    )
    return list(result.indices)


def escalate_low_conf(
    blocks: Iterable[Block],
    recipe_name: str,
    *,
    max_confidence: Optional[float] = None,
    allow_types: Sequence[str] | None = None,
    limit: Optional[int] = None,
    selection: SelectionResult | None = None,
) -> List[Block]:
    """Escalate low-confidence predictions via a configured LLM driver."""

    recipe = load_recipe(recipe_name)
    if not recipe.llm or not recipe.llm.get("enable"):
        return list(blocks)

    original_blocks = list(blocks)
    if selection is not None:
        indices = list(selection.indices)
    else:
        indices = select_escalation_indices(
            original_blocks,
            max_confidence=max_confidence,
            allow_types=allow_types,
            limit=limit,
        )
        selection = get_escalation_policy().last()
    logger_instance = get_escalation_logger()
    if selection is not None:
        logger_instance.log_selection(
            recipe.name,
            selection,
            original_blocks,
            metadata={
                "max_confidence": max_confidence,
                "allow_types": list(allow_types or ()),
                "limit": limit,
            },
        )
    if not indices:
        return original_blocks

    manager = _get_driver_manager()
    tenant = str(recipe.llm.get("tenant") or manager.tenant_manager.get_default_tenant())
    try:
        driver = manager.get_driver(tenant, recipe.llm)
    except Exception as exc:  # pragma: no cover - configuration failure path
        logger.warning(
            "LLM escalation skipped because driver could not be resolved for tenant '%s': %s",
            tenant,
            exc,
        )
        return original_blocks
    prompt_template = recipe.llm.get("prompt_template")
    target_blocks = [original_blocks[i] for i in indices]
    context = "\n\n".join(block.text for block in target_blocks)
    if prompt_template:
        try:
            prompt = str(prompt_template).format(context=context, recipe=recipe.name)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Failed to render prompt template for recipe '%s'", recipe.name)
            prompt = context
    else:
        prompt = context

    metadata = {
        "recipe": recipe.name,
        "block_count": len(target_blocks),
        "indices": indices,
    }
    try:
        raw_response = driver.generate(prompt, metadata=metadata)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("LLM escalation failed for tenant '%s' with driver '%s': %s", tenant, driver.name, exc)
        if selection is not None:
            logger_instance.log_failure(
                recipe.name,
                reason=str(exc),
                selection=selection,
            )
        return original_blocks

    normalized = _normalizer.normalize(driver.name, raw_response, prompt=prompt)
    payload = asdict(normalized)
    payload.update({"tenant": tenant, "driver": driver.name})

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
