# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM delegation used for low-confidence escalation."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Iterable, List, Optional, Sequence

from .drivers.manager import DriverManager
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

    allowed = {t for t in allow_types or ()}
    use_filter = bool(allowed)
    indices: List[int] = []
    for idx, block in enumerate(blocks):
        if max_confidence is not None and block.confidence > max_confidence:
            continue
        if use_filter and block.type not in allowed:
            continue
        indices.append(idx)
        if isinstance(limit, int) and limit > 0 and len(indices) >= limit:
            break
    return indices


def escalate_low_conf(
    blocks: Iterable[Block],
    recipe_name: str,
    *,
    max_confidence: Optional[float] = None,
    allow_types: Sequence[str] | None = None,
    limit: Optional[int] = None,
) -> List[Block]:
    """Escalate low-confidence predictions via a configured LLM driver."""

    recipe = load_recipe(recipe_name)
    if not recipe.llm or not recipe.llm.get("enable"):
        return list(blocks)

    original_blocks = list(blocks)
    indices = select_escalation_indices(
        original_blocks,
        max_confidence=max_confidence,
        allow_types=allow_types,
        limit=limit,
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
        escalated[idx] = clone_model(block, attrs=attrs)
    return escalated
