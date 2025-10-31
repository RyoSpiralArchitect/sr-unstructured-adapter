# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM delegation used for low-confidence escalation."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Iterable, List

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


def escalate_low_conf(blocks: Iterable[Block], recipe_name: str) -> List[Block]:
    """Escalate low-confidence predictions via a configured LLM driver."""

    recipe = load_recipe(recipe_name)
    if not recipe.llm or not recipe.llm.get("enable"):
        return list(blocks)

    original_blocks = list(blocks)
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
    context = "\n\n".join(block.text for block in original_blocks)
    if prompt_template:
        try:
            prompt = str(prompt_template).format(context=context, recipe=recipe.name)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Failed to render prompt template for recipe '%s'", recipe.name)
            prompt = context
    else:
        prompt = context

    metadata = {"recipe": recipe.name, "block_count": len(original_blocks)}
    try:
        raw_response = driver.generate(prompt, metadata=metadata)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("LLM escalation failed for tenant '%s' with driver '%s': %s", tenant, driver.name, exc)
        return original_blocks

    normalized = _normalizer.normalize(driver.name, raw_response, prompt=prompt)
    payload = asdict(normalized)
    payload.update({"tenant": tenant, "driver": driver.name})

    escalated: List[Block] = []
    for block in original_blocks:
        attrs = dict(block.attrs)
        escalations = list(attrs.get("llm_escalations", []))
        escalations.append(copy.deepcopy(payload))
        attrs["llm_escalations"] = escalations
        escalated.append(clone_model(block, attrs=attrs))
    return escalated
