# SPDX-License-Identifier: AGPL-3.0-or-later
"""High level conversion pipeline wiring parsers and writers together."""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .delegate import escalate_low_conf
from .distributed import run_asyncio, run_dask, run_ray, run_threadpool
from .normalize import normalize_block, normalize_blocks
from .profiles import ProcessingProfile, resolve_profile
from .runtime import NativeKernelRuntime, get_native_runtime
from .refiner import HybridRefiner
from .settings import get_settings
from .escalation import get_escalation_policy
from .parsers import (
    parse_csv,
    parse_docx,
    parse_eml,
    parse_html,
    parse_image,
    parse_ics,
    parse_ini,
    parse_json,
    parse_jsonl,
    parse_md,
    parse_pptx,
    parse_pdf,
    parse_toml,
    parse_txt,
    parse_xlsx,
    parse_yaml,
    stream_image,
    stream_pdf,
)
from .language import detect_language_guesses, merge_language_hints
from .recipe import apply_recipe, apply_recipe_block, load_recipe
from .schema import Block, Document, clone_model
from .sniff import detect_type


# ---- Parser registry ---------------------------------------------------------

ParserFunc = Callable[[str | Path], List[Block]]

class ParserRegistry:
    """Resolve parsers by detected-type or MIME; allow runtime extension."""
    def __init__(self) -> None:
        self.by_key: Dict[str, ParserFunc] = {}
        self.alias_to_key: Dict[str, str] = {}
        # Common MIME → key mapping (fallback to detect_type if not found)
        self.mime_to_key: Dict[str, str] = {
            "text/plain": "text",
            "text/markdown": "md",
            "text/html": "html",
            "text/csv": "csv",
            "application/json": "json",
            "application/ld+json": "json",
            "application/x-ndjson": "jsonl",
            "application/jsonl": "jsonl",
            "application/ndjson": "jsonl",
            "application/x-yaml": "yaml",
            "text/yaml": "yaml",
            "application/yaml": "yaml",
            "application/toml": "toml",
            "application/x-toml": "toml",
            "text/ini": "ini",
            "text/x-ini": "ini",
            "application/ini": "ini",
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "image/png": "image",
            "image/jpeg": "image",
            "image/tiff": "image",
            "image/bmp": "image",
            "image/gif": "image",
            "image/webp": "image",
        }

    def register(self, key: str, func: ParserFunc, *, also: Tuple[str, ...] = ()) -> None:
        self.by_key[key] = func
        self.alias_to_key[key] = key
        for alias in also:
            self.by_key[alias] = func
            self.alias_to_key[alias] = key

    def resolve(self, *, detected: Optional[str], mime: Optional[str]) -> ParserFunc:
        func, _ = self.resolve_with_key(detected=detected, mime=mime)
        return func

    def resolve_with_key(self, *, detected: Optional[str], mime: Optional[str]) -> Tuple[ParserFunc, str]:
        if mime:
            alias = self.mime_to_key.get(mime.split(";")[0].strip())
            if alias:
                key = self.alias_to_key.get(alias, alias)
                func = self.by_key.get(key)
                if func:
                    return func, key
        if detected:
            key = self.alias_to_key.get(detected, detected)
            func = self.by_key.get(key)
            if func:
                return func, key
        fallback = self.by_key.get("text", parse_txt)
        canonical = self.alias_to_key.get("text", "text")
        return fallback, canonical


REGISTRY = ParserRegistry()
REGISTRY.register("text", parse_txt, also=("txt",))
REGISTRY.register("md", parse_md, also=("markdown",))
REGISTRY.register("html", parse_html)
REGISTRY.register("csv", parse_csv)
REGISTRY.register("json", parse_json)
REGISTRY.register("jsonl", parse_jsonl)
REGISTRY.register("ini", parse_ini, also=("cfg", "conf", "properties"))
REGISTRY.register("pdf", parse_pdf)
REGISTRY.register("docx", parse_docx)
REGISTRY.register("pptx", parse_pptx)
REGISTRY.register("xlsx", parse_xlsx)
REGISTRY.register("image", parse_image, also=("png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp"))
REGISTRY.register("eml", parse_eml)
REGISTRY.register("ics", parse_ics)
REGISTRY.register("yaml", parse_yaml, also=("yml",))
REGISTRY.register("toml", parse_toml)

# 外部拡張用ヘルパ（プラグイン等から利用）
def register_parser(key: str, func: ParserFunc, *, also: Tuple[str, ...] = ()) -> None:
    REGISTRY.register(key, func, also=also)


# Streaming implementations for heavy parsers
_STREAMERS: Dict[str, Callable[[str | Path], Iterable[Block]]] = {
    "pdf": stream_pdf,
    "image": stream_image,
}


# ---- Profile orchestration ---------------------------------------------------


@dataclass
class PipelineMetrics:
    """Track step durations and elapsed time during conversion."""

    start: float = field(default_factory=time.perf_counter)
    last_checkpoint: float = field(init=False)
    parse_ms: float = 0.0
    normalize_ms: float = 0.0
    recipe_ms: float = 0.0

    def __post_init__(self) -> None:
        self.last_checkpoint = self.start

    def checkpoint(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self.last_checkpoint) * 1000.0
        self.last_checkpoint = now
        return elapsed

    def spent_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0


class PipelineOrchestrator:
    """Coordinate runtime normalisation and LLM policy for a profile."""

    def __init__(
        self,
        profile: ProcessingProfile,
        *,
        runtime: Optional[NativeKernelRuntime] = None,
    ) -> None:
        self.profile = profile
        if runtime is None:
            runtime = get_native_runtime(
                layout_profile=profile.layout_profile,
                layout_batch_size=profile.layout_batch_size,
            )
        self.runtime = runtime
        self._refiner = HybridRefiner()
        if self.runtime and self.profile.warm_runtime:
            try:
                self.runtime.warm()
            except Exception:  # pragma: no cover - defensive guard
                pass

    # ------------------------------------------------------------------ helpers
    def _normalize(self, blocks: List[Block]) -> List[Block]:
        if not blocks:
            return []
        if self.runtime is None:
            normalized = list(normalize_blocks(blocks))
        elif self.profile.stream_normalize:
            normalized = list(
                self.runtime.normalize_stream(
                    blocks,
                    batch_size=max(1, self.profile.text_batch_size),
                )
            )
        else:
            normalized = self.runtime.normalize(blocks)
        return self._refiner.refine(normalized)

    def _effective_deadline(
        self,
        deadline_ms: Optional[int],
    ) -> Optional[float]:
        deadline = deadline_ms or self.profile.default_deadline_ms
        llm_deadline = self.profile.llm_policy.deadline_ms
        if llm_deadline is None:
            return float(deadline) if deadline is not None else None
        if deadline is None:
            return float(llm_deadline)
        return float(min(deadline, llm_deadline))

    # ------------------------------------------------------------------- convert
    def convert(
        self,
        path: str | Path,
        recipe: str,
        llm_ok: bool = True,
        *,
        mime: Optional[str] = None,
        deadline_ms: Optional[int] = None,
        max_blocks: Optional[int] = None,
    ) -> Document:
        metrics = PipelineMetrics()
        source = Path(path)
        detected = detect_type(source)

        blocks = _parse(source, detected=detected, mime=mime)
        metrics.parse_ms = metrics.checkpoint()

        blocks = self._normalize(blocks)
        metrics.normalize_ms = metrics.checkpoint()

        blocks = apply_recipe(blocks, recipe)
        metrics.recipe_ms = metrics.checkpoint()

        effective_max = max_blocks if max_blocks is not None else self.profile.max_blocks
        effective_deadline = self._effective_deadline(deadline_ms)

        no_llm_env = os.getenv("SR_ADAPTER_NO_LLM", "").strip().lower() in {"1", "true", "yes"}
        policy = self.profile.llm_policy
        do_llm = bool(llm_ok and not no_llm_env and policy.enabled)
        selection = None
        targets: List[int] = []
        if do_llm:
            policy_engine = get_escalation_policy()
            selection = policy_engine.evaluate(
                blocks,
                max_confidence=policy.max_confidence,
                allow_types=policy.limit_block_types,
                limit=policy.max_blocks,
            )
            targets = list(selection.indices)
            if not targets:
                do_llm = False
            elif effective_deadline is not None and metrics.spent_ms() >= effective_deadline:
                do_llm = False

        escalations = 0
        if do_llm:
            result = escalate_low_conf(
                blocks,
                recipe,
                max_confidence=policy.max_confidence,
                allow_types=policy.limit_block_types,
                limit=policy.max_blocks,
                selection=selection,
            )
            escalations = sum(
                1
                for idx in targets
                if idx < len(result)
                and result[idx].attrs.get("llm_escalations")
            )
            blocks = result

        truncated = 0
        if isinstance(effective_max, int) and effective_max > 0 and len(blocks) > effective_max:
            truncated = len(blocks) - effective_max
            blocks = blocks[:effective_max]

        blocks, detected_languages = _annotate_languages(blocks)

        meta = {
            "source": str(source),
            "type": detected,
            "mime": mime or "",
            "metrics_parse_ms": round(metrics.parse_ms, 2),
            "metrics_normalize_ms": round(metrics.normalize_ms, 2),
            "metrics_recipe_ms": round(metrics.recipe_ms, 2),
            "metrics_total_ms": round(metrics.spent_ms(), 2),
            "llm_escalations": int(escalations),
            "truncated_blocks": int(truncated),
            "block_count": int(len(blocks)),
            "env_no_llm": bool(no_llm_env),
            "processing_profile": self.profile.name,
            "llm_policy": policy.to_meta(),
            "runtime_text_enabled": bool(self.runtime and self.runtime.text_enabled),
            "runtime_layout_enabled": bool(self.runtime and self.runtime.layout_enabled),
        }

        meta["languages"] = detected_languages
        if detected_languages:
            meta["primary_language"] = detected_languages[0]

        document = Document(
            blocks=list(blocks),
            meta=meta,
        )
        try:
            from .profile_auto import record_profile_outcome

            if hasattr(meta, "model_dump"):
                payload = meta.model_dump()
            elif hasattr(meta, "dict"):
                payload = meta.dict()  # type: ignore[call-arg]
            elif isinstance(meta, dict):
                payload = dict(meta)
            else:
                payload = {key: value for key, value in dict(meta).items()}  # type: ignore[arg-type]
            record_profile_outcome(self.profile, payload)
        except Exception:  # pragma: no cover - adaptive feedback is best effort
            pass
        return document


# ---- Internal helpers --------------------------------------------------------


def _language_sample(block: Block) -> Optional[str]:
    candidates: List[str] = []
    text = block.text.strip()
    if len(text) >= 12:
        candidates.append(text)
    value = block.attrs.get("value") if block.attrs else None
    if isinstance(value, str):
        trimmed = value.strip()
        if len(trimmed) >= 12:
            candidates.append(trimmed)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _annotate_languages(blocks: Iterable[Block]) -> Tuple[List[Block], List[str]]:
    enriched: List[Block] = []
    votes: Counter[str] = Counter()

    for block in blocks:
        attrs = dict(block.attrs)
        updated = False
        existing_attr_hints = attrs.get("language_hints")
        existing_hints = existing_attr_hints if isinstance(existing_attr_hints, list) else []
        attr_languages_attr = attrs.get("ocr_languages")
        attr_langs = attr_languages_attr if isinstance(attr_languages_attr, list) else []

        sample = _language_sample(block)
        guesses = detect_language_guesses(sample) if sample else []
        guess_langs = [guess.lang for guess in guesses]

        for guess in guesses:
            weight = min(len(sample or ""), 500)
            if weight <= 0:
                weight = 1
            votes[guess.lang] += weight * guess.prob

        for lang in attr_langs:
            if isinstance(lang, str) and lang:
                votes[lang] += 250.0

        merged_hints = merge_language_hints(
            existing_hints,
            [lang for lang in attr_langs if isinstance(lang, str)],
            guess_langs,
        )

        if guesses:
            attrs["language_scores"] = [
                {"lang": guess.lang, "probability": round(guess.prob, 4)} for guess in guesses
            ]
            updated = True
        elif "language_scores" in attrs:
            attrs.pop("language_scores", None)
            updated = True

        if merged_hints:
            attrs["language_hints"] = merged_hints
            updated = True
        elif "language_hints" in attrs:
            attrs.pop("language_hints", None)
            updated = True

        lang_value = block.lang
        if not lang_value:
            if guess_langs:
                lang_value = guess_langs[0]
            elif merged_hints:
                lang_value = merged_hints[0]

        if updated or lang_value != block.lang:
            enriched.append(clone_model(block, attrs=attrs, lang=lang_value))
        else:
            enriched.append(block)

    ranked = [lang for lang, _ in votes.most_common()]
    return enriched, ranked


def _parse(path: Path, *, detected: str, mime: Optional[str]) -> List[Block]:
    parser, key = REGISTRY.resolve_with_key(detected=detected, mime=mime)
    streamer = _STREAMERS.get(key)
    try:
        if streamer:
            return list(streamer(path))
        result = parser(path)
        return list(result) if not isinstance(result, list) else result
    except Exception:
        # 防御的フォールバック：plain text
        return parse_txt(path)


def _stream_raw(path: Path, *, detected: str, mime: Optional[str]) -> Iterable[Block]:
    parser, key = REGISTRY.resolve_with_key(detected=detected, mime=mime)
    streamer = _STREAMERS.get(key)
    if streamer:
        try:
            yield from streamer(path)
            return
        except Exception:
            parser = parse_txt
            key = "text"
    try:
        result = parser(path)
    except Exception:
        result = parse_txt(path)
    if isinstance(result, list):
        for block in result:
            yield block
    else:
        yield from result


def stream_convert(
    path: str | Path,
    recipe: str,
    *,
    mime: Optional[str] = None,
    max_blocks: Optional[int] = None,
) -> Iterator[Block]:
    """Stream blocks through parse→normalise→recipe without LLM escalation."""

    source = Path(path)
    detected = detect_type(source)
    recipe_config = load_recipe(recipe)
    count = 0
    for block in _stream_raw(source, detected=detected, mime=mime):
        normalised = normalize_block(block)
        transformed = apply_recipe_block(normalised, recipe_config)
        yield transformed
        count += 1
        if max_blocks and count >= max_blocks:
            break


# ---- Public API --------------------------------------------------------------

def _profile_context(
    path: str | Path,
    *,
    deadline_ms: Optional[int],
    max_blocks: Optional[int],
    mime: Optional[str],
) -> Dict[str, object]:
    context: Dict[str, object] = {}
    try:
        resolved = Path(path)
        if resolved.exists():
            context["size_bytes"] = resolved.stat().st_size
    except Exception:  # pragma: no cover - file system race
        pass
    if deadline_ms is not None:
        context["deadline_ms"] = int(deadline_ms)
    if max_blocks is not None:
        context["max_blocks"] = int(max_blocks)
    if mime:
        context["mime"] = mime
    return context


def convert(
    path: str | Path,
    recipe: str,
    llm_ok: bool = True,
    *,
    mime: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_blocks: Optional[int] = None,
    profile: str | ProcessingProfile | None = None,
    runtime: Optional[NativeKernelRuntime] = None,
) -> Document:
    """Convert *path* to the unified :class:`Document`."""

    context = _profile_context(
        path,
        deadline_ms=deadline_ms,
        max_blocks=max_blocks,
        mime=mime,
    )
    profile_obj = resolve_profile(profile, context=context)
    orchestrator = PipelineOrchestrator(profile_obj, runtime=runtime)
    return orchestrator.convert(
        path,
        recipe,
        llm_ok=llm_ok,
        mime=mime,
        deadline_ms=deadline_ms,
        max_blocks=max_blocks,
    )


def _build_worker(
    profile_spec: str | ProcessingProfile | None,
    recipe: str,
    *,
    llm_ok: bool,
    mime_by_path: Optional[Dict[str, str]],
    deadline_ms: Optional[int],
    max_blocks: Optional[int],
) -> Callable[[str | Path], Document]:
    def _worker(p: str | Path) -> Document:
        context = _profile_context(
            p,
            deadline_ms=deadline_ms,
            max_blocks=max_blocks,
            mime=(mime_by_path or {}).get(str(p)),
        )
        profile_obj = resolve_profile(profile_spec, context=context)
        runtime = get_native_runtime(
            layout_profile=profile_obj.layout_profile,
            layout_batch_size=profile_obj.layout_batch_size,
        )
        orchestrator = PipelineOrchestrator(profile_obj, runtime=runtime)
        return orchestrator.convert(
            p,
            recipe=recipe,
            llm_ok=llm_ok,
            mime=(mime_by_path or {}).get(str(p)),
            deadline_ms=deadline_ms,
            max_blocks=max_blocks,
        )

    return _worker


def batch_convert(
    paths: Iterable[str | Path],
    recipe: str,
    llm_ok: bool = True,
    *,
    mime_by_path: Optional[Dict[str, str]] = None,
    deadline_ms: Optional[int] = None,
    max_blocks: Optional[int] = None,
    concurrency: int = 0,
    profile: str | ProcessingProfile | None = None,
    backend: str | None = None,
    dask_scheduler: str | None = None,
    ray_address: str | None = None,
) -> List[Document]:
    """Convert multiple paths; optional thread parallelism for I/O bound workloads."""
    items = list(paths)
    settings = get_settings()
    dist_settings = settings.distributed
    worker_count = concurrency if concurrency > 0 else (dist_settings.max_workers or 0)
    chosen_backend = (backend or dist_settings.default_backend or "auto").lower()
    if chosen_backend == "auto":
        chosen_backend = "threadpool" if worker_count and worker_count > 1 else "sync"

    worker = _build_worker(
        profile,
        recipe,
        llm_ok=llm_ok,
        mime_by_path=mime_by_path,
        deadline_ms=deadline_ms,
        max_blocks=max_blocks,
    )

    if chosen_backend in {"sync", "sequential", "none"}:
        return [worker(path) for path in items]

    if chosen_backend in {"thread", "threads", "threadpool"}:
        workers = worker_count or 4
        return run_threadpool(worker, items, workers=workers)

    if chosen_backend in {"async", "asyncio"}:
        return run_asyncio(worker, items, workers=worker_count or None)

    if chosen_backend == "dask":
        scheduler = dask_scheduler or dist_settings.dask_scheduler
        workers = worker_count or None
        return run_dask(worker, items, scheduler=scheduler, workers=workers)

    if chosen_backend == "ray":
        address = ray_address or dist_settings.ray_address
        workers = worker_count or None
        return run_ray(worker, items, address=address, workers=workers)

    raise ValueError(f"Unknown batch_convert backend '{chosen_backend}'")
