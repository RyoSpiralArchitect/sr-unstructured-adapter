"""High level conversion pipeline wiring parsers and writers together."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .delegate import escalate_low_conf
from .normalize import normalize_blocks
from .parsers import (
    parse_csv,
    parse_docx,
    parse_eml,
    parse_html,
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
)
from .recipe import apply_recipe
from .schema import Block, Document
from .sniff import detect_type


# ---- Parser registry ---------------------------------------------------------

ParserFunc = Callable[[str | Path], List[Block]]

class ParserRegistry:
    """Resolve parsers by detected-type or MIME; allow runtime extension."""
    def __init__(self) -> None:
        self.by_key: Dict[str, ParserFunc] = {}
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
        }

    def register(self, key: str, func: ParserFunc, *, also: Tuple[str, ...] = ()) -> None:
        self.by_key[key] = func
        for alias in also:
            self.by_key[alias] = func

    def resolve(self, *, detected: Optional[str], mime: Optional[str]) -> ParserFunc:
        if mime:
            key = self.mime_to_key.get(mime.split(";")[0].strip())
            if key and key in self.by_key:
                return self.by_key[key]
        if detected and detected in self.by_key:
            return self.by_key[detected]
        return self.by_key.get("text", parse_txt)


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
REGISTRY.register("eml", parse_eml)
REGISTRY.register("ics", parse_ics)
REGISTRY.register("yaml", parse_yaml, also=("yml",))
REGISTRY.register("toml", parse_toml)

# 外部拡張用ヘルパ（プラグイン等から利用）
def register_parser(key: str, func: ParserFunc, *, also: Tuple[str, ...] = ()) -> None:
    REGISTRY.register(key, func, also=also)


# ---- Internal helpers --------------------------------------------------------

def _parse(path: Path, *, detected: str, mime: Optional[str]) -> List[Block]:
    parser = REGISTRY.resolve(detected=detected, mime=mime)
    try:
        return parser(path)
    except Exception:
        # 防御的フォールバック：plain text
        return parse_txt(path)


# ---- Public API --------------------------------------------------------------

def convert(
    path: str | Path,
    recipe: str,
    llm_ok: bool = True,
    *,
    mime: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    max_blocks: Optional[int] = None,
) -> Document:
    """Convert *path* to the unified :class:`Document`.

    Parameters
    ----------
    recipe:
        apply_recipe に渡す処理レシピ名。
    llm_ok:
        低信頼ブロックの LLM エスカレーションを許可するか。
        環境変数 SR_ADAPTER_NO_LLM が真の場合は強制無効。
    mime:
        既知の場合は MIME を指定。未指定なら sniff/detect に委譲。
    deadline_ms:
        ここで指定した予算（ミリ秒）を超えたら LLM エスカレーションをスキップ。
    max_blocks:
        出力ブロックの上限（超えた分は切り捨てて meta に記録）。
    """

    t0 = time.perf_counter()
    source = Path(path)
    detected = detect_type(source)

    blocks = _parse(source, detected=detected, mime=mime)
    t_parse = (time.perf_counter() - t0) * 1000.0

    blocks = normalize_blocks(blocks)
    t_norm = (time.perf_counter() - t0) * 1000.0

    blocks = apply_recipe(blocks, recipe)
    t_recipe = (time.perf_counter() - t0) * 1000.0

    # ガード：環境や締切で LLM エスカレーションを抑制
    no_llm_env = os.getenv("SR_ADAPTER_NO_LLM", "").strip().lower() in {"1", "true", "yes"}
    do_llm = bool(llm_ok and not no_llm_env)
    if do_llm and deadline_ms is not None:
        # 予算が既に尽きていればスキップ
        spent = (time.perf_counter() - t0) * 1000.0
        if spent >= float(deadline_ms):
            do_llm = False

    escalations = 0
    if do_llm:
        before = len(blocks)
        blocks = escalate_low_conf(blocks, recipe)
        escalations = max(0, len(blocks) - before)
    t_all = (time.perf_counter() - t0) * 1000.0

    # ハードキャップ
    truncated = 0
    if isinstance(max_blocks, int) and max_blocks > 0 and len(blocks) > max_blocks:
        truncated = len(blocks) - max_blocks
        blocks = blocks[:max_blocks]

    document = Document(
        blocks=list(blocks),
        meta={
            "source": str(source),
            "type": detected,
            "mime": mime or "",
            "metrics_parse_ms": round(t_parse, 2),
            "metrics_normalize_ms": round(t_norm - t_parse, 2),
            "metrics_recipe_ms": round(t_recipe - t_norm, 2),
            "metrics_total_ms": round(t_all, 2),
            "llm_escalations": int(escalations),
            "truncated_blocks": int(truncated),
            "block_count": int(len(blocks)),
            "env_no_llm": bool(no_llm_env),
        },
    )
    return document


def batch_convert(
    paths: Iterable[str | Path],
    recipe: str,
    llm_ok: bool = True,
    *,
    mime_by_path: Optional[Dict[str, str]] = None,
    deadline_ms: Optional[int] = None,
    max_blocks: Optional[int] = None,
    concurrency: int = 0,
) -> List[Document]:
    """Convert multiple paths; optional thread parallelism for I/O bound workloads."""
    items = list(paths)
    if concurrency and concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        def _one(p: str | Path) -> Document:
            m = (mime_by_path or {}).get(str(p))
            return convert(p, recipe=recipe, llm_ok=llm_ok, mime=m,
                           deadline_ms=deadline_ms, max_blocks=max_blocks)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(_one, items))
    else:
        return [
            convert(p, recipe=recipe, llm_ok=llm_ok,
                    mime=(mime_by_path or {}).get(str(p)),
                    deadline_ms=deadline_ms, max_blocks=max_blocks)
            for p in items
        ]