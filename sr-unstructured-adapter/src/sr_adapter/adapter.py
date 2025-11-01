# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================
from __future__ import annotations
import json, mimetypes, os, sys

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def to_unified_payload(path: str) -> dict:
    mime = _guess_mime(path)
    txt = _read_text(path) if mime.startswith("text") or path.endswith(('.md','.json')) else ''
    return {
        "source": path,
        "mime": mime,
        "text": txt,
        "meta": {"size": os.path.getsize(path)}
    }

def to_llm_messages(payload: dict) -> list[dict]:
    # Minimal “ready-to-use” chat format (extend later as needed)
    content = f"[{payload['mime']}] {payload['text'][:4000] if isinstance(payload['text'], str) else ''}"
    return [{"role": "user", "content": content}]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m sr_adapter.adapter <file1> [file2 ...]", file=sys.stderr)
        sys.exit(2)
    for p in sys.argv[1:]:
        print(json.dumps(to_unified_payload(p), ensure_ascii=False))
