# SPDX-License-Identifier: Apache-2.0
"""Shared prompt construction and result I/O for the trace experiments."""

# Standard
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_MODEL_LEN = 8000


def load_trace(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(result, temp, ensure_ascii=False, indent=2, allow_nan=False)
            temp.write("\n")
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def build_prompt(tokenizer, sys_prompt, urls, current_docs, question) -> list[int]:
    """Preserve the paper's [BOS+system][documents][question] token layout."""
    token_ids = list(tokenizer.encode(sys_prompt))
    for url in urls:
        token_ids.extend(tokenizer.encode(current_docs[url])[1:])
    token_ids.extend(tokenizer.encode(question)[1:])
    return token_ids
