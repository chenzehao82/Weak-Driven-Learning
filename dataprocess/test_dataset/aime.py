#!/usr/bin/env python3
"""Download and validate the pinned AIME 2025 evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY = "opencompass/AIME2025"
SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
EXPECTED_ROWS = 30
EXPECTED_OUTPUT_SHA256 = (
    "de1b2907208f7e7302825a16af356e5f3782401e9c51150a46d83240e4f3db97"
)
EXPECTED_SEMANTIC_SHA256 = (
    "acf4548122ca97493cb6f1c6d9ca4e62aacdae3f127e636f49cd2bf92bbab806"
)
SOURCES = (
    {
        "path": "opencompass/AIME2025/aime2025-I.jsonl",
        "url": (
            "https://huggingface.co/datasets/opencompass/AIME2025/resolve/"
            f"{SOURCE_COMMIT}/aime2025-I.jsonl"
        ),
        "sha256": "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e",
    },
    {
        "path": "opencompass/AIME2025/aime2025-II.jsonl",
        "url": (
            "https://huggingface.co/datasets/opencompass/AIME2025/resolve/"
            f"{SOURCE_COMMIT}/aime2025-II.jsonl"
        ),
        "sha256": "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e",
    },
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "aime2025" / "test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional local source tree containing opencompass/AIME2025/*.jsonl; "
            "otherwise the pinned Hugging Face files are downloaded."
        ),
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def semantic_sha256(rows: list[dict[str, Any]]) -> str:
    pairs = [[row["question"].strip(), str(row["answer"]).strip()] for row in rows]
    payload = json.dumps(
        pairs,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def read_source(source: dict[str, str], source_root: Path | None) -> bytes:
    if source_root is not None:
        path = source_root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source file: {path}")
        return path.read_bytes()

    request = urllib.request.Request(
        source["url"],
        headers={"User-Agent": "Weak-Driven-Learning-eval-data/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def parse_source(payload: bytes, source_name: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"source must contain JSON objects: {source_name}")

    normalized = []
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if question is None or answer is None:
            raise ValueError(
                f"{source_name} row {index} is missing question or answer"
            )
        question = str(question).strip()
        answer = str(answer).strip()
        if not question or not answer:
            raise ValueError(
                f"{source_name} row {index} has an empty question or answer"
            )
        normalized.append({"question": question, "answer": answer})
    return normalized


def prepare_aime2025(output: Path, source_root: Path | None = None) -> None:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        payload = read_source(source, source_root)
        actual_sha256 = sha256_bytes(payload)
        if actual_sha256 != source["sha256"]:
            raise RuntimeError(
                f"source sha256 mismatch for {source['path']}: "
                f"{actual_sha256} != {source['sha256']}"
            )
        rows.extend(parse_source(payload, source["path"]))

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"AIME2025 should contain {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    actual_semantic_sha256 = semantic_sha256(rows)
    if actual_semantic_sha256 != EXPECTED_SEMANTIC_SHA256:
        raise RuntimeError(
            "assembled AIME2025 semantic sha256 mismatch: "
            f"{actual_semantic_sha256} != {EXPECTED_SEMANTIC_SHA256}"
        )

    serialized = (json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    actual_output_sha256 = sha256_bytes(serialized)
    if actual_output_sha256 != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(
            "assembled AIME2025 file sha256 mismatch: "
            f"{actual_output_sha256} != {EXPECTED_OUTPUT_SHA256}"
        )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    temporary.write_bytes(serialized)
    os.replace(temporary, output)
    print(
        f"[OK] {REPOSITORY}@{SOURCE_COMMIT} rows={len(rows)} "
        f"sha256={actual_output_sha256} path={output}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    if source_root is not None and not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    prepare_aime2025(args.output, source_root)


if __name__ == "__main__":
    main()
