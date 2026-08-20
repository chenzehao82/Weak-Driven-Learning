#!/usr/bin/env python3
"""Portable, reproducible vLLM evaluation for the WDL math benchmarks.

The default is the paper's deterministic evaluation protocol. The ``opd``
protocol exposes the prompt and length contract used by recent Qwen3 OPD
checkpoint evaluations. Sampled evaluation submits each problem once with
``SamplingParams(n=K)`` and reports mean@K and observed pass@K separately.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional


# Keep imports working both from the repository root and from ``scripts/``.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prompts import SYSTEM_PROMPT  # noqa: E402


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
EXPECTED_ROWS = {
    "math500": 500,
    "aime2025": 30,
    "amc23": 40,
    "aqua": 254,
    "gsm8k": 1319,
    "mawps": 238,
    "svamp": 1000,
}

# Paper evaluation reserves up to 4K prompt tokens. MATH500 uses 4K output;
# AIME2025 and AMC23 use 8K output.
PAPER_DATASET_LIMITS = {
    "math500": {"max_input_tokens": 4096, "max_new_tokens": 4096, "max_model_len": 8192},
    "aime2025": {"max_input_tokens": 4096, "max_new_tokens": 8192, "max_model_len": 12288},
    "amc23": {"max_input_tokens": 4096, "max_new_tokens": 8192, "max_model_len": 12288},
}
PAPER_DEFAULT_LIMITS = {
    "max_input_tokens": 4096,
    "max_new_tokens": 4096,
    "max_model_len": 8192,
}

# The recent three-set OPD runs used tighter input reservations with the same
# output budgets. Keeping this as a named preset prevents silent protocol drift.
OPD_DATASET_LIMITS = {
    "math500": {"max_input_tokens": 1024, "max_new_tokens": 4096, "max_model_len": 5120},
    "aime2025": {"max_input_tokens": 2048, "max_new_tokens": 8192, "max_model_len": 10240},
    "amc23": {"max_input_tokens": 2048, "max_new_tokens": 8192, "max_model_len": 10240},
}
OPD_DEFAULT_LIMITS = {
    "max_input_tokens": 1024,
    "max_new_tokens": 4096,
    "max_model_len": 5120,
}
QUESTION_FIELDS = (
    "question",
    "instruction",
    "problem",
    "input",
    "original_text",
    "prompt",
)
ANSWER_FIELDS = ("answer", "target", "label", "solution", "output", "ans")
QWEN_STOP_TOKENS = {151643: "<|endoftext|>", 151645: "<|im_end|>"}
OPD_GRADER_SHA256 = (
    "6e7f8ea703258c051e4c28379443416a485046c235196f4ee25a244c216e994c"
)
OPD_RUNTIME_ENV = {
    "VLLM_USE_DEEP_GEMM": "0",
    "VLLM_DEEP_GEMM_WARMUP": "skip",
    "TOKENIZERS_PARALLELISM": "false",
}


def normalize_dataset_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    aliases = {
        "aime": "aime2025",
        "aime25": "aime2025",
        "math500test": "math500",
        "amc": "amc23",
    }
    return aliases.get(normalized, normalized)


def get_last_checkpoint(folder: str | os.PathLike[str]) -> Optional[str]:
    """Return the numerically latest checkpoint directory, if one exists."""
    checkpoints = [
        path
        for path in Path(folder).iterdir()
        if path.is_dir() and _RE_CHECKPOINT.fullmatch(path.name)
    ]
    if not checkpoints:
        return None
    return str(
        max(
            checkpoints,
            key=lambda path: int(_RE_CHECKPOINT.fullmatch(path.name).group(1)),
        )
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in row and row[field] is not None:
            value = row[field]
            # Numeric zero is a valid answer.
            if not isinstance(value, str) or value.strip():
                return value
    return None


def _prompt_to_question(value: Any, *, require_user_only: bool = False) -> str:
    """Normalize a plain prompt or a role/content message sequence."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, (list, dict)):
            value = decoded
        else:
            return value
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        user_messages = []
        roles = []
        for message in value:
            mapped = _as_mapping(message)
            if mapped is None or not isinstance(mapped.get("content"), str):
                raise ValueError("prompt contains an invalid role/content message")
            role = str(mapped.get("role", "user"))
            roles.append(role)
            if role == "user":
                user_messages.append(mapped["content"])
        if require_user_only and roles != ["user"]:
            raise ValueError(f"OPD prompt must contain one user message, got roles={roles}")
        if len(user_messages) != 1:
            raise ValueError("expected exactly one user message in prompt")
        return user_messages[0]
    raise ValueError(f"unsupported prompt type: {type(value).__name__}")


def _read_raw_dataset(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        rows = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("data") if isinstance(payload, Mapping) else payload
    else:
        raise ValueError("dataset must be .json, .jsonl, or .parquet")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"dataset must contain a list of objects: {path}")
    return rows


def load_general_dataset(
    path: str | os.PathLike[str],
    *,
    dataset_name: Optional[str] = None,
    validate_count: bool = False,
    protocol: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load public JSON/JSONL and the prompt/reward_model OPD schema.

    Malformed rows are rejected instead of silently disappearing from the
    denominator, which was a reproducibility bug in the previous evaluator.
    """
    source = Path(path)
    raw_rows = _read_raw_dataset(source)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        reward_model = _as_mapping(row.get("reward_model"))
        if protocol == "opd" and row.get("prompt") is not None:
            question = row["prompt"]
        elif reward_model is not None and row.get("prompt") is not None:
            question = row["prompt"]
        else:
            question = _first_present(row, QUESTION_FIELDS)
        answer = reward_model.get("ground_truth") if reward_model else None
        if answer is None:
            answer = _first_present(row, ANSWER_FIELDS)
        if question is None or answer is None:
            missing = []
            if question is None:
                missing.append("question")
            if answer is None:
                missing.append("answer")
            raise ValueError(f"{source}: row {index} lacks {', '.join(missing)}")
        normalized_question = _prompt_to_question(
            question, require_user_only=protocol == "opd"
        ).strip()
        if not normalized_question:
            raise ValueError(f"{source}: row {index} has an empty question")
        normalized_answer = str(answer).strip()
        if not normalized_answer:
            raise ValueError(f"{source}: row {index} has an empty answer")
        normalized.append(
            {
                "id": str(row.get("unique_id", row.get("id", index))),
                "position": index,
                "question": normalized_question,
                "answer": normalized_answer,
            }
        )
    if not normalized:
        raise ValueError(f"dataset is empty: {source}")
    if validate_count and dataset_name:
        canonical_name = normalize_dataset_name(dataset_name)
        expected = EXPECTED_ROWS.get(canonical_name)
        if expected is not None and len(normalized) != expected:
            raise ValueError(
                f"{canonical_name} requires {expected} rows, found {len(normalized)}"
            )
    return normalized


def apply_chat_template(
    tokenizer: Any,
    question: str,
    thinking: bool = True,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    *,
    strict: bool = False,
) -> str:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(**kwargs, enable_thinking=thinking)
    except (TypeError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                "tokenizer cannot render the locked thinking chat template"
            ) from exc
        try:
            return tokenizer.apply_chat_template(**kwargs)
        except (AttributeError, TypeError, ValueError):
            parts = [f"{message['role']}: {message['content']}" for message in messages]
            parts.append("assistant:")
            return "\n\n".join(parts)


def extract_answer_number(sentence: str) -> float:
    matches = re.findall(r"-?\d+(?:\.\d+)?", sentence.replace(",", ""))
    if not matches:
        return float("inf")
    try:
        return float(matches[-1])
    except ValueError:
        return float("inf")


def extract_answer_letter(sentence: str) -> str:
    return _extract_choice(sentence) or ""


def verify_simple_number(pred_text: str, gold: str, miss: float = 1e-3) -> Optional[bool]:
    try:
        predicted = extract_answer_number(pred_text)
        return predicted != float("inf") and abs(float(gold) - predicted) <= miss
    except (TypeError, ValueError):
        return None


def verify_simple_letter(pred_text: str, gold: str) -> Optional[bool]:
    predicted = extract_answer_letter(pred_text)
    return bool(predicted) and predicted == gold.strip().upper()


def _extract_choice(text: str) -> Optional[str]:
    patterns = (
        r"\\boxed\{\s*([A-E])\s*\}",
        r"(?:final\s+)?answer\s*(?:is|:)?\s*\(?([A-E])\)?\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def _parse_answer_choices(question: str) -> dict[str, str]:
    matches = re.findall(
        r"\(([A-E])\)\s*([^(]+?)(?=\s*\([A-E]\)|\s*$)", question
    )
    return {letter: value.strip().rstrip(".") for letter, value in matches}


def _math_verify(prediction: str, answer: str) -> Optional[bool]:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    gold_parsed = parse(f"${answer.strip().strip('$')}$", extraction_mode="first_match")
    if not gold_parsed:
        return None
    prediction_parsed = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="all",
    )
    return bool(verify(gold_parsed, prediction_parsed))


def _match_value_to_choice(
    prediction: str, question: str, gold_letter: str
) -> Optional[bool]:
    choices = _parse_answer_choices(question)
    boxed_values = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", prediction)
    if not choices or not boxed_values:
        return None
    predicted_value = boxed_values[-1].strip()
    for letter, choice_value in choices.items():
        try:
            equivalent = _math_verify(f"\\boxed{{{predicted_value}}}", choice_value)
        except Exception:
            equivalent = predicted_value == choice_value
        if equivalent:
            return letter == gold_letter
    return None


def verify_with_latex(
    idx: str, prediction: str, answer: str, question: str = ""
) -> Optional[bool]:
    """Canonical public scorer, including AQuA choice handling."""
    del idx
    gold = answer.strip().strip("$").upper()
    if len(gold) == 1 and gold in "ABCDE":
        choice = _extract_choice(prediction)
        if choice is not None:
            return choice == gold
        matched = _match_value_to_choice(prediction, question, gold)
        if matched is not None:
            return matched
    try:
        return _math_verify(prediction, answer)
    except Exception as exc:
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def get_verification_method(dataset_name: str) -> str:
    """Kept for callers of the previous public evaluator."""
    name = normalize_dataset_name(dataset_name)
    if name in {"gsm8k", "svamp", "mawps", "multiarith", "addsub", "singleeq"}:
        return "simple_number"
    if name == "aqua":
        return "multiple_choice"
    return "latex"


def should_use_chat_template(dataset_name: str) -> bool:
    """All formal WDL evaluations use the model chat template."""
    del dataset_name
    return True


def load_external_grader(path: str | os.PathLike[str]) -> Callable[[str, str], bool]:
    grader_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "wdl_external_math_grader",
        grader_path,
        submodule_search_locations=[str(grader_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import grader from {grader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "grade_answer_verl"):
        return lambda response, answer: bool(module.grade_answer_verl(response, answer))
    if hasattr(module, "compute_score"):

        def compute_score(response: str, answer: str) -> bool:
            result = module.compute_score(response, answer)
            return bool(result.get("acc")) if isinstance(result, Mapping) else bool(result)

        return compute_score
    raise AttributeError(f"grader lacks grade_answer_verl/compute_score: {grader_path}")


def configure_runtime_environment(protocol: str) -> dict[str, str]:
    """Set runtime flags before vLLM is imported and reject conflicting OPD values."""
    required = OPD_RUNTIME_ENV if protocol == "opd" else {}
    for name, value in required.items():
        existing = os.environ.get(name)
        if existing is not None and existing.lower() != value.lower():
            raise ValueError(
                f"{protocol} runtime requires {name}={value}, found {existing}"
            )
        os.environ[name] = value
    return {name: os.environ[name] for name in required}


def grade_with_timeout(
    grader: Callable[[str, str], bool],
    response: str,
    answer: str,
    timeout_seconds: int,
) -> tuple[Optional[bool], Optional[str]]:
    """Run the symbolic external grader with a Linux alarm timeout."""

    def handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("grader timed out")

    if not hasattr(signal, "SIGALRM"):
        try:
            return bool(grader(response, answer)), None
        except Exception as exc:
            return None, f"{type(exc).__name__}:{exc}"
    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(timeout_seconds)
    try:
        return bool(grader(response, answer)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class VLLMBackend:
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int,
        max_model_len: int,
        max_new_tokens: int,
        dtype: str,
        gpu_memory_utilization: float,
        seed: int,
        temperature: float,
        top_p: float,
        n_samples: int,
        stop_token_ids: Optional[list[int]],
        top_k: int,
        repetition_penalty: float,
        skip_special_tokens: bool,
        hf_cache: Optional[str],
        enforce_eager: bool,
        trust_remote_code: bool,
    ) -> None:
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": model,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "seed": seed,
            "trust_remote_code": trust_remote_code,
            "enable_prefix_caching": False,
        }
        if hf_cache:
            llm_kwargs["download_dir"] = hf_cache
        self.LLM = LLM(**llm_kwargs)
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "n": n_samples,
            "seed": seed,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "skip_special_tokens": skip_special_tokens,
        }
        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.n_samples = n_samples

    def generate(self, prompts: Sequence[str]) -> list[list[dict[str, Any]]]:
        outputs = self.LLM.generate(list(prompts), self.sampling_params)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} prompt outputs; expected {len(prompts)}"
            )
        matrix = []
        for output in outputs:
            candidates = []
            for candidate in output.outputs:
                stop_reason: Any = getattr(candidate, "stop_reason", None)
                if not isinstance(stop_reason, (str, int, float, bool, type(None))):
                    stop_reason = str(stop_reason)
                candidates.append(
                    {
                        "text": candidate.text,
                        "token_ids": list(getattr(candidate, "token_ids", [])),
                        "finish_reason": str(getattr(candidate, "finish_reason", "") or ""),
                        "stop_reason": stop_reason,
                    }
                )
            matrix.append(candidates)
        for index, candidates in enumerate(matrix):
            if len(candidates) != self.n_samples:
                raise RuntimeError(
                    f"prompt {index} returned {len(candidates)} outputs; expected {self.n_samples}"
                )
        return matrix


def dataset_limits(dataset_name: str, protocol: str = "paper") -> dict[str, int]:
    if protocol == "paper":
        profiles = PAPER_DATASET_LIMITS
        fallback = PAPER_DEFAULT_LIMITS
    elif protocol == "opd":
        profiles = OPD_DATASET_LIMITS
        fallback = OPD_DEFAULT_LIMITS
    else:
        raise ValueError(f"unsupported protocol: {protocol}")
    return dict(profiles.get(normalize_dataset_name(dataset_name), fallback))


def resolve_decoding(args: argparse.Namespace, dataset_name: str) -> dict[str, Any]:
    limits = dataset_limits(dataset_name, args.protocol)
    defaults = {
        "greedy": {"temperature": 0.0, "n_samples": 1},
        "pass1": {"temperature": 0.5, "n_samples": 1},
        "passk": {"temperature": 0.5, "n_samples": 8},
    }[args.mode]
    temperature = defaults["temperature"] if args.temperature is None else args.temperature
    n_samples = defaults["n_samples"] if args.num_samples is None else args.num_samples
    top_p = 1.0 if args.top_p is None else args.top_p
    if args.mode in {"greedy", "pass1"} and n_samples != 1:
        raise ValueError(f"{args.mode} requires exactly one sample per prompt")
    if args.mode == "greedy" and temperature != 0:
        raise ValueError("greedy mode requires --temperature 0")
    if args.mode == "passk" and n_samples < 2:
        raise ValueError("passk requires --num-samples >= 2")
    if temperature == 0 and n_samples > 1:
        raise ValueError("multiple identical greedy samples do not define pass@k")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be non-negative and top-p must be in (0, 1]")
    if args.repetition_penalty <= 0:
        raise ValueError("repetition-penalty must be positive")
    max_input_tokens = args.max_input_tokens or limits["max_input_tokens"]
    max_new_tokens = args.max_new_tokens or limits["max_new_tokens"]
    max_model_len = args.max_model_len or limits["max_model_len"]
    if max_input_tokens + max_new_tokens > max_model_len:
        raise ValueError(
            "max-input-tokens + max-new-tokens must not exceed max-model-len"
        )
    stop_token_ids = args.stop_token_ids
    if stop_token_ids is None and args.protocol == "opd":
        stop_token_ids = list(QWEN_STOP_TOKENS)
    if args.protocol == "opd":
        locked_mismatches = {}
        canonical_dataset = normalize_dataset_name(dataset_name)
        expected_limits = OPD_DATASET_LIMITS.get(canonical_dataset)
        if expected_limits is None:
            locked_mismatches["dataset"] = canonical_dataset
        else:
            actual_limits = {
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "max_model_len": max_model_len,
            }
            if actual_limits != expected_limits:
                locked_mismatches["lengths"] = actual_limits
        expected_mode = {
            "greedy": {"temperatures": {0.0}, "n_samples": 1},
            "pass1": {"temperatures": {0.1, 0.5}, "n_samples": 1},
            "passk": {"temperatures": {0.5}, "n_samples": 8},
        }[args.mode]
        if temperature not in expected_mode["temperatures"]:
            locked_mismatches["temperature"] = temperature
        if n_samples != expected_mode["n_samples"]:
            locked_mismatches["n_samples"] = n_samples
        if top_p != 1.0:
            locked_mismatches["top_p"] = top_p
        if args.seed != 42:
            locked_mismatches["seed"] = args.seed
        if args.tp != 1:
            locked_mismatches["tensor_parallel_size"] = args.tp
        if args.dtype != "bfloat16":
            locked_mismatches["dtype"] = args.dtype
        if stop_token_ids != list(QWEN_STOP_TOKENS):
            locked_mismatches["stop_token_ids"] = stop_token_ids
        if args.top_k != -1:
            locked_mismatches["top_k"] = args.top_k
        if args.repetition_penalty != 1.0:
            locked_mismatches["repetition_penalty"] = args.repetition_penalty
        if args.skip_special_tokens is True:
            locked_mismatches["skip_special_tokens"] = True
        if args.trust_remote_code is False:
            locked_mismatches["trust_remote_code"] = False
        if locked_mismatches:
            raise ValueError(f"OPD decoding contract mismatch: {locked_mismatches}")
    return {
        "mode": args.mode,
        "temperature": temperature,
        "top_p": top_p,
        "n_samples": n_samples,
        "seed": args.seed,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "stop_token_ids": stop_token_ids,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "skip_special_tokens": (
            args.skip_special_tokens
            if args.skip_special_tokens is not None
            else args.protocol != "opd"
        ),
        "trust_remote_code": (
            args.trust_remote_code
            if args.trust_remote_code is not None
            else args.protocol == "opd"
        ),
    }


def _validate_stop_tokens(tokenizer: Any, protocol: str, stop_ids: Optional[list[int]]) -> None:
    if protocol != "opd" or stop_ids != list(QWEN_STOP_TOKENS):
        return
    actual = {token_id: tokenizer.convert_ids_to_tokens(token_id) for token_id in stop_ids}
    if actual != QWEN_STOP_TOKENS:
        raise ValueError(
            f"OPD stop-token contract expects {QWEN_STOP_TOKENS}, tokenizer returned {actual}"
        )


def _token_count(tokenizer: Any, prompt: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def summarize_rows(rows: Sequence[Mapping[str, Any]], n_samples: int) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["position"]), []).append(row)
    if not grouped:
        raise ValueError("no evaluation rows to summarize")
    expected_indices = list(range(n_samples))
    for position, prompt_rows in grouped.items():
        prompt_rows.sort(key=lambda row: int(row["sample_index"]))
        indices = [int(row["sample_index"]) for row in prompt_rows]
        if indices != expected_indices:
            raise RuntimeError(
                f"position {position} sample indices must be {expected_indices}, got {indices}"
            )
    generation_count = len(rows)
    prompt_count = len(grouped)
    correct_generations = sum(int(bool(row["correct"])) for row in rows)
    first_correct = sum(
        int(bool(prompt_rows[0]["correct"])) for prompt_rows in grouped.values()
    )
    passed_prompts = sum(
        int(any(bool(row["correct"]) for row in prompt_rows))
        for prompt_rows in grouped.values()
    )
    mean_at_n = correct_generations / generation_count
    first_sample_accuracy = first_correct / prompt_count
    # A valid pass@1 is produced by a separate n=1 run. Sample zero from an
    # n>1 request is retained only as a diagnostic and is not labelled pass@1.
    pass_at_1 = first_sample_accuracy if n_samples == 1 else None
    pass_at_n = passed_prompts / prompt_count
    return {
        "n_prompts": prompt_count,
        "n_generations": generation_count,
        "n_samples": n_samples,
        "correct_generations": correct_generations,
        "first_sample_correct": first_correct,
        "first_sample_accuracy": first_sample_accuracy,
        "passed_prompts": passed_prompts,
        "mean_accuracy": mean_at_n,
        f"mean_at_{n_samples}": mean_at_n,
        "pass_at_1": pass_at_1,
        f"pass_at_{n_samples}": pass_at_n,
        "observed_pass_at_k": pass_at_n,
        # Backward-compatible names from the original public script.
        "acc": mean_at_n,
        "correct": correct_generations,
        "total": generation_count,
        "repeat": n_samples,
        "pass_at_k": pass_at_n,
        "pass_success": passed_prompts,
        "orig_total": prompt_count,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluate(
    dataset_path: str,
    model_path: str,
    *,
    dataset_name: str,
    tp: int,
    out_csv: str,
    out_json: str,
    protocol: str,
    decoding: Mapping[str, Any],
    thinking: bool,
    system_prompt: Optional[str],
    dtype: str,
    gpu_memory_utilization: float,
    hf_cache: Optional[str],
    enforce_eager: bool,
    validate_count: bool,
    expected_dataset_sha256: Optional[str],
    grader_path: Optional[str],
    expected_grader_sha256: Optional[str],
    grader_timeout_seconds: int,
    batch_size: int,
) -> dict[str, Any]:
    locked_runtime_env = configure_runtime_environment(protocol)
    dataset_sha256 = sha256_file(dataset_path)
    if expected_dataset_sha256 and dataset_sha256 != expected_dataset_sha256.lower():
        raise ValueError(
            "dataset sha256 mismatch: "
            f"{dataset_sha256} != {expected_dataset_sha256.lower()}"
        )
    samples = load_general_dataset(
        dataset_path,
        dataset_name=dataset_name,
        validate_count=validate_count,
        protocol=protocol,
    )
    grader_sha256 = sha256_file(grader_path) if grader_path else None
    if expected_grader_sha256 and grader_sha256 != expected_grader_sha256.lower():
        raise ValueError(
            f"grader sha256 mismatch: {grader_sha256} != {expected_grader_sha256.lower()}"
        )
    grader = load_external_grader(grader_path) if grader_path else None
    engine = VLLMBackend(
        model_path,
        tensor_parallel_size=tp,
        max_model_len=int(decoding["max_model_len"]),
        max_new_tokens=int(decoding["max_new_tokens"]),
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=int(decoding["seed"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        n_samples=int(decoding["n_samples"]),
        stop_token_ids=decoding["stop_token_ids"],
        top_k=int(decoding["top_k"]),
        repetition_penalty=float(decoding["repetition_penalty"]),
        skip_special_tokens=bool(decoding["skip_special_tokens"]),
        hf_cache=hf_cache,
        enforce_eager=enforce_eager,
        trust_remote_code=bool(decoding["trust_remote_code"]),
    )
    tokenizer = engine.LLM.get_tokenizer()
    _validate_stop_tokens(tokenizer, protocol, decoding["stop_token_ids"])
    prompts = [
        apply_chat_template(
            tokenizer,
            sample["question"],
            thinking=thinking,
            system_prompt=system_prompt,
            strict=protocol == "opd",
        )
        for sample in samples
    ]
    if protocol == "opd" and any("<|im_start|>system" in prompt for prompt in prompts):
        raise RuntimeError("OPD prompt unexpectedly contains a system message")
    prompt_lengths = [_token_count(tokenizer, prompt) for prompt in prompts]
    too_long = [
        (index, length)
        for index, length in enumerate(prompt_lengths)
        if length > int(decoding["max_input_tokens"])
    ]
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts exceed max-input-tokens="
            f"{decoding['max_input_tokens']}; first={too_long[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        sample_batch = samples[start : start + batch_size]
        length_batch = prompt_lengths[start : start + batch_size]
        started = time.monotonic()
        completion_matrix = engine.generate(prompt_batch)
        batch_latency = time.monotonic() - started
        latency_per_prompt = batch_latency / len(prompt_batch)
        for sample, prompt_length, completions in zip(
            sample_batch, length_batch, completion_matrix
        ):
            for sample_index, candidate in enumerate(completions):
                completion = str(candidate["text"])
                grader_error = None
                if grader is not None:
                    verified, grader_error = grade_with_timeout(
                        grader,
                        completion,
                        sample["answer"],
                        grader_timeout_seconds,
                    )
                    if grader_error is not None:
                        print(
                            f"grader failed for {sample['id']}: {grader_error}",
                            file=sys.stderr,
                        )
                else:
                    verified = verify_with_latex(
                        sample["id"],
                        completion,
                        sample["answer"],
                        question=sample["question"],
                    )
                output_tokens = len(candidate["token_ids"])
                finish_reason = str(candidate["finish_reason"])
                rows.append(
                    {
                        "id": sample["id"],
                        "orig_id": sample["id"],
                        "position": sample["position"],
                        "sample_index": sample_index,
                        "question": sample["question"],
                        "ground_truth": sample["answer"],
                        "response": completion.strip(),
                        "token_ids": json.dumps(candidate["token_ids"]),
                        "output_tokens": output_tokens,
                        "finish_reason": finish_reason,
                        "stop_reason": candidate["stop_reason"],
                        "normal_stop": finish_reason == "stop",
                        "truncated": (
                            finish_reason == "length"
                            or output_tokens >= int(decoding["max_new_tokens"])
                        ),
                        "verified": verified,
                        "correct": int(verified is True),
                        "grader_error": grader_error,
                        "prompt_tokens": prompt_length,
                        "latency_s_per_prompt": round(latency_per_prompt, 6),
                    }
                )

    metrics = summarize_rows(rows, int(decoding["n_samples"]))
    metrics.update(
        {
            "truncated_generations": sum(bool(row["truncated"]) for row in rows),
            "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
            "grader_errors": sum(row["grader_error"] is not None for row in rows),
        }
    )
    csv_path = Path(out_csv).resolve()
    json_path = Path(out_json).resolve()
    _write_csv(csv_path, rows)
    summary: dict[str, Any] = {
        **metrics,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset_sha256,
        "expected_dataset_sha256": expected_dataset_sha256,
        "dataset_name": normalize_dataset_name(dataset_name),
        "expected_rows": EXPECTED_ROWS.get(normalize_dataset_name(dataset_name)),
        "model_path": model_path,
        "protocol": protocol,
        "prompt": {"thinking": thinking, "system_prompt": system_prompt},
        "decoding": dict(decoding),
        "runtime": {
            "tensor_parallel_size": tp,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "batch_size": batch_size,
            "grader_timeout_seconds": grader_timeout_seconds,
            "locked_environment": locked_runtime_env,
        },
        "scorer": {
            "name": "external" if grader_path else "verify_with_latex",
            "path": str(Path(grader_path).resolve()) if grader_path else None,
            "sha256": grader_sha256,
            "expected_sha256": expected_grader_sha256,
        },
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "csv_file": str(csv_path),
        "seed": int(decoding["seed"]),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_name(json_path.name + ".tmp")
    with temporary_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    os.replace(temporary_json, json_path)

    n_samples = int(decoding["n_samples"])
    print(f"Dataset: {summary['dataset_name']} ({metrics['n_prompts']} prompts)")
    print(f"Mean@{n_samples}: {metrics[f'mean_at_{n_samples}']:.4f}")
    if n_samples == 1:
        print(f"Pass@1: {metrics['pass_at_1']:.4f}")
    else:
        print(
            "First-sample accuracy (diagnostic; not an independent pass@1): "
            f"{metrics['first_sample_accuracy']:.4f}"
        )
    print(f"Observed pass@{n_samples}: {metrics[f'pass_at_{n_samples}']:.4f}")
    print(f"Details: {csv_path}")
    print(f"Summary: {json_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name")
    parser.add_argument("--protocol", choices=("paper", "opd"), default="paper")
    parser.add_argument("--mode", choices=("greedy", "pass1", "passk"), default="greedy")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float)
    parser.add_argument(
        "--num-samples",
        "--n-samples",
        "--repeat",
        dest="num_samples",
        type=int,
        help="Samples per prompt. Legacy --repeat is retained as an alias.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--max-model-len", "--max_model_len", dest="max_model_len", type=int
    )
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-prompt")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument(
        "--stop-token-id", dest="stop_token_ids", type=int, action="append"
    )
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--skip-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME"))
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
    )
    parser.add_argument(
        "--grader",
        help="Pinned Python file exposing grade_answer_verl or compute_score",
    )
    parser.add_argument(
        "--grader-sha256",
        help="Fail unless the external grader has this SHA256.",
    )
    parser.add_argument(
        "--allow-public-grader",
        action="store_true",
        help="Allow the OPD prompt/decoding preset with the public math-verify scorer.",
    )
    parser.add_argument("--grader-timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--dataset-sha256",
        help="Fail unless the evaluation dataset has this SHA256.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir")
    parser.add_argument("--epoch", help="Legacy output-path component")
    parser.add_argument("--latest-checkpoint", action="store_true")
    parser.add_argument("--allow-row-count-mismatch", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace evaluator artifacts already present in --output-dir.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tp <= 0 or args.batch_size <= 0:
        parser.error("--tp and --batch-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.no_system_prompt and args.system_prompt is not None:
        parser.error("--system-prompt and --no-system-prompt are mutually exclusive")
    if args.protocol == "opd" and args.system_prompt is not None:
        parser.error("the OPD protocol is user-only and rejects --system-prompt")
    if args.protocol == "opd" and not args.thinking:
        parser.error("the OPD protocol requires --thinking")
    if args.protocol == "opd" and not args.grader and not args.allow_public_grader:
        parser.error(
            "the OPD protocol requires --grader; use --allow-public-grader "
            "only for an intentional scorer change"
        )
    if args.protocol == "opd" and args.grader and args.grader_sha256 is None:
        args.grader_sha256 = OPD_GRADER_SHA256
    if args.grader_sha256 and not args.grader:
        parser.error("--grader-sha256 requires --grader")
    if args.grader_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.grader_sha256):
        parser.error("--grader-sha256 must contain 64 hexadecimal characters")
    if args.dataset_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.dataset_sha256
    ):
        parser.error("--dataset-sha256 must contain 64 hexadecimal characters")
    if args.grader_timeout_seconds <= 0:
        parser.error("--grader-timeout-seconds must be positive")
    return args


def _infer_dataset_name(path: str) -> str:
    source = Path(path)
    return normalize_dataset_name(source.parent.name or source.stem)


def _safe_output_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "model"


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    dataset_name = normalize_dataset_name(
        args.dataset_name or _infer_dataset_name(args.dataset)
    )
    decoding = resolve_decoding(args, dataset_name)

    model_path = args.model.rstrip("/")
    if args.latest_checkpoint:
        if not Path(model_path).is_dir():
            raise ValueError("--latest-checkpoint requires a local model directory")
        latest = get_last_checkpoint(model_path)
        if latest is None:
            raise ValueError(f"no checkpoint-* directories under {model_path}")
        model_path = latest

    if args.no_system_prompt or args.protocol == "opd":
        system_prompt = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = SYSTEM_PROMPT

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = _safe_output_component(Path(model_path).name or model_path)
        decoding_name = (
            f"{args.protocol}-{args.mode}-t{decoding['temperature']}-"
            f"n{decoding['n_samples']}-seed{decoding['seed']}"
        )
        output_dir = REPO_ROOT / "results" / model_name
        if args.epoch:
            output_dir /= _safe_output_component(args.epoch)
        output_dir /= dataset_name / decoding_name
    existing_artifacts = (
        output_dir / "eval_detail.csv",
        output_dir / "eval_summary.json",
    )
    if not args.overwrite and any(path.exists() for path in existing_artifacts):
        raise FileExistsError(
            f"evaluation artifacts already exist in {output_dir}; "
            "choose a new directory or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    return evaluate(
        dataset_path=args.dataset,
        model_path=model_path,
        dataset_name=dataset_name,
        tp=args.tp,
        out_csv=str(output_dir / "eval_detail.csv"),
        out_json=str(output_dir / "eval_summary.json"),
        protocol=args.protocol,
        decoding=decoding,
        thinking=args.thinking,
        system_prompt=system_prompt,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_cache=args.hf_cache,
        enforce_eager=args.enforce_eager,
        validate_count=not args.allow_row_count_mismatch,
        expected_dataset_sha256=args.dataset_sha256,
        grader_path=args.grader,
        expected_grader_sha256=args.grader_sha256,
        grader_timeout_seconds=args.grader_timeout_seconds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
