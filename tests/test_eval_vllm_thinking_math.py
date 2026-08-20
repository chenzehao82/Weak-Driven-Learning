import importlib.util
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ensemble" / "eval_vllm_thinking_math.py"
SPEC = importlib.util.spec_from_file_location("wdl_public_eval", MODULE_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL)


class DatasetLoadingTest(unittest.TestCase):
    def write_json(self, payload):
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with temp:
            json.dump(payload, temp)
        self.addCleanup(Path(temp.name).unlink, missing_ok=True)
        return temp.name

    def test_loader_preserves_zero_answer_and_aliases(self):
        path = self.write_json([{"original_text": "q", "ans": 0}])
        rows = EVAL.load_general_dataset(path)
        self.assertEqual(rows[0]["question"], "q")
        self.assertEqual(rows[0]["answer"], "0")

    def test_loader_accepts_opd_prompt_and_reward_schema(self):
        path = self.write_json(
            [
                {
                    "question": "this alias must not replace the OPD prompt",
                    "prompt": [{"role": "user", "content": "2+2?"}],
                    "reward_model": {"ground_truth": "4"},
                }
            ]
        )
        rows = EVAL.load_general_dataset(path, protocol="opd")
        self.assertEqual(rows[0]["question"], "2+2?")
        self.assertEqual(rows[0]["answer"], "4")

    def test_opd_loader_rejects_source_system_message(self):
        path = self.write_json(
            [
                {
                    "prompt": [
                        {"role": "system", "content": "unexpected"},
                        {"role": "user", "content": "2+2?"},
                    ],
                    "reward_model": {"ground_truth": "4"},
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "OPD prompt must contain one user"):
            EVAL.load_general_dataset(path, protocol="opd")

    def test_loader_rejects_missing_fields_instead_of_shrinking_denominator(self):
        path = self.write_json([{"question": "missing answer"}])
        with self.assertRaisesRegex(ValueError, "lacks answer"):
            EVAL.load_general_dataset(path)

    def test_known_dataset_count_is_enforced(self):
        path = self.write_json([{"question": "q", "answer": "1"}])
        with self.assertRaisesRegex(ValueError, "requires 30 rows"):
            EVAL.load_general_dataset(
                path, dataset_name="aime2025", validate_count=True
            )

    def test_bundled_aime2025_is_the_pinned_30_problem_set(self):
        path = REPO_ROOT / "dataprocess" / "test_dataset" / "aime2025" / "test.json"
        rows = EVAL.load_general_dataset(
            path, dataset_name="aime2025", validate_count=True
        )
        self.assertEqual(len(rows), 30)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "de1b2907208f7e7302825a16af356e5f3782401e9c51150a46d83240e4f3db97",
        )


class PromptAndScoringTest(unittest.TestCase):
    def test_chat_template_falls_back_for_base_tokenizer(self):
        class Tokenizer:
            def apply_chat_template(self, **kwargs):
                raise ValueError("no template")

        prompt = EVAL.apply_chat_template(
            Tokenizer(), "question", thinking=True, system_prompt="system"
        )
        self.assertEqual(prompt, "system: system\n\nuser: question\n\nassistant:")

    def test_opd_chat_template_is_fail_closed(self):
        class Tokenizer:
            def apply_chat_template(self, **kwargs):
                raise ValueError("no template")

        with self.assertRaisesRegex(RuntimeError, "locked thinking chat template"):
            EVAL.apply_chat_template(
                Tokenizer(),
                "question",
                thinking=True,
                system_prompt=None,
                strict=True,
            )

    def test_multiple_choice_extraction_ignores_unanchored_reasoning_letters(self):
        self.assertEqual(EVAL._extract_choice("work... \\boxed{C}"), "C")
        self.assertEqual(EVAL._extract_choice("The final answer is (D)."), "D")
        self.assertIsNone(EVAL._extract_choice("Compare A and B before deciding."))


class ProtocolTest(unittest.TestCase):
    def args(self, *extra):
        return EVAL.parse_args(
            ["--dataset", "aime2025/test.json", "--model", "model", *extra]
        )

    def test_paper_default_is_greedy_seed42(self):
        args = self.args()
        decoding = EVAL.resolve_decoding(args, "aime2025")
        self.assertEqual(args.protocol, "paper")
        self.assertTrue(args.thinking)
        self.assertEqual(decoding["temperature"], 0.0)
        self.assertEqual(decoding["top_p"], 1.0)
        self.assertEqual(decoding["n_samples"], 1)
        self.assertEqual(decoding["seed"], 42)

    def test_paper_and_recent_opd_length_profiles(self):
        self.assertEqual(
            EVAL.dataset_limits("math500", "paper"),
            {"max_input_tokens": 4096, "max_new_tokens": 4096, "max_model_len": 8192},
        )
        self.assertEqual(
            EVAL.dataset_limits("aime2025", "paper"),
            {
                "max_input_tokens": 4096,
                "max_new_tokens": 8192,
                "max_model_len": 12288,
            },
        )
        self.assertEqual(
            EVAL.dataset_limits("math500", "opd"),
            {"max_input_tokens": 1024, "max_new_tokens": 4096, "max_model_len": 5120},
        )
        for dataset in ("aime2025", "amc23"):
            self.assertEqual(
                EVAL.dataset_limits(dataset, "opd"),
                {
                    "max_input_tokens": 2048,
                    "max_new_tokens": 8192,
                    "max_model_len": 10240,
                },
            )

    def test_pass8_is_stochastic_batch_n(self):
        args = self.args(
            "--protocol",
            "opd",
            "--grader",
            "/tmp/ttrl_math/__init__.py",
            "--mode",
            "passk",
            "--temperature",
            "0.5",
            "--num-samples",
            "8",
        )
        decoding = EVAL.resolve_decoding(args, "aime2025")
        self.assertEqual(decoding["n_samples"], 8)
        self.assertEqual(decoding["stop_token_ids"], [151643, 151645])
        self.assertFalse(decoding["skip_special_tokens"])
        self.assertTrue(decoding["trust_remote_code"])
        self.assertEqual(args.grader_sha256, EVAL.OPD_GRADER_SHA256)

    def test_opd_requires_explicit_scorer_choice(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--protocol", "opd", "--mode", "passk")

    def test_greedy_repeat_is_rejected(self):
        args = self.args("--mode", "greedy", "--repeat", "8")
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            EVAL.resolve_decoding(args, "aime2025")


class AggregationTest(unittest.TestCase):
    def test_mean8_and_pass8_are_not_conflated(self):
        rows = []
        for position in range(2):
            for sample_index in range(8):
                rows.append(
                    {
                        "position": position,
                        "sample_index": sample_index,
                        "correct": int(position == 0 and sample_index == 7),
                    }
                )
        summary = EVAL.summarize_rows(rows, n_samples=8)
        self.assertEqual(summary["mean_at_8"], 1 / 16)
        self.assertEqual(summary["pass_at_8"], 1 / 2)
        self.assertEqual(summary["first_sample_accuracy"], 0)
        self.assertIsNone(summary["pass_at_1"])


class VLLMContractTest(unittest.TestCase):
    def test_backend_uses_sampling_n_instead_of_repeating_prompts(self):
        calls = {}

        class Candidate:
            def __init__(self, text, index):
                self.text = text
                self.token_ids = [index, index + 1]
                self.finish_reason = "stop"
                self.stop_reason = 151645

        class Output:
            def __init__(self, n):
                self.outputs = [Candidate(str(index), index) for index in range(n)]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                calls["sampling"] = kwargs
                self.n = kwargs["n"]

        class FakeLLM:
            def __init__(self, **kwargs):
                calls["llm"] = kwargs

            def generate(self, prompts, params):
                calls["prompts"] = list(prompts)
                return [Output(params.n) for _ in prompts]

        old_vllm = sys.modules.get("vllm")
        sys.modules["vllm"] = types.SimpleNamespace(
            LLM=FakeLLM, SamplingParams=FakeSamplingParams
        )
        self.addCleanup(
            lambda: sys.modules.__setitem__("vllm", old_vllm)
            if old_vllm is not None
            else sys.modules.pop("vllm", None)
        )
        backend = EVAL.VLLMBackend(
            "model",
            tensor_parallel_size=1,
            max_model_len=10240,
            max_new_tokens=8192,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            seed=42,
            temperature=0.5,
            top_p=1.0,
            n_samples=8,
            stop_token_ids=[151643, 151645],
            top_k=-1,
            repetition_penalty=1.0,
            skip_special_tokens=False,
            hf_cache=None,
            enforce_eager=False,
            trust_remote_code=True,
        )
        outputs = backend.generate(["one prompt"])
        self.assertEqual(calls["prompts"], ["one prompt"])
        self.assertEqual(calls["sampling"]["n"], 8)
        self.assertEqual(calls["sampling"]["max_tokens"], 8192)
        self.assertEqual(calls["sampling"]["top_k"], -1)
        self.assertEqual(calls["sampling"]["repetition_penalty"], 1.0)
        self.assertFalse(calls["sampling"]["skip_special_tokens"])
        self.assertTrue(calls["llm"]["trust_remote_code"])
        self.assertEqual(len(outputs[0]), 8)
        self.assertEqual(outputs[0][0]["token_ids"], [0, 1])
        self.assertEqual(outputs[0][0]["finish_reason"], "stop")
        self.assertEqual(outputs[0][0]["stop_reason"], 151645)


if __name__ == "__main__":
    unittest.main()
