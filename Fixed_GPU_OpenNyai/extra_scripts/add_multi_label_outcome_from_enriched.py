#!/usr/bin/env python3
"""Add multi-label case outcome labels to summary-enriched JSON files.

This script reads the enriched JSONs produced by
`run_opennyai_summarizer_custom.py`, extracts:

- `opennyai_summary.decision`
- all sentence texts whose `rhetorical_role` is `RPC`

It then applies a Mistral/vLLM classification that produces:

  6 binary labels (each 0 or 1):
    1. appeal_allowed        — petition/appeal/application allowed
    2. appeal_dismissed      — petition/appeal/application dismissed/rejected
    3. bail_granted          — bail or custody relief granted
    4. conviction_set_aside  — prior conviction or order quashed/set aside
    5. remanded              — case remanded or sent back for fresh consideration
    6. partial_relief        — only partial relief or order modification granted

  1 final ternary label:
    case_outcome_label  : appellant_won | postponed_or_procedural | appellant_lost
    case_outcome_score  : 1 | 0 | -1

The 6 binary labels are designed to be cross-verifiable:
- appeal_allowed=1      =>  appeal_dismissed must be 0
- appeal_dismissed=1    =>  appeal_allowed must be 0
- bail_granted=1        =>  appeal_allowed must be 1
- conviction_set_aside=1 => appeal_allowed must be 1
- remanded=1            =>  case_outcome_score should be 0

This allows post-hoc CSV-level consistency checks and reduces LLM hallucination
by forcing the model to commit to fine-grained, cross-constrained answers.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import write_json  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_CACHE_DIR = (
    WORKSPACE_ROOT
    / "hf_cache"
    / "hub"
    / f"models--{DEFAULT_MODEL_ID.replace('/', '--')}"
)

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "summary_from_annotations" / "enriched_jsons"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "summary_from_annotations_multi_labelled"

BINARY_LABELS = [
    "appeal_allowed",
    "appeal_dismissed",
    "bail_granted",
    "conviction_set_aside",
    "remanded",
    "partial_relief",
]

LABEL_TO_SCORE = {
    "appellant_won": 1,
    "postponed_or_procedural": 0,
    "appellant_lost": -1,
}
SCORE_TO_LABEL = {v: k for k, v in LABEL_TO_SCORE.items()}

SYSTEM_PROMPT = """You are a legal outcome classifier for Indian court cases.

You will receive the decision summary and the final operative paragraphs (RPC)
from one case. Answer 6 YES/NO questions and give one final outcome label.

BINARY QUESTIONS — answer 1 (yes) or 0 (no):

  appeal_allowed       — Was the petition / appeal / application allowed?
  appeal_dismissed     — Was the petition / appeal / application dismissed or rejected?
  bail_granted         — Was bail (regular, interim, or anticipatory) granted?
  conviction_set_aside — Was a prior conviction or court order quashed / set aside?
  remanded             — Was the case remanded or sent back for fresh disposal?
  partial_relief       — Was only partial relief granted, or was the order modified
                         rather than fully allowed or outright dismissed?

RULES (must hold):
  - appeal_allowed and appeal_dismissed cannot both be 1.
  - If bail_granted = 1, then appeal_allowed must also be 1.
  - If conviction_set_aside = 1, then appeal_allowed must also be 1.
  - If remanded = 1 and neither appeal_allowed nor appeal_dismissed is 1,
    then case_outcome_score should be 0.

FINAL OUTCOME — pick exactly one:
  appellant_won  (score 1)   — appeal allowed, bail granted, conviction set aside, etc.
  postponed_or_procedural (score 0) — remanded, adjourned, notice issued, no clear win/loss.
  appellant_lost (score -1)  — appeal dismissed, bail denied, conviction upheld, etc.

Return JSON only — no extra text:
{
  "appeal_allowed": 1 or 0,
  "appeal_dismissed": 1 or 0,
  "bail_granted": 1 or 0,
  "conviction_set_aside": 1 or 0,
  "remanded": 1 or 0,
  "partial_relief": 1 or 0,
  "case_outcome_label": "appellant_won" | "postponed_or_procedural" | "appellant_lost",
  "case_outcome_score": 1 | 0 | -1,
  "confidence": "high" | "medium" | "low",
  "short_explanation": "one or two short sentences"
}
"""


# ---------------------------------------------------------------------------
# Model-path helpers (same as add_case_outcome_labels_mistral.py)
# ---------------------------------------------------------------------------

def resolve_hf_cache_snapshot(cache_dir: Path) -> Path | None:
    if not cache_dir.is_dir():
        return None
    if (cache_dir / "config.json").is_file():
        return cache_dir
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    ref_path = cache_dir / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = snapshots_dir / revision
            if (snapshot_dir / "config.json").is_file():
                return snapshot_dir
    snapshot_candidates = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_candidates:
        if (snapshot_dir / "config.json").is_file():
            return snapshot_dir
    return None


def resolve_local_model_source(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    snapshot_dir = resolve_hf_cache_snapshot(candidate)
    if snapshot_dir is not None:
        return str(snapshot_dir)
    if model_id == DEFAULT_MODEL_ID:
        snapshot_dir = resolve_hf_cache_snapshot(DEFAULT_LOCAL_MODEL_CACHE_DIR)
        if snapshot_dir is not None:
            return str(snapshot_dir)
    return model_id


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_user_prompt(*, file_id: str, decision_text: str, rpc_texts: list[str]) -> str:
    rpc_block = "\n\n".join(
        f"RPC {i}:\n{text}" for i, text in enumerate(rpc_texts, start=1)
    )
    if not rpc_block:
        rpc_block = "[NONE]"
    if not decision_text:
        decision_text = "[NONE]"
    return (
        f"Case ID: {file_id}\n\n"
        f"Decision Summary:\n{decision_text}\n\n"
        f"RPC Text:\n{rpc_block}\n"
    )


def render_chat_prompt(
    tokenizer: Any,
    *,
    file_id: str,
    decision_text: str,
    rpc_texts: list[str],
) -> str:
    user_prompt = build_user_prompt(
        file_id=file_id,
        decision_text=decision_text,
        rpc_texts=rpc_texts,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


# ---------------------------------------------------------------------------
# Output parsing and normalisation
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def normalize_model_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse, type-coerce, consistency-check, and return the classification."""

    # --- binary labels ---
    binary_values: dict[str, int] = {}
    for label in BINARY_LABELS:
        raw = payload.get(label)
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = 0
        binary_values[label] = 1 if val else 0

    # --- final label / score ---
    label = str(payload.get("case_outcome_label", "")).strip().lower()
    score = payload.get("case_outcome_score")
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None

    if label not in LABEL_TO_SCORE and score in SCORE_TO_LABEL:
        label = SCORE_TO_LABEL[score]
    if score not in SCORE_TO_LABEL and label in LABEL_TO_SCORE:
        score = LABEL_TO_SCORE[label]

    if label not in LABEL_TO_SCORE or score not in SCORE_TO_LABEL:
        raise ValueError(
            f"Unexpected model output: case_outcome_label={label!r}, "
            f"case_outcome_score={score!r}"
        )
    if LABEL_TO_SCORE[label] != score:
        raise ValueError(
            f"Label/score mismatch: label={label!r}, score={score!r}"
        )

    # --- enforce cross-label consistency ---
    # Rule 1: appeal_allowed and appeal_dismissed cannot both be 1
    if binary_values["appeal_allowed"] == 1:
        binary_values["appeal_dismissed"] = 0
    elif binary_values["appeal_dismissed"] == 1:
        binary_values["appeal_allowed"] = 0
    # Rule 2: bail_granted=1 implies appeal_allowed=1
    if binary_values["bail_granted"] == 1:
        binary_values["appeal_allowed"] = 1
        binary_values["appeal_dismissed"] = 0
    # Rule 3: conviction_set_aside=1 implies appeal_allowed=1
    if binary_values["conviction_set_aside"] == 1:
        binary_values["appeal_allowed"] = 1
        binary_values["appeal_dismissed"] = 0
    # Rule 4: remanded=1 with no clear win/loss → override final score to 0
    if (
        binary_values["remanded"] == 1
        and binary_values["appeal_allowed"] == 0
        and binary_values["appeal_dismissed"] == 0
        and score != 0
    ):
        label = "postponed_or_procedural"
        score = 0

    # --- confidence ---
    confidence = str(payload.get("confidence", "medium")).strip().lower() or "medium"
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return {
        **binary_values,
        "case_outcome_label": label,
        "case_outcome_score": score,
        "confidence": confidence,
        "short_explanation": str(payload.get("short_explanation", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class RemoteHFClassifier:
    def __init__(self, *, model_id: str, provider: str, hf_token: str | None, timeout: float):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for --backend remote_hf."
            ) from exc
        self.model_id = model_id
        self.client = InferenceClient(
            provider=provider,
            token=hf_token,
            timeout=timeout,
        )

    def classify(
        self,
        *,
        file_id: str,
        decision_text: str,
        rpc_texts: list[str],
        max_retries: int,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        prompt = build_user_prompt(
            file_id=file_id,
            decision_text=decision_text,
            rpc_texts=rpc_texts,
        )
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat_completion(
                    model=self.model_id,
                    temperature=0.0,
                    max_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                parsed = extract_json_object(content)
                normalized = normalize_model_output(parsed)
                return {**normalized, "raw_model_response": content}
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(
            f"Multi-label classification failed for {file_id}: {last_error}"
        ) from last_error


class LocalVLLMClassifier:
    def __init__(
        self,
        *,
        model_id: str,
        hf_token: str | None,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        dtype: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        enforce_eager: bool,
        max_output_tokens: int,
    ):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "transformers and vllm are required for --backend local_vllm. "
                "Use the `llm` micromamba environment."
            ) from exc

        self.model_id = model_id
        self.model_source = resolve_local_model_source(model_id)
        if self.model_source != model_id:
            print(f"Resolved model_id '{model_id}' to local snapshot '{self.model_source}'.")

        local_files_only = Path(self.model_source).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.llm = LLM(
            model=self.model_source,
            tokenizer=self.model_source,
            tokenizer_mode=tokenizer_mode,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_output_tokens,
        )

    def classify_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            render_chat_prompt(
                self.tokenizer,
                file_id=item["file_id"],
                decision_text=item["decision_text"],
                rpc_texts=item["rpc_texts"],
            )
            for item in items
        ]
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        if len(outputs) != len(items):
            raise RuntimeError(
                f"Expected {len(items)} vLLM outputs, but received {len(outputs)}."
            )
        results = []
        for item, output in zip(items, outputs):
            if not output.outputs:
                raise RuntimeError(f"Empty generation for {item['file_id']}")
            content = output.outputs[0].text
            parsed = extract_json_object(content)
            normalized = normalize_model_output(parsed)
            results.append({**normalized, "raw_model_response": content})
        return results

    def classify(
        self,
        *,
        file_id: str,
        decision_text: str,
        rpc_texts: list[str],
        max_retries: int,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        del max_retries, max_output_tokens
        return self.classify_batch(
            [{"file_id": file_id, "decision_text": decision_text, "rpc_texts": rpc_texts}]
        )[0]


def build_classifier(args: argparse.Namespace) -> Any:
    if args.backend == "remote_hf":
        return RemoteHFClassifier(
            model_id=args.model_id,
            provider=args.provider,
            hf_token=args.hf_token,
            timeout=args.timeout,
        )
    return LocalVLLMClassifier(
        model_id=args.model_id,
        hf_token=args.hf_token,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        max_output_tokens=args.max_output_tokens,
    )


# ---------------------------------------------------------------------------
# Input extraction (same OpenNyAI enriched-JSON format as original)
# ---------------------------------------------------------------------------

def collect_input_texts_from_enriched(payload: dict[str, Any]) -> dict[str, Any]:
    summary_mapping = payload.get("opennyai_summary", {})
    if not isinstance(summary_mapping, dict):
        summary_mapping = {}

    decision_text = str(summary_mapping.get("decision", "")).strip()

    rpc_texts: list[str] = []
    for sentence in payload.get("sentences", []):
        if not isinstance(sentence, dict):
            continue
        role = str(sentence.get("rhetorical_role") or "NONE").strip().upper()
        if role != "RPC":
            continue
        text = str(sentence.get("text", "")).strip()
        if text:
            rpc_texts.append(text)

    return {
        "decision_text": decision_text,
        "rpc_texts": rpc_texts,
    }


def apply_classification_to_payload(
    *,
    combined_payload: dict[str, Any],
    extracted: dict[str, Any],
    classification: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write all labels into the top-level payload dict."""
    # Flat binary labels at top level for easy CSV extraction
    for label in BINARY_LABELS:
        combined_payload[label] = classification[label]

    # Final ternary outcome (mirrors existing field names for compatibility)
    combined_payload["case_outcome_label"] = classification["case_outcome_label"]
    combined_payload["case_outcome_score"] = classification["case_outcome_score"]

    # Full provenance block
    combined_payload["llm_multi_label_outcome"] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "provider": args.provider if args.backend == "remote_hf" else "local_gpu",
        "decision_text": extracted["decision_text"],
        "rpc_texts": extracted["rpc_texts"],
        **{label: classification[label] for label in BINARY_LABELS},
        "case_outcome_label": classification["case_outcome_label"],
        "case_outcome_score": classification["case_outcome_score"],
        "confidence": classification["confidence"],
        "short_explanation": classification["short_explanation"],
        "raw_model_response": classification["raw_model_response"],
    }


# ---------------------------------------------------------------------------
# Batch flush helper
# ---------------------------------------------------------------------------

def flush_local_vllm_batch(
    *,
    pending_batch: list[dict[str, Any]],
    classifier: Any,
    args: argparse.Namespace,
) -> int:
    if not pending_batch:
        return 0

    errors = 0
    batch_inputs = [
        {
            "file_id": item["file_id"],
            "decision_text": item["extracted"]["decision_text"],
            "rpc_texts": item["extracted"]["rpc_texts"],
        }
        for item in pending_batch
    ]

    try:
        classifications = classifier.classify_batch(batch_inputs)
        if len(classifications) != len(pending_batch):
            raise RuntimeError(
                f"Expected {len(pending_batch)} vLLM outputs, got {len(classifications)}."
            )
        for item, classification in zip(pending_batch, classifications):
            item["record"].update(classification)
            item["record"]["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=item["payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(item["output_path"], item["payload"])
        return errors
    except Exception:
        pass  # fall back to one-by-one

    for item in pending_batch:
        try:
            classification = classifier.classify(
                file_id=item["file_id"],
                decision_text=item["extracted"]["decision_text"],
                rpc_texts=item["extracted"]["rpc_texts"],
                max_retries=args.max_retries,
                max_output_tokens=args.max_output_tokens,
            )
            item["record"].update(classification)
            item["record"]["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=item["payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(item["output_path"], item["payload"])
        except Exception as exc:
            item["record"]["status"] = "error"
            item["record"]["error"] = str(exc)
            errors += 1
            write_json(item["output_path"], item["payload"])

    return errors


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=["local_vllm", "remote_hf"], default="local_vllm")
    parser.add_argument("--provider", default="auto")
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-finished files in the output directory.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_output_tokens", type=int, default=400)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=16,
        help="Number of cases to batch into one local vLLM generate call.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tokenizer_mode", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--cuda_visible_devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value for local_vllm runs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    labelled_dir = output_dir / "labelled_jsons"
    manifest_path = output_dir / "case_outcomes_multi_label.jsonl"
    run_summary_path = output_dir / "run_summary.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.hf_token:
        raise EnvironmentError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
        )

    classifier = None
    if not args.dry_run:
        classifier = build_classifier(args)

    paths = sorted(input_dir.glob("*.json"))
    if args.resume:
        paths = [p for p in paths if not (labelled_dir / p.name).exists()]
    if args.max_files is not None:
        paths = paths[: args.max_files]

    skipped = 0
    errors = 0
    pending_local_vllm_batch: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    from tqdm import tqdm

    for path in tqdm(paths, desc="Multi-label case outcome labelling"):
        output_path = labelled_dir / path.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        extracted = collect_input_texts_from_enriched(payload)
        file_id = str(payload.get("file_id") or path.stem).strip() or path.stem

        record: dict[str, Any] = {
            "file_id": file_id,
            "source_json": str(path),
            "labelled_json": str(output_path),
            "decision_text": extracted["decision_text"],
            "rpc_texts": extracted["rpc_texts"],
            # binary label placeholders
            **{label: None for label in BINARY_LABELS},
            # final label placeholders
            "case_outcome_label": None,
            "case_outcome_score": None,
            "status": "pending",
        }
        records.append(record)

        if not extracted["decision_text"] and not extracted["rpc_texts"]:
            record["status"] = "skipped_missing_decision_and_rpc"
            skipped += 1
            write_json(output_path, payload)
            continue

        if args.dry_run:
            record["status"] = "dry_run_only"
            write_json(output_path, payload)
            continue

        payload = copy.deepcopy(payload)

        if args.backend == "local_vllm":
            pending_local_vllm_batch.append(
                {
                    "file_id": file_id,
                    "payload": payload,
                    "extracted": extracted,
                    "record": record,
                    "output_path": output_path,
                }
            )
            if len(pending_local_vllm_batch) >= args.generation_batch_size:
                errors += flush_local_vllm_batch(
                    pending_batch=pending_local_vllm_batch,
                    classifier=classifier,
                    args=args,
                )
                pending_local_vllm_batch = []
            continue

        # remote_hf path
        try:
            classification = classifier.classify(
                file_id=file_id,
                decision_text=extracted["decision_text"],
                rpc_texts=extracted["rpc_texts"],
                max_retries=args.max_retries,
                max_output_tokens=args.max_output_tokens,
            )
            record.update(classification)
            record["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=payload,
                extracted=extracted,
                classification=classification,
                args=args,
            )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            errors += 1

        write_json(output_path, payload)
        if args.backend == "remote_hf" and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if pending_local_vllm_batch:
        errors += flush_local_vllm_batch(
            pending_batch=pending_local_vllm_batch,
            classifier=classifier,
            args=args,
        )

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok_records = [r for r in records if r["status"] == "ok"]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "labelled_dir": str(labelled_dir),
        "backend": args.backend,
        "model_id": args.model_id,
        "provider": args.provider if args.backend == "remote_hf" else "local_gpu",
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "files_attempted_this_run": len(paths),
        "successful_files": len(ok_records),
        "dry_run_files": sum(1 for r in records if r["status"] == "dry_run_only"),
        "skipped_files": skipped,
        "error_files": errors,
        # final label counts
        "label_counts": {
            label: sum(1 for r in ok_records if r.get("case_outcome_label") == label)
            for label in LABEL_TO_SCORE
        },
        # binary label sums (how many cases had each condition)
        "binary_label_sums": {
            label: sum(1 for r in ok_records if r.get(label) == 1)
            for label in BINARY_LABELS
        },
    }
    write_json(run_summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
