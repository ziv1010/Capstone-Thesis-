#!/usr/bin/env python3
"""Add cross-validated LLM-based case outcome labels to OpenNyai combined JSON outputs.

This script reads combined JSON files, extracts only:
- `raw_result.summary.decision`
- all `RPC`-labeled annotation texts

It then classifies the case outcome using a CROSS-VALIDATION approach:
    - Eight binary yes/no questions are posed to the model in a single call.
    - Answers are grouped into three semantic buckets:
        Group A (Win Signals):     Q1-Q3
        Group B (Loss Signals):    Q4-Q6
        Group C (Neutral Signals): Q7-Q8
    - A deterministic aggregation function (`determine_final_outcome`) resolves
      contradictions and produces the final label + confidence score.

Final labels:
    - `appellant_won`          → score  1
    - `postponed_or_procedural` → score  0
    - `appellant_lost`          → score -1
    - `unknown`                 → score  None   (all eight answers = 0)

Default backend: local GPU inference with `vllm`.
Fallback backend: Hugging Face remote inference API.

The original JSON files are not modified. Augmented copies are written to a
separate output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False


DEFAULT_MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_CACHE_DIR = (
    WORKSPACE_ROOT
    / "hf_cache"
    / "hub"
    / f"models--{DEFAULT_MODEL_ID.replace('/', '--')}"
)
DEFAULT_INPUT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/OpenNyai/outputs/current_output/combined"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/OpenNyai/outputs/current_output/combined_mistral24b_case_outcomes_crossval"
)

LABEL_TO_SCORE = {
    "appellant_won": 1,
    "postponed_or_procedural": 0,
    "appellant_lost": -1,
    "unknown": None,
}
SCORE_TO_LABEL = {1: "appellant_won", 0: "postponed_or_procedural", -1: "appellant_lost"}

# ---------------------------------------------------------------------------
# Cross-validation system prompt
# ---------------------------------------------------------------------------
# We ask 8 binary questions in one call. The model must return a JSON object
# with exactly these keys: q1 … q8, each = 1 (YES) or 0 (NO).
SYSTEM_PROMPT = """You are a strict legal outcome classifier.

You will receive the decision summary and final operative paragraphs (RPC) of one case.
You must answer EIGHT yes/no questions about the outcome FROM THE PERSPECTIVE of the
appellant / petitioner / applicant.

⚠️  STRICT RULES:
• Answer ONLY with integer 1 (= YES) or 0 (= NO) for every question.
• Do NOT add explanation, commentary, or any text outside the JSON object.
• Return a SINGLE valid JSON object with exactly these keys: q1 q2 q3 q4 q5 q6 q7 q8.

QUESTIONS:

Group A — Win signals (from the appellant's perspective):
  q1: Did the appellant win the case?
  q2: Was the petition / appeal / application allowed?
  q3: Did the court rule in favour of the petitioner / appellant?

Group B — Loss signals (from the appellant's perspective):
  q4: Did the respondent win the case?
  q5: Was the petition / appeal / application dismissed?
  q6: Did the court rule against the appellant?

Group C — Neutral / ambiguous signals:
  q7: Was the case adjourned, remanded, or deferred without a final decision?
  q8: Is there no clear final judgment in this case?

Return format (ONLY this, nothing else):
{
  "q1": 0,
  "q2": 0,
  "q3": 0,
  "q4": 0,
  "q5": 0,
  "q6": 0,
  "q7": 0,
  "q8": 0
}
"""


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------

def determine_final_outcome(answers: dict[str, int]) -> dict[str, Any]:
    """Aggregate 8 binary question answers into a single outcome label.

    Priority rules (highest → lowest):
        1. Neutral/ambiguous signals trump everything.
        2. Pure win / pure loss with no contradiction.
        3. Contradiction resolution by majority signal; penalise confidence.
        4. All-zero fallback → UNKNOWN.

    Args:
        answers: dict with keys 'q1' … 'q8', values 0 or 1.

    Returns:
        dict with keys:
            'label'      – 'WIN' | 'LOSS' | 'NEUTRAL' | 'UNKNOWN'
            'value'      – 1 | -1 | 0 | None
            'confidence' – float in [0, 1]
    """
    win_score     = answers["q1"] + answers["q2"] + answers["q3"]
    loss_score    = answers["q4"] + answers["q5"] + answers["q6"]
    neutral_score = answers["q7"] + answers["q8"]

    # Rule 1: Adjournment / remand / no-final-judgment overrides everything.
    if neutral_score > 0:
        return {
            "label": "NEUTRAL",
            "value": 0,
            "confidence": round(neutral_score / 2.0, 4),
        }

    # Rule 2 & 3: Compare win vs loss signals.
    if win_score > 0 and loss_score > 0:
        # Tied contradiction → neutral with zero confidence.
        if win_score == loss_score:
            return {"label": "NEUTRAL", "value": 0, "confidence": 0.0}
        if win_score > loss_score:
            return {
                "label": "WIN",
                "value": 1,
                "confidence": round((win_score - loss_score) / 3.0, 4),
            }
        return {
            "label": "LOSS",
            "value": -1,
            "confidence": round((loss_score - win_score) / 3.0, 4),
        }

    # Rule 4: Clean, unambiguous classifications.
    if win_score > 0:
        return {"label": "WIN", "value": 1, "confidence": round(win_score / 3.0, 4)}
    if loss_score > 0:
        return {"label": "LOSS", "value": -1, "confidence": round(loss_score / 3.0, 4)}

    # Fallback: model answered 0 to everything.
    return {"label": "UNKNOWN", "value": None, "confidence": 0.0}


def label_value_to_standard(label: str, value: int | None) -> tuple[str, int | None]:
    """Convert internal WIN/LOSS/NEUTRAL/UNKNOWN labels to the project-standard schema."""
    mapping = {
        "WIN":     ("appellant_won",             1),
        "LOSS":    ("appellant_lost",            -1),
        "NEUTRAL": ("postponed_or_procedural",    0),
        "UNKNOWN": ("unknown",                   None),
    }
    return mapping.get(label, ("unknown", None))


# ---------------------------------------------------------------------------
# Model-cache helpers  (identical to the original script)
# ---------------------------------------------------------------------------

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
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_candidates:
        if (snapshot_dir / "config.json").is_file():
            return snapshot_dir

    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir",  default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_id",   default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=["local_vllm", "remote_hf"], default="local_vllm")
    parser.add_argument("--provider", default="auto")
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output_dir by skipping already-finished files.",
    )
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--sleep_seconds",          type=float, default=0.0)
    parser.add_argument("--timeout",                type=float, default=120.0)
    parser.add_argument("--max_retries",            type=int,   default=3)
    parser.add_argument("--max_output_tokens",      type=int,   default=200)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=16,
        help="Number of cases to batch into one local vLLM generate call.",
    )
    parser.add_argument("--tensor_parallel_size",   type=int,   default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len",          type=int,   default=4096)
    parser.add_argument("--dtype",           default="auto")
    parser.add_argument("--tokenizer_mode",  default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager",     action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
        end   = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


# ---------------------------------------------------------------------------
# Parse and validate cross-val model output
# ---------------------------------------------------------------------------

REQUIRED_QUESTION_KEYS = {"q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"}


def parse_binary_answers(payload: dict[str, Any]) -> dict[str, int]:
    """Extract and validate q1…q8 integer answers from the model response.

    Raises ValueError on missing keys or non-binary values.
    """
    answers: dict[str, int] = {}
    missing = REQUIRED_QUESTION_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Model response missing keys: {sorted(missing)}")

    for key in sorted(REQUIRED_QUESTION_KEYS):
        raw = payload[key]
        try:
            val = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer answer for {key}: {raw!r}")
        if val not in (0, 1):
            raise ValueError(f"Non-binary answer for {key}: {val!r}")
        answers[key] = val

    return answers


def normalize_crossval_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse raw model output into cross-val answers + final aggregated outcome."""
    answers = parse_binary_answers(payload)
    outcome = determine_final_outcome(answers)

    standard_label, standard_score = label_value_to_standard(
        outcome["label"], outcome["value"]
    )

    return {
        # Per-question binary answers (preserved verbatim for downstream use)
        "crossval_answers": answers,
        # Intermediate aggregation
        "win_score":     answers["q1"] + answers["q2"] + answers["q3"],
        "loss_score":    answers["q4"] + answers["q5"] + answers["q6"],
        "neutral_score": answers["q7"] + answers["q8"],
        # Final outcome (matches schema used by the original script)
        "case_outcome_label": standard_label,
        "case_outcome_score": standard_score,
        "crossval_confidence": outcome["confidence"],
        "crossval_label":      outcome["label"],   # WIN / LOSS / NEUTRAL / UNKNOWN
    }


# ---------------------------------------------------------------------------
# Manifest persistence  (identical to the original script)
# ---------------------------------------------------------------------------

def load_existing_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    records_by_source: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return records_by_source

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for raw_line in manifest_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_json = str(record.get("source_json", "")).strip()
            if not source_json:
                continue
            records_by_source[source_json] = record
    return records_by_source


def rewrite_manifest(manifest_path: Path, records_by_source: dict[str, dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records_by_source.values():
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_completed_sources_for_resume(
    *,
    input_dir: Path,
    augmented_dir: Path,
    records_by_source: dict[str, dict[str, Any]],
) -> set[str]:
    completed_sources: set[str] = set()

    for source_json, record in records_by_source.items():
        status = str(record.get("status", "")).strip()
        if status == "ok":
            if (augmented_dir / Path(source_json).name).exists():
                completed_sources.add(source_json)
        elif status == "skipped_missing_decision_and_rpc":
            completed_sources.add(source_json)

    for output_path in augmented_dir.glob("*.json"):
        source_path = input_dir / output_path.name
        if source_path.exists():
            completed_sources.add(str(source_path))

    return completed_sources


# ---------------------------------------------------------------------------
# Input text extraction
# ---------------------------------------------------------------------------

def collect_input_texts(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract decision text and RPC sentences from an enriched JSON document.

    Supports two formats:
      1. Enriched format (add_case_outcome_labels_from_enriched pipeline):
             payload["opennyai_summary"]["decision"]
             payload["sentences"][*]["rhetorical_role"] == "RPC"
      2. Combined format (add_case_outcome_labels_mistral pipeline):
             payload["raw_result"]["summary"]["decision"]
             payload["raw_result"]["annotations"][*]["labels"] contains "RPC"
    """
    # ── Format 1: enriched (opennyai_summary + sentences) ──
    summary_mapping = payload.get("opennyai_summary")
    if isinstance(summary_mapping, dict) and summary_mapping:
        decision_text = str(summary_mapping.get("decision", "")).strip()
        rpc_texts: list[str] = []
        for sentence in payload.get("sentences", []):
            if not isinstance(sentence, dict):
                continue
            role = str(sentence.get("rhetorical_role") or "").strip().upper()
            if role != "RPC":
                continue
            text = str(sentence.get("text", "")).strip()
            if text:
                rpc_texts.append(text)
        return {"decision_text": decision_text, "rpc_texts": rpc_texts}

    # ── Format 2: combined (raw_result) ──
    raw_result    = payload.get("raw_result", {})
    decision_text = str(raw_result.get("summary", {}).get("decision", "")).strip()
    rpc_texts = []
    for annotation in raw_result.get("annotations", []):
        labels = annotation.get("labels", [])
        if "RPC" in labels:
            text = str(annotation.get("text", "")).strip()
            if text:
                rpc_texts.append(text)

    return {"decision_text": decision_text, "rpc_texts": rpc_texts}


def build_user_prompt(*, file_id: str, decision_text: str, rpc_texts: list[str]) -> str:
    rpc_block = "\n\n".join(
        f"RPC {index}:\n{text}" for index, text in enumerate(rpc_texts, start=1)
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


def render_chat_prompt(tokenizer: Any, *, file_id: str, decision_text: str, rpc_texts: list[str]) -> str:
    user_prompt = build_user_prompt(
        file_id=file_id,
        decision_text=decision_text,
        rpc_texts=rpc_texts,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class RemoteHFClassifier:
    def __init__(self, *, model_id: str, provider: str, hf_token: str | None, timeout: float):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required for --backend remote_hf."
            ) from exc

        self.model_id = model_id
        self.client   = InferenceClient(
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
                        {"role": "user",   "content": prompt},
                    ],
                )
                content  = response.choices[0].message.content
                parsed   = extract_json_object(content)
                normalized = normalize_crossval_output(parsed)
                return {**normalized, "raw_model_response": content}
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(f"Classification failed for {file_id}: {last_error}") from last_error


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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers and vllm are required for --backend local_vllm. "
                "Use the `llm` micromamba environment."
            ) from exc

        self.model_id     = model_id
        self.model_source = resolve_local_model_source(model_id)
        if self.model_source != model_id:
            print(f"Resolved model_id '{model_id}' to local snapshot '{self.model_source}'.")

        local_files_only = Path(self.model_source).exists()
        self.tokenizer   = AutoTokenizer.from_pretrained(
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
            content    = output.outputs[0].text
            parsed     = extract_json_object(content)
            normalized = normalize_crossval_output(parsed)
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
# Persistence helpers
# ---------------------------------------------------------------------------

def persist_record(
    *,
    path: Path,
    combined_payload: dict[str, Any],
    record: dict[str, Any],
    augmented_dir: Path,
    manifest_file: Any,
    records_by_source: dict[str, dict[str, Any]],
) -> None:
    if record["status"] == "ok":
        write_json(augmented_dir / path.name, combined_payload)

    manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_file.flush()
    records_by_source[record["source_json"]] = record


def apply_classification_to_payload(
    *,
    combined_payload: dict[str, Any],
    extracted: dict[str, Any],
    classification: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write cross-val classification results into the combined payload dict (in place)."""
    combined_payload["case_outcome_label"] = classification["case_outcome_label"]
    combined_payload["case_outcome_score"] = classification["case_outcome_score"]
    combined_payload["llm_case_outcome"] = {
        "backend":   args.backend,
        "model_id":  args.model_id,
        "provider":  args.provider if args.backend == "remote_hf" else "local_gpu",
        "method":    "cross_validation_8q",
        # Input texts fed to the model
        "decision_text": extracted["decision_text"],
        "rpc_texts":     extracted["rpc_texts"],
        # Cross-val detail
        "crossval_answers":    classification["crossval_answers"],
        "win_score":           classification["win_score"],
        "loss_score":          classification["loss_score"],
        "neutral_score":       classification["neutral_score"],
        "crossval_label":      classification["crossval_label"],      # WIN/LOSS/NEUTRAL/UNKNOWN
        "crossval_confidence": classification["crossval_confidence"],
        # Raw model output for debugging
        "raw_model_response": classification["raw_model_response"],
    }


# ---------------------------------------------------------------------------
# Batch flushing  (local vLLM)
# ---------------------------------------------------------------------------

def flush_local_vllm_batch(
    *,
    pending_batch: list[dict[str, Any]],
    classifier: LocalVLLMClassifier,
    args: argparse.Namespace,
    augmented_dir: Path,
    manifest_file: Any,
    records_by_source: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    if not pending_batch:
        return 0, 0

    errors  = 0
    skipped = 0
    batch_inputs = [
        {
            "file_id":       item["record"]["file_id"],
            "decision_text": item["extracted"]["decision_text"],
            "rpc_texts":     item["extracted"]["rpc_texts"],
        }
        for item in pending_batch
    ]

    try:
        classifications = classifier.classify_batch(batch_inputs)
        for item, classification in zip(pending_batch, classifications):
            record = item["record"]
            record.update(classification)
            record["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=item["combined_payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            persist_record(
                path=item["path"],
                combined_payload=item["combined_payload"],
                record=record,
                augmented_dir=augmented_dir,
                manifest_file=manifest_file,
                records_by_source=records_by_source,
            )
        return skipped, errors
    except Exception:
        # Fall back to one-by-one so a single bad prompt doesn't drop the whole batch.
        for item in pending_batch:
            record = item["record"]
            try:
                classification = classifier.classify(
                    file_id=record["file_id"],
                    decision_text=item["extracted"]["decision_text"],
                    rpc_texts=item["extracted"]["rpc_texts"],
                    max_retries=args.max_retries,
                    max_output_tokens=args.max_output_tokens,
                )
                record.update(classification)
                record["status"] = "ok"
                apply_classification_to_payload(
                    combined_payload=item["combined_payload"],
                    extracted=item["extracted"],
                    classification=classification,
                    args=args,
                )
            except Exception as exc:  # pragma: no cover
                record["status"] = "error"
                record["error"]  = str(exc)
                errors += 1

            persist_record(
                path=item["path"],
                combined_payload=item["combined_payload"],
                record=record,
                augmented_dir=augmented_dir,
                manifest_file=manifest_file,
                records_by_source=records_by_source,
            )
        return skipped, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    args = parse_args()

    input_dir        = Path(args.input_dir).resolve()
    output_dir       = Path(args.output_dir).resolve()
    augmented_dir    = output_dir / "augmented_jsons"
    manifest_path    = output_dir / "case_outcomes_crossval.jsonl"
    run_summary_path = output_dir / "run_summary.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if not args.dry_run and args.backend == "remote_hf" and not args.hf_token:
        raise EnvironmentError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
        )

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    records_by_source = load_existing_records(manifest_path) if args.resume else {}
    completed_sources = (
        collect_completed_sources_for_resume(
            input_dir=input_dir,
            augmented_dir=augmented_dir,
            records_by_source=records_by_source,
        )
        if args.resume
        else set()
    )

    classifier = None
    if not args.dry_run:
        classifier = build_classifier(args)

    paths = sorted(input_dir.glob("*.json"))
    if completed_sources:
        paths = [path for path in paths if str(path) not in completed_sources]
    if args.max_files is not None:
        paths = paths[: args.max_files]

    skipped = 0
    errors  = 0
    pending_local_vllm_batch: list[dict[str, Any]] = []

    manifest_mode = "a" if args.resume else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_file:
        from tqdm import tqdm

        for path in tqdm(paths, desc="Labeling case outcomes (cross-val)"):
            combined_payload = json.loads(path.read_text(encoding="utf-8"))
            extracted        = collect_input_texts(combined_payload)

            file_id = str(
                combined_payload.get("file_id")
                or combined_payload.get("internal_file_id")
                or path.stem
            )

            record = {
                "file_id":             file_id,
                "source_json":         str(path),
                "decision_text":       extracted["decision_text"],
                "rpc_texts":           extracted["rpc_texts"],
                "case_outcome_label":  None,
                "case_outcome_score":  None,
                "crossval_answers":    None,
                "win_score":           None,
                "loss_score":          None,
                "neutral_score":       None,
                "crossval_label":      None,
                "crossval_confidence": None,
                "status":              "pending",
            }

            if not extracted["decision_text"] and not extracted["rpc_texts"]:
                record["status"] = "skipped_missing_decision_and_rpc"
                skipped += 1
                persist_record(
                    path=path,
                    combined_payload=combined_payload,
                    record=record,
                    augmented_dir=augmented_dir,
                    manifest_file=manifest_file,
                    records_by_source=records_by_source,
                )

            elif args.dry_run:
                record["status"] = "dry_run_only"
                persist_record(
                    path=path,
                    combined_payload=combined_payload,
                    record=record,
                    augmented_dir=augmented_dir,
                    manifest_file=manifest_file,
                    records_by_source=records_by_source,
                )

            elif args.backend == "local_vllm":
                pending_local_vllm_batch.append(
                    {
                        "path":             path,
                        "combined_payload": combined_payload,
                        "extracted":        extracted,
                        "record":           record,
                    }
                )
                if len(pending_local_vllm_batch) >= args.generation_batch_size:
                    batch_skipped, batch_errors = flush_local_vllm_batch(
                        pending_batch=pending_local_vllm_batch,
                        classifier=classifier,
                        args=args,
                        augmented_dir=augmented_dir,
                        manifest_file=manifest_file,
                        records_by_source=records_by_source,
                    )
                    skipped += batch_skipped
                    errors  += batch_errors
                    pending_local_vllm_batch = []

            else:
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
                        combined_payload=combined_payload,
                        extracted=extracted,
                        classification=classification,
                        args=args,
                    )
                except Exception as exc:  # pragma: no cover
                    record["status"] = "error"
                    record["error"]  = str(exc)
                    errors += 1

                persist_record(
                    path=path,
                    combined_payload=combined_payload,
                    record=record,
                    augmented_dir=augmented_dir,
                    manifest_file=manifest_file,
                    records_by_source=records_by_source,
                )

            if (
                args.backend == "remote_hf"
                and not args.dry_run
                and args.sleep_seconds > 0
            ):
                time.sleep(args.sleep_seconds)

        # Flush any remaining items in the local batch.
        if pending_local_vllm_batch:
            batch_skipped, batch_errors = flush_local_vllm_batch(
                pending_batch=pending_local_vllm_batch,
                classifier=classifier,
                args=args,
                augmented_dir=augmented_dir,
                manifest_file=manifest_file,
                records_by_source=records_by_source,
            )
            skipped += batch_skipped
            errors  += batch_errors

    rewrite_manifest(manifest_path, records_by_source)
    results = list(records_by_source.values())

    # Count how often each internal cross-val label fires.
    crossval_label_counts = {}
    for item in results:
        lbl = item.get("crossval_label")
        if lbl is not None:
            crossval_label_counts[lbl] = crossval_label_counts.get(lbl, 0) + 1

    summary = {
        "input_dir":                   str(input_dir),
        "output_dir":                  str(output_dir),
        "backend":                     args.backend,
        "model_id":                    args.model_id,
        "provider":                    args.provider if args.backend == "remote_hf" else "local_gpu",
        "method":                      "cross_validation_8q",
        "resume_enabled":              args.resume,
        "resumed_completed_files":     len(completed_sources),
        "files_attempted_this_run":    len(paths),
        "processed_files":             len(results),
        "successful_files":            sum(1 for item in results if item["status"] == "ok"),
        "dry_run_files":               sum(1 for item in results if item["status"] == "dry_run_only"),
        "skipped_files":               skipped,
        "error_files":                 errors,
        "label_counts": {
            label: sum(1 for item in results if item.get("case_outcome_label") == label)
            for label in LABEL_TO_SCORE
        },
        "crossval_label_counts":       crossval_label_counts,
    }
    write_json(run_summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
