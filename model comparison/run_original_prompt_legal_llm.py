#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from run_inlegalllama import (
    compact_text,
    existing_case_ids,
    load_case,
    load_model,
    load_tokenizer,
    optional_subfolder,
    output_path,
    resolve_device,
    resolve_paths,
)


INLEGALLLAMA_SFT_PREDICTION_ONLY = "INLegalLlama/SFT/Prediction_Only/pred_only_sft_llama2"
DEFAULT_SECTIONS = (
    "preamble",
    "facts",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
)
SECTION_TITLES = {
    "preamble": "Preamble",
    "facts": "Facts",
    "arguments": "Arguments",
    "petitioner_arguments": "Petitioner or appellant arguments",
    "respondent_arguments": "Respondent arguments",
    "other_lawyer_arguments": "Other legal reasoning and relied authorities",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run legal LLM checkpoints using their published-style 0/1 prompt format. "
            "Predictions are mapped back to this project's -1/1 label space for metrics."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/original_prompt_run"))
    parser.add_argument("--model-name", default="L-NLProc/InLegalLlama")
    parser.add_argument("--model-subfolder", default=INLEGALLLAMA_SFT_PREDICTION_ONLY)
    parser.add_argument("--adapter-mode", choices=["auto", "peft", "full"], default="peft")
    parser.add_argument(
        "--prompt-profile",
        choices=["inlegalllama_case_proceeding", "factlegal_facts"],
        default="inlegalllama_case_proceeding",
    )
    parser.add_argument("--case-glob", default="*.json")
    parser.add_argument("--sections", nargs="+", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--max-section-chars", type=int, default=12000)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    return parser.parse_args()


def normalize_project_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().strip("\"'` .,;:")
    if text in {"1", "+1", "accepted", "acceptance", "allowed", "won", "win"}:
        return "1"
    if text in {"-1", "0", "rejected", "rejection", "dismissed", "lost", "lose"}:
        return "-1"
    if re.fullmatch(r"\+?1(?:\.0)?", text):
        return "1"
    if re.fullmatch(r"-1(?:\.0)?|0(?:\.0)?", text):
        return "-1"
    return None


def prompt_label_to_project(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip().lower().strip("\"'` .,;:")
    accepted = {
        "1",
        "+1",
        "accepted",
        "acceptance",
        "accept",
        "allowed",
        "allow",
        "won",
        "win",
        "positive",
    }
    rejected = {
        "0",
        "-1",
        "rejected",
        "rejection",
        "reject",
        "dismissed",
        "dismiss",
        "lost",
        "lose",
        "negative",
    }
    if text in accepted or re.fullmatch(r"\+?1(?:\.0)?", text):
        return "1", "1"
    if text in rejected or re.fullmatch(r"-1(?:\.0)?|0(?:\.0)?", text):
        return "0", "-1"
    return None, None


def section_text(case: dict[str, Any], sections: list[str], max_section_chars: int) -> str:
    texts = case.get("texts") or {}
    blocks = []
    for key in sections:
        text = compact_text(texts.get(key, ""), max_section_chars)
        if text:
            blocks.append(f"{SECTION_TITLES.get(key, key)}:\n{text}")
    return "\n\n".join(blocks) if blocks else "No usable text sections were found."


def build_prompt(case: dict[str, Any], args: argparse.Namespace) -> str:
    texts = case.get("texts") or {}
    if args.prompt_profile == "factlegal_facts":
        facts = compact_text(texts.get("facts", ""), args.max_section_chars)
        case_text = facts or "No usable facts section was found."
        instruction = (
            "Given the facts of the case, just predict the outcome as '1' for acceptance "
            "or '0' for rejection."
        )
    else:
        case_text = section_text(case, args.sections, args.max_section_chars)
        instruction = (
            "Given the case proceeding, just predict the outcome as '1' for acceptance "
            "or '0' for rejection."
        )

    return f"""### Instructions: {instruction}
### Input: <{case_text}>
### Response:"""


def parse_model_output(text: str) -> tuple[str | None, str | None, float | None]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            raw_label, project_label = prompt_label_to_project(
                payload.get("predicted_label")
                or payload.get("case_outcome_score")
                or payload.get("label")
                or payload.get("outcome")
            )
            confidence = payload.get("confidence")
            try:
                confidence_value = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_value = None
            if project_label is not None:
                return raw_label, project_label, confidence_value
        except json.JSONDecodeError:
            pass

    keyed = re.search(
        r"(?:predicted_label|case_outcome_score|label|outcome)\s*[:=]\s*[\"']?(-?1|0)[\"']?",
        text,
        flags=re.IGNORECASE,
    )
    if keyed:
        raw_label, project_label = prompt_label_to_project(keyed.group(1))
        return raw_label, project_label, None

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    leading = re.search(r"^\s*[\"']?(-?1|0)(?:\b|[\"'.:,;])", first_line)
    if leading:
        raw_label, project_label = prompt_label_to_project(leading.group(1))
        return raw_label, project_label, None

    word_label = re.search(
        r"\b(accepted|acceptance|allowed|won|rejected|rejection|dismissed|lost)\b",
        first_line,
        flags=re.IGNORECASE,
    )
    if word_label:
        raw_label, project_label = prompt_label_to_project(word_label.group(1))
        return raw_label, project_label, None

    return None, None, None


def tokenize_prompt(tokenizer: Any, prompt: str, max_input_tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int, int]:
    encoded = tokenize_prompt(tokenizer, prompt, args.max_input_tokens, device)
    input_len = int(encoded["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    new_tokens = output[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return decoded, input_len, int(new_tokens.shape[-1])


def tokenize_prompt_batch(
    tokenizer: Any,
    prompts: list[str],
    max_input_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[tuple[str, int, int]]:
    encoded = tokenize_prompt_batch(tokenizer, prompts, args.max_input_tokens, device)
    input_width = int(encoded["input_ids"].shape[-1])
    prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)

    results: list[tuple[str, int, int]] = []
    for row_index in range(len(prompts)):
        new_tokens = output[row_index][input_width:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            (
                decoded,
                int(prompt_token_counts[row_index]),
                int(new_tokens.shape[-1]),
            )
        )
    return results


def write_record(out: Any, record: dict[str, Any], started: float) -> None:
    record["elapsed_seconds"] = round(time.time() - started, 4)
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()


def process_batch(
    batch: list[tuple[Path, dict[str, Any], dict[str, Any], float]],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    out: Any,
) -> None:
    prompts: list[str] = []
    prompt_items: list[tuple[dict[str, Any], float]] = []
    for _path, case, record, started in batch:
        try:
            prompts.append(build_prompt(case, args))
            prompt_items.append((record, started))
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "raw_predicted_label": None,
                    "predicted_label": None,
                    "confidence": None,
                    "raw_output": "",
                    "error": repr(exc),
                }
            )
            write_record(out, record, started)

    if not prompts:
        return

    try:
        generated = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            args=args,
            device=device,
        )
        for (record, started), (raw_output, prompt_tokens, generated_tokens) in zip(
            prompt_items,
            generated,
        ):
            raw_predicted_label, predicted_label, confidence = parse_model_output(raw_output)
            record.update(
                {
                    "status": "ok",
                    "raw_predicted_label": raw_predicted_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "raw_output": raw_output,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                }
            )
            write_record(out, record, started)
    except Exception as batch_exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for prompt, (record, started) in zip(prompts, prompt_items):
            try:
                raw_output, prompt_tokens, generated_tokens = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    args=args,
                    device=device,
                )
                raw_predicted_label, predicted_label, confidence = parse_model_output(raw_output)
                record.update(
                    {
                        "status": "ok",
                        "raw_predicted_label": raw_predicted_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "raw_output": raw_output,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "batch_fallback_error": repr(batch_exc),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "raw_predicted_label": None,
                        "predicted_label": None,
                        "confidence": None,
                        "raw_output": "",
                        "error": repr(exc),
                        "batch_fallback_error": repr(batch_exc),
                    }
                )
            write_record(out, record, started)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    paths = resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_path(args)

    preview_prompt = None
    if paths and (args.dry_run or args.print_prompt):
        preview_prompt = build_prompt(load_case(paths[0]), args)

    if args.dry_run:
        payload = {
            "input_dir": str(args.input_dir),
            "output_file": str(pred_path),
            "model_name": args.model_name,
            "model_subfolder": optional_subfolder(args.model_subfolder),
            "adapter_mode": args.adapter_mode,
            "prompt_profile": args.prompt_profile,
            "label_space_prompt": "0=reject, 1=accept",
            "label_space_metrics": "-1=reject, 1=accept",
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "batch_size": args.batch_size,
            "cases_for_this_shard": len(paths),
            "first_case": paths[0].name if paths else None,
        }
        if args.print_prompt:
            payload["prompt_preview"] = preview_prompt
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    device = resolve_device(args.device)
    print(
        json.dumps(
            {
                "model": args.model_name,
                "model_subfolder": optional_subfolder(args.model_subfolder),
                "adapter_mode": args.adapter_mode,
                "prompt_profile": args.prompt_profile,
                "label_space_prompt": "0=reject, 1=accept",
                "label_space_metrics": "-1=reject, 1=accept",
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_device_count": torch.cuda.device_count(),
                "cases_for_this_shard": len(paths),
                "output_file": str(pred_path),
                "batch_size": args.batch_size,
            }
        ),
        flush=True,
    )

    tokenizer = load_tokenizer(args)
    model = load_model(args)
    model.to(device)
    model.eval()

    seen = existing_case_ids(pred_path) if args.skip_existing else set()
    mode = "a" if args.skip_existing else "w"
    with pred_path.open(mode, encoding="utf-8") as out:
        batch: list[tuple[Path, dict[str, Any], dict[str, Any], float]] = []
        for path in tqdm(paths, desc=f"shard {args.shard_index:02d}", unit="case"):
            case = load_case(path)
            case_id = str(case.get("case_id") or path.stem)
            if case_id in seen:
                continue

            started = time.time()
            record: dict[str, Any] = {
                "case_id": case_id,
                "file_name": case.get("file_name") or path.name,
                "source_path": str(path),
                "true_label": normalize_project_label(case.get("raw_label")),
                "model_name": args.model_name,
                "model_subfolder": optional_subfolder(args.model_subfolder),
                "adapter_mode": args.adapter_mode,
                "prompt_profile": args.prompt_profile,
                "label_space_prompt": "0=reject, 1=accept",
                "label_space_metrics": "-1=reject, 1=accept",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "batch_size": args.batch_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            batch.append((path, case, record, started))
            if len(batch) >= args.batch_size:
                process_batch(batch, model, tokenizer, args, device, out)
                for _path, _case, written_record, _started in batch:
                    seen.add(str(written_record.get("case_id")))
                batch = []

        if batch:
            process_batch(batch, model, tokenizer, args, device, out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
