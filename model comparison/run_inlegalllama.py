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
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "L-NLProc/InLegalLlama"
DEFAULT_MODEL_SUBFOLDER = "INLegalLlama/CPT/llama2_cpt_checkpoint_3000_seq_2048"
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
        description="Run L-NLProc/InLegalLlama over filtered cleaned-case JSON files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inlegalllama_run"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-subfolder", default=DEFAULT_MODEL_SUBFOLDER)
    parser.add_argument("--adapter-mode", choices=["auto", "peft", "full"], default="auto")
    parser.add_argument("--case-glob", default="*.json")
    parser.add_argument("--sections", nargs="+", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--max-section-chars", type=int, default=12000)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
    return parser.parse_args()


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n[...truncated...]\n{text[-tail:]}"


def build_prompt(case: dict[str, Any], sections: list[str], max_section_chars: int) -> str:
    title = case.get("case_id") or case.get("file_name") or "unknown case"
    blocks = []
    texts = case.get("texts") or {}
    for key in sections:
        text = compact_text(texts.get(key, ""), max_section_chars)
        if text:
            blocks.append(f"{SECTION_TITLES.get(key, key)}:\n{text}")

    case_text = "\n\n".join(blocks) if blocks else "No usable text sections were found."
    return f"""You are classifying an Indian legal case outcome from preprocessed case text.

Predict the binary case_outcome_score used in this project:
1 = the appellant, petitioner, claimant, or moving party substantially won or received substantive relief.
-1 = the appellant, petitioner, claimant, or moving party lost, received no substantive relief, or the matter is procedural/neutral.

The text has already been preprocessed, filtered, and outcome-leakage phrases may have been masked. Use only the provided text.

Return exactly one JSON object with these keys:
{{"predicted_label": "1" or "-1", "confidence": number from 0 to 1, "rationale": "one short sentence"}}

Case title: {title}

{case_text}
"""


def output_path(args: argparse.Namespace) -> Path:
    if args.num_shards > 1:
        return args.output_dir / f"predictions_shard_{args.shard_index:02d}.jsonl"
    return args.output_dir / "predictions.jsonl"


def optional_subfolder(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("/")
    return value or None


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    paths = sorted(args.input_dir.glob(args.case_glob))
    paths = [path for idx, path in enumerate(paths) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = payload.get("case_id")
            if case_id:
                seen.add(str(case_id))
    return seen


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.strip("\"'` .,;:")
    positive = {
        "1",
        "+1",
        "won",
        "win",
        "winning",
        "allowed",
        "appellant_won",
        "petitioner_won",
        "claimant_won",
        "for_appellant",
        "for_petitioner",
        "for_claimant",
    }
    negative = {
        "-1",
        "lost",
        "lose",
        "losing",
        "dismissed",
        "appellant_lost",
        "petitioner_lost",
        "claimant_lost",
        "against_appellant",
        "against_petitioner",
        "against_claimant",
        "procedural",
        "neutral",
    }
    if text in positive:
        return "1"
    if text in negative:
        return "-1"
    if re.fullmatch(r"\+?1(?:\.0)?", text):
        return "1"
    if re.fullmatch(r"-1(?:\.0)?", text):
        return "-1"
    return None


def parse_model_output(text: str) -> tuple[str | None, float | None]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            label = normalize_label(
                payload.get("predicted_label")
                or payload.get("case_outcome_score")
                or payload.get("label")
            )
            confidence = payload.get("confidence")
            try:
                confidence_value = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_value = None
            if label is not None:
                return label, confidence_value
        except json.JSONDecodeError:
            pass

    keyed = re.search(
        r"(?:predicted_label|case_outcome_score|label)\s*[:=]\s*[\"']?(-?1)[\"']?",
        text,
        flags=re.IGNORECASE,
    )
    if keyed:
        return normalize_label(keyed.group(1)), None

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    standalone = re.search(r"(?<![\d-])-1(?!\d)|(?<![\d-])1(?!\d)", first_line)
    if standalone:
        return normalize_label(standalone.group(0)), None
    return None, None


def torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def should_load_as_peft(args: argparse.Namespace) -> bool:
    if args.adapter_mode == "peft":
        return True
    if args.adapter_mode == "full":
        return False
    return args.model_name == DEFAULT_MODEL and optional_subfolder(args.model_subfolder) is not None


def load_tokenizer(args: argparse.Namespace) -> Any:
    subfolder = optional_subfolder(args.model_subfolder)
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if subfolder:
        tokenizer_kwargs["subfolder"] = subfolder
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace) -> Any:
    dtype = torch_dtype(args.dtype)
    subfolder = optional_subfolder(args.model_subfolder)
    if should_load_as_peft(args):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.model_name, subfolder=subfolder)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        return PeftModel.from_pretrained(base_model, args.model_name, subfolder=subfolder)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if subfolder:
        model_kwargs["subfolder"] = subfolder
    return AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def tokenize_prompt(tokenizer: Any, prompt: str, max_input_tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    else:
        encoded = tokenizer(
            prompt + "\n\nAnswer:",
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    return {key: value.to(device) for key, value in encoded.items()}


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt + "\n\nAnswer:"


def tokenize_prompt_batch(
    tokenizer: Any,
    prompts: list[str],
    max_input_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    use_chat_template = bool(getattr(tokenizer, "chat_template", None))
    rendered = [render_prompt(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=not use_chat_template,
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
            prompts.append(build_prompt(case, args.sections, args.max_section_chars))
            prompt_items.append((record, started))
        except Exception as exc:
            record.update(
                {
                    "status": "error",
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
            predicted_label, confidence = parse_model_output(raw_output)
            record.update(
                {
                    "status": "ok",
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
                predicted_label, confidence = parse_model_output(raw_output)
                record.update(
                    {
                        "status": "ok",
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

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_dir": str(args.input_dir),
                    "output_file": str(pred_path),
                    "model_name": args.model_name,
                    "model_subfolder": optional_subfolder(args.model_subfolder),
                    "adapter_mode": args.adapter_mode,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "batch_size": args.batch_size,
                    "cases_for_this_shard": len(paths),
                    "first_case": paths[0].name if paths else None,
                },
                indent=2,
            )
        )
        return

    device = resolve_device(args.device)
    print(
        json.dumps(
            {
                "model": args.model_name,
                "model_subfolder": optional_subfolder(args.model_subfolder),
                "adapter_mode": args.adapter_mode,
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
                "true_label": normalize_label(case.get("raw_label")),
                "model_name": args.model_name,
                "model_subfolder": optional_subfolder(args.model_subfolder),
                "adapter_mode": args.adapter_mode,
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
