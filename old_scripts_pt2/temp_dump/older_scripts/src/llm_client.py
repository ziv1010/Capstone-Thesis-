from __future__ import annotations

import json
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .schema import LLM_OUTPUT_SCHEMA


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class HuggingFaceLLMClient:
    model_name: str
    temperature: float = 0.0
    max_new_tokens: int = 4096
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        dtype_name = str(self.torch_dtype).strip()
        if not dtype_name:
            return None
        if dtype_name == "auto":
            return "auto"
        if hasattr(torch_module, dtype_name):
            return getattr(torch_module, dtype_name)
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        load_start = time.perf_counter()
        print(
            f"[LLM] Loading model={self.model_name} device_map={self.device_map} dtype={self.torch_dtype}",
            flush=True,
        )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing LLM dependencies. Install torch, transformers, and accelerate."
            ) from exc

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        dtype = self._resolve_torch_dtype(torch)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print(f"[LLM] Model ready in {time.perf_counter() - load_start:.1f}s", flush=True)

    def _input_device(self) -> Any:
        if self._model is None:
            return "cpu"
        return next(self._model.parameters()).device

    def _generate_with_heartbeat(
        self,
        generate_fn: Callable[[], Any],
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> Any:
        started_at = time.perf_counter()
        print(
            f"[LLM] Generation started | prompt_tokens={prompt_tokens:,} "
            f"max_new_tokens={max_new_tokens:,}",
            flush=True,
        )
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(20.0):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[LLM] Generation in progress | elapsed={elapsed:.1f}s",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        success = False
        try:
            result = generate_fn()
            success = True
            return result
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=0.1)
            status = "finished" if success else "failed"
            elapsed = time.perf_counter() - started_at
            print(f"[LLM] Generation {status} in {elapsed:.1f}s", flush=True)

    def generate(self, prompt: str) -> str:
        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None

        system_message = (
            "You are an extraction engine. Return strict JSON only, with no markdown or extra text."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            # Use tokenize=False to get a formatted string, then encode
            # separately.  Some transformers versions return a BatchEncoding
            # (not a plain Tensor) when tokenize=True, which breaks
            # model.generate() at `inputs_tensor.shape[0]`.
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self._tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self._input_device())
            do_sample = float(self.temperature) > 0.0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(self.max_new_tokens),
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(self.temperature)
            else:
                # Suppress UserWarnings about temperature/top_p/top_k being set
                # in the model's default generation_config when do_sample=False.
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None
            if self._tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            if self._tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

            output_ids = self._generate_with_heartbeat(
                generate_fn=lambda: self._model.generate(input_ids, **gen_kwargs),
                prompt_tokens=int(input_ids.shape[-1]),
                max_new_tokens=int(self.max_new_tokens),
            )
            generated_ids = output_ids[0, input_ids.shape[-1] :]
            print(f"[LLM] Generated {generated_ids.shape[-1]:,} new tokens", flush=True)
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self._input_device()) for k, v in encoded.items()}
        do_sample = float(self.temperature) > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(self.temperature)
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
        if self._tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
        if self._tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        output_ids = self._generate_with_heartbeat(
            generate_fn=lambda: self._model.generate(**encoded, **gen_kwargs),
            prompt_tokens=int(encoded["input_ids"].shape[-1]),
            max_new_tokens=int(self.max_new_tokens),
        )
        prompt_tokens = encoded["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_tokens:]
        print(f"[LLM] Generated {generated_ids.shape[-1]:,} new tokens", flush=True)
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _format_paragraphs_for_prompt(paragraphs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for para in paragraphs:
        idx = para.get("index")
        text = str(para.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def build_llm_prompt(
    raw_text: str,
    paragraphs: list[dict[str, Any]],
    ner: dict[str, list[str]],
    regex: dict[str, list[str]],
    max_chars_to_send: int = 0,
) -> str:
    truncated_text = raw_text[:max_chars_to_send] if max_chars_to_send > 0 else raw_text
    para_text = _format_paragraphs_for_prompt(paragraphs)

    instructions = {
        "task": "Extract a strict JSON case record for an Indian court judgment/order",
        "hard_rules": [
            "Return JSON ONLY. No markdown. No explanation.",
            "All schema keys must be present. Use null or [] when unknown.",
            "Do NOT copy the full judgment into output fields.",
            "Use verbatim text spans from supplied paragraphs where possible.",
            "Use short excerpts only (1-3 sentences per field, concise).",
            "If uncertain, prefer null over long speculative text.",
            "texts.raw_text must be an empty string ('').",
            "Put dispositive/outcome language ONLY in texts.decision_text and outcome.* — never in any other field.",
            "texts.facts_text must contain ONLY background facts and procedural history. No rulings.",
            "texts.arguments_petitioner must contain ONLY petitioner/applicant/prosecution side submissions.",
            "texts.arguments_respondent must contain ONLY respondent/state/defense side submissions.",
            "texts.reasoning_text may contain court analysis/reasoning paragraphs ONLY IF they do not reveal the final outcome.",
            "ml.input_text = concatenation of facts_text + arguments_petitioner + arguments_respondent ONLY. Must be completely free of outcome/disposition language.",
            "If you removed a paragraph from ml.input_text for safety, record it in ml.removed_spans with paragraph_index and reason='outcome_phrase'.",
            "Fill both arguments fields best-effort: petitioner/prosecution side vs respondent/defense side.",
            "Map sides correctly: petitioner_applicant = person who filed (bail applicant, writ petitioner, appellant); respondent_state_defendant = state/opposite party.",
            "decision must be a short final label in snake_case.",
            "Prefer decision='for_appellant' or decision='against_appellant'.",
            "If neither side clearly won/lost, set decision to a procedural status such as 'dismissed', 'delayed', 'withdrawn', 'remanded', or another concise status.",
            "Date must be YYYY-MM-DD or null.",
            "Keep key_facts_bullets / issues_bullets / holdings_bullets to max 5 items each.",
        ],
        "leakage_safety": {
            "MUST go into decision_text (NEVER into input_text)": [
                "Any sentence containing: 'is/are allowed', 'is/are dismissed', 'is/are rejected', 'is/are granted', 'is/are refused', 'is/are quashed'",
                "Any sentence containing: 'anticipatory bail is granted/rejected/refused'",
                "Any sentence containing: 'be released on bail', 'released on bail'",
                "Any sentence containing: 'petition succeeds', 'petition fails'",
                "Any sentence containing: 'prayer is granted/rejected/dismissed'",
                "Any sentence containing: 'rule is made absolute', 'rule is discharged'",
                "Any sentence containing: 'proceedings are quashed', 'proceedings stand quashed'",
                "Any sentence containing: 'not inclined to grant', 'inclined to grant'",
                "Any sentence containing: 'I am of the view that bail should/should not be granted'",
                "Any sentence containing: 'The following order is passed', 'ORDER:', 'ORAL ORDER:'",
                "Final numbered court directions like '(i) In the event of arrest... be released...'",
                "Any sentence revealing who won or lost.",
            ],
            "SAFE to include in input_text": [
                "Facts: what happened, background, FIR details, charges, procedural history.",
                "Arguments made by petitioner/applicant counsel.",
                "Arguments made by respondent/state/APP counsel.",
                "Court's neutral recitation of facts ('It appears from the record that...').",
                "Cited precedents and how they were argued (not applied as rulings).",
                "Section/provision numbers referenced in arguments.",
                "Phrases like 'in order to submit', 'an order dated X was passed by the trial court' (historical orders, not THIS court's outcome).",
            ],
        },
        "schema": LLM_OUTPUT_SCHEMA,
        "ner_hints": ner,
        "regex_hints": regex,
    }

    prompt = (
        "You are a legal information extraction engine specializing in Indian court documents.\n"
        "Your PRIMARY responsibility is leakage safety: ml.input_text must NEVER reveal the outcome.\n"
        "Use the provided text to populate the JSON schema exactly.\n\n"
        f"INSTRUCTIONS:\n{json.dumps(instructions, ensure_ascii=False, indent=2)}\n\n"
        "RAW_TEXT (possibly truncated):\n"
        f"{truncated_text}\n\n"
        "PARAGRAPHS WITH INDICES:\n"
        f"{para_text}\n\n"
        "CRITICAL: Before returning, re-read ml.input_text and verify it contains zero outcome phrases.\n"
        "Return strict JSON only."
    )
    return prompt


def build_json_fix_prompt(
    invalid_response_text: str,
    validation_error: str,
    source_prompt: str = "",
    partial_record: dict[str, Any] | None = None,
    max_invalid_chars: int = 4000,
    max_source_chars: int = 14000,
    max_partial_chars: int = 8000,
) -> str:
    # Cap the broken response that we echo back: if it was truncated the first
    # time, sending the full broken text bloats the fix-prompt and causes the
    # second generation to also be cut off.
    truncated_invalid = invalid_response_text[:max_invalid_chars]
    if len(invalid_response_text) > max_invalid_chars:
        truncated_invalid += "\n... [truncated]"

    source_snippet = source_prompt[:max_source_chars]
    if len(source_prompt) > max_source_chars:
        source_snippet += "\n... [truncated]"

    partial_json = json.dumps(partial_record or {}, ensure_ascii=False)
    if len(partial_json) > max_partial_chars:
        partial_json = partial_json[:max_partial_chars] + "\n... [truncated]"

    missing_fields = _summarize_missing_fields(validation_error)
    missing_fields_text = "\n".join(f"- {item}" for item in missing_fields) if missing_fields else "- unknown"

    top_level_keys = list((LLM_OUTPUT_SCHEMA.get("properties") or {}).keys())
    return (
        "Continue extraction from a PARTIAL JSON for the same document.\n"
        "Return JSON only. No markdown. No explanation.\n"
        "Return ONLY fields that are missing/invalid; do not repeat long already-correct text fields.\n"
        "Keep values concise and factual.\n\n"
        f"TARGET_TOP_LEVEL_KEYS:\n{json.dumps(top_level_keys, ensure_ascii=False)}\n\n"
        f"MISSING_OR_INVALID_FIELDS:\n{missing_fields_text}\n\n"
        f"VALIDATION_ERROR:\n{validation_error}\n\n"
        f"CURRENT_PARTIAL_JSON (may be truncated):\n{partial_json}\n\n"
        f"ORIGINAL_SOURCE_PROMPT (may be truncated):\n{source_snippet}\n\n"
        f"LAST_INVALID_RESPONSE (may be truncated):\n{truncated_invalid}"
    )


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    # Strip <think>...</think> blocks emitted by some Qwen reasoning models.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _auto_close_json(text: str) -> str:
    """Balance unclosed brackets/braces in a truncated JSON string.

    If the LLM was cut off mid-output, the JSON may be missing its closing
    chars. This heuristic appends the right number of ]/} to make it parseable.
    It also handles a dangling comma before the inserted closing chars.
    """
    stack: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}" , "]") and stack and stack[-1] == ch:
            stack.pop()

    # Strip a trailing comma that would make the JSON invalid before closing.
    suffix = text.rstrip()
    if suffix.endswith(","):
        suffix = suffix[:-1]

    return suffix + "".join(reversed(stack))


def _summarize_missing_fields(validation_error: str, max_items: int = 24) -> list[str]:
    msg = str(validation_error).replace("Schema validation failed: ", "").strip()
    if not msg:
        return []

    items: list[str] = []
    seen: set[str] = set()
    for part in msg.split(" | "):
        part = part.strip()
        if not part:
            continue

        required_match = re.match(r"^([^:]+): '([^']+)' is a required property$", part)
        if required_match:
            parent_path = required_match.group(1).strip()
            child_key = required_match.group(2).strip()
            path = child_key if parent_path in ("$", "") else f"{parent_path}.{child_key}"
        else:
            path = part.split(":", 1)[0].strip()

        if path in ("case_id", "file_name"):
            continue
        if path and path not in seen:
            seen.add(path)
            items.append(path)
        if len(items) >= max_items:
            break
    return items


def _merge_json_values(base: Any, incoming: Any) -> Any:
    if base is None:
        return incoming
    if incoming is None:
        return base

    if isinstance(base, dict) and isinstance(incoming, dict):
        merged: dict[str, Any] = dict(base)
        for key, incoming_val in incoming.items():
            if key in merged:
                merged[key] = _merge_json_values(merged[key], incoming_val)
            else:
                merged[key] = incoming_val
        return merged

    if isinstance(base, list) and isinstance(incoming, list):
        if all(isinstance(x, str) for x in base + incoming):
            merged_list: list[str] = []
            seen: set[str] = set()
            for item in base + incoming:
                normalized = re.sub(r"\s+", " ", item).strip()
                if not normalized:
                    continue
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged_list.append(normalized)
            return merged_list

        merged_generic: list[Any] = list(base)
        for item in incoming:
            if item not in merged_generic:
                merged_generic.append(item)
        return merged_generic

    if isinstance(base, str) and isinstance(incoming, str):
        if incoming.strip():
            return incoming
        return base

    return incoming


def parse_json_from_llm_text(response_text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, with best-effort partial recovery.

    Strategy (in order):
    1. Direct json.loads on stripped text.
    2. Extract the outermost {...} substring and try json.loads.
    3. Try json_repair (if installed) on the candidate.
    4. Manually balance brackets with _auto_close_json and try json.loads.
    5. Raise ValueError so the caller can handle the failure.
    """
    cleaned = _strip_fences(response_text)

    # 1. Clean parse.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object boundaries.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = cleaned[start:] if start >= 0 else cleaned  # may be truncated (no closing })

    if start >= 0 and end > start:
        candidate_closed = cleaned[start : end + 1]
        # 2. Try the cleanly bounded substring.
        try:
            parsed = json.loads(candidate_closed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    else:
        candidate_closed = candidate

    # 3. Try json_repair (pip install json-repair) — handles many truncation patterns.
    try:
        import json_repair  # type: ignore
        repaired = json_repair.repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            print("[LLM] parse_json: recovered partial response via json_repair", flush=True)
            return repaired
    except (ImportError, Exception):
        pass

    # 4. Manual bracket-balancing on the truncated candidate.
    try:
        closed = _auto_close_json(candidate)
        parsed = json.loads(closed)
        if isinstance(parsed, dict) and parsed:
            print("[LLM] parse_json: recovered partial response via bracket-balancing", flush=True)
            return parsed
    except (json.JSONDecodeError, Exception):
        pass

    # 5. Nothing worked.
    try:
        # Re-raise with a descriptive error using the original candidate.
        json.loads(candidate_closed)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON object from LLM response (truncated/malformed): {e}"
        ) from e
    raise ValueError("Could not parse a JSON object from LLM response")


def extract_case_json_with_retry(
    client: LLMClientProtocol,
    prompt: str,
    validate_fn: Callable[[dict[str, Any]], None],
    max_attempts: int = 4,
) -> dict[str, Any]:
    max_attempts = max(1, int(max_attempts))
    merged_record: dict[str, Any] = {}
    last_response_text = ""
    last_error = "unknown"
    parsed_any_response = False

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            attempt_prompt = prompt
            print(f"[LLM] Attempt {attempt}/{max_attempts}: primary extraction", flush=True)
        else:
            attempt_prompt = build_json_fix_prompt(
                invalid_response_text=last_response_text,
                validation_error=last_error,
                source_prompt=prompt,
                partial_record=merged_record,
            )
            print(
                f"[LLM] Attempt {attempt}/{max_attempts}: continuation for missing fields",
                flush=True,
            )

        response_text = client.generate(attempt_prompt)
        last_response_text = response_text

        try:
            parsed = parse_json_from_llm_text(response_text)
            if not parsed:
                raise ValueError("Parsed empty JSON object from LLM response")
            parsed_any_response = True
            merged_record = _merge_json_values(merged_record, parsed)
            assert isinstance(merged_record, dict)
        except Exception as parse_error:
            last_error = f"parse_error: {repr(parse_error)}"
            print(f"[LLM] Attempt {attempt}/{max_attempts} parse failed: {repr(parse_error)}", flush=True)
            if attempt == max_attempts:
                break
            continue

        try:
            validate_fn(merged_record)
            print(f"[LLM] Attempt {attempt}/{max_attempts}: validation passed", flush=True)
            return merged_record
        except Exception as validation_error:
            last_error = str(validation_error)
            print(
                f"[LLM] Attempt {attempt}/{max_attempts}: validation incomplete, continuing",
                flush=True,
            )
            print(f"[LLM] Validation issue: {repr(validation_error)}", flush=True)

    if parsed_any_response and merged_record:
        print(
            "[LLM] Returning best-effort merged extraction after max attempts",
            flush=True,
        )
        return merged_record

    raise RuntimeError(
        f"LLM extraction failed after {max_attempts} attempts. Last error: {last_error}"
    )
