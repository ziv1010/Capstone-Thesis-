#!/usr/bin/env python
"""Phase 5: Feed each Phase-4 evidence bundle to a local Mistral (via vLLM)
and produce a natural-language explanation that is *strictly grounded* in the
provided statutes, precedents, arguments, and similar-case references.

Run this in the `llm` micromamba env (vLLM-capable)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyser.loader import load_config  # noqa: E402


SYSTEM_PROMPT = (
    "You are a legal reasoning assistant trained to interpret the output of a "
    "Heterogeneous Graph Transformer (HGT) that predicts Indian case outcomes. "
    "The model's label space is:\n"
    "  - petitioner_wins  (predicted class '1') — petition/appeal allowed / relief granted\n"
    "  - petitioner_loses (predicted class '-1') — petition/appeal dismissed / relief refused\n"
    "\n"
    "You will receive a JSON evidence bundle with:\n"
    "  * the target case's own facts / arguments (loaded from the record itself),\n"
    "  * top-importance statutes (parent acts, e.g. 'Code of Criminal Procedure'),\n"
    "  * top-importance provisions (specific sections, e.g. 'Section 482'),\n"
    "  * top-importance precedents (cases cited by the target),\n"
    "  * similar-case argument context (argument snippets from structurally "
    "similar cases whose argument nodes propagated influence through the graph),\n"
    "  * similar past training cases retrieved by embedding cosine similarity.\n"
    "\n"
    "Write a LEGAL-STYLE explanation, not an ML-style one. Avoid phrases like "
    "'the model was influenced by' or 'strong signal from precedent X'. Instead "
    "interpret the evidence the way a lawyer would: explain what the relied-on "
    "provisions and precedents legally establish, and tie them to the facts.\n"
    "\n"
    "HARD RULES:\n"
    "  1. Use ONLY the content in the JSON bundle. Do not invent citations.\n"
    "  2. Keep statutes and provisions as DISTINCT categories. If the bundle has "
    "provisions but no statutes (or vice-versa), do not contradict yourself.\n"
    "  3. Frame similar-case argument context as 'argument patterns observed in "
    "structurally similar cases', not as arguments made in the target case.\n"
    "  4. Frame retrieved similar training cases as 'analogous past cases' or "
    "'cases with similar fact patterns', never as direct citations.\n"
    "  5. When common CrPC / HMA provisions appear, interpret them briefly (e.g. "
    "Section 482 CrPC = inherent powers of the High Court; Section 320 CrPC = "
    "compounding of offences; Section 13B HMA = mutual-consent divorce). Only "
    "if such provisions actually appear in the bundle.\n"
    "  6. If a category is empty, say so in one line instead of fabricating.\n"
    "  7. Do not cite the target case as its own precedent (any such leakage "
    "has already been filtered, but never self-cite).\n"
)


USER_TEMPLATE = (
    "### Target case\n"
    "- Title: {case_id}\n"
    "- Predicted outcome: **{pred_label}** (confidence {conf:.3f})\n"
    "- Actual outcome (for reference): {gold_label}\n\n"
    "### Evidence bundle (JSON)\n"
    "```json\n{bundle_json}\n```\n\n"
    "### Produce the explanation in this exact structure\n"
    "**1. Predicted outcome.** One sentence stating the predicted label in "
    "legal terms (petition allowed / dismissed), with the confidence.\n\n"
    "**2. Governing statutes.** List the statutes (parent acts) with a "
    "one-line explanation of each. If none, write 'No specific parent statute "
    "was identified by the explainer.'\n\n"
    "**3. Key provisions.** List each section with (a) its legal meaning, "
    "(b) how it bears on the target case's facts. If none, say so.\n\n"
    "**4. Supporting precedents.** For each precedent, say briefly what legal "
    "proposition it stands for, based on its citation and how it connects to "
    "the provisions / facts. If none, say so.\n\n"
    "**5. Argument patterns from similar cases.** Summarise what argument "
    "theme(s) appear across the `similar_case_argument_context` snippets. "
    "Make clear these are patterns the GNN recognised, NOT arguments made in "
    "the target case.\n\n"
    "**6. Analogous past cases.** In 2–3 sentences, note that retrieval "
    "surfaced N highly similar training cases (give the top cases' titles and "
    "outcomes), and say that these are retrieval-based analogues, not cited "
    "authorities.\n\n"
    "**7. Bottom-line legal reasoning.** A 3–5 sentence paragraph, phrased "
    "the way a judge might write it: interpret the relied-on provisions, "
    "connect them to the precedents, apply them to the target case's facts, "
    "and conclude why the predicted outcome follows. No ML jargon.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of bundles processed (0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )
    parser.add_argument("--rank",       type=int, default=0, help="Worker rank (0-indexed)")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of parallel workers")
    return parser.parse_args()


LABEL_DISPLAY = {
    "1": "petitioner_wins",
    "-1": "petitioner_loses",
    "0": "petitioner_loses",
}


def humanize_label(raw_label: str | None) -> str:
    if raw_label is None:
        return "unknown"
    return LABEL_DISPLAY.get(str(raw_label), str(raw_label))


def compress_bundle_for_prompt(bundle: dict[str, Any]) -> dict[str, Any]:
    """Strip fields the LLM does not need, keep the legally meaningful ones,
    and truncate long text nodes so the prompt fits in context.
    """
    def _trim(txt: str, n: int) -> str:
        if txt is None:
            return ""
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    compact_top: dict[str, list[dict[str, Any]]] = {}
    for key in ("statute", "provision", "precedent"):
        items = bundle.get("top_nodes", {}).get(key, [])
        compact_top[key] = [
            {
                "text": _trim(item.get("text", ""), 200),
                "importance": round(float(item.get("importance", 0.0)), 4),
            }
            for item in items
        ]

    sim_ctx = []
    for item in bundle.get("similar_case_argument_context", []):
        sim_ctx.append(
            {
                "owning_case": item.get("owning_case"),
                "argument_section": item.get("argument_section"),
                "importance": round(float(item.get("importance", 0.0)), 4),
                "snippet": _trim(item.get("snippet", ""), 500),
            }
        )

    similar = []
    for sc in bundle.get("similar_training_cases", []):
        similar.append(
            {
                "rank": sc.get("rank"),
                "case_id": sc.get("case_id"),
                "actual_outcome": humanize_label(sc.get("target_label")),
                "model_outcome": humanize_label(sc.get("predicted_label")),
                "similarity": round(float(sc.get("similarity", 0.0)), 4),
            }
        )

    target_text_raw = bundle.get("target_case_text", {}) or {}
    target_text = {
        section: _trim(target_text_raw.get(section, ""), 800)
        for section in ("preamble", "facts", "arguments", "petitioner_arguments", "respondent_arguments")
        if target_text_raw.get(section)
    }

    return {
        "case_id": bundle.get("case_id"),
        "predicted_outcome": humanize_label(bundle.get("predicted_label")),
        "actual_outcome": humanize_label(bundle.get("target_label")),
        "confidence": round(float(bundle.get("confidence", 0.0)), 4),
        "target_case_text": target_text,
        "top_nodes": compact_top,
        "similar_case_argument_context": sim_ctx,
        "similar_training_cases": similar,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    phase4_dir = Path(cfg["output_root"]) / "phase4_explanations" / "cases"
    output_root = Path(cfg["output_root"]) / "phase5_llm_reasoning"
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_paths = sorted(phase4_dir.glob("case_*.json"))
    if not bundle_paths:
        raise FileNotFoundError(f"No bundles found in {phase4_dir}")
    if args.limit > 0:
        bundle_paths = bundle_paths[: args.limit]

    # Shard across workers; skip already-done files
    bundle_paths = bundle_paths[args.rank :: args.world_size]
    bundle_paths = [
        p for p in bundle_paths
        if not (output_root / f"explanation_{p.stem.split('_')[1]}.json").exists()
    ]
    if not bundle_paths:
        print(f"[phase5 rank={args.rank}] nothing to do (all cached)", flush=True)
        return
    print(f"[phase5 rank={args.rank}/{args.world_size}] {len(bundle_paths)} bundles to process", flush=True)

    llm_cfg = cfg.get("llm", {})
    model_path = str(llm_cfg["model_path"])

    print(f"[phase5 rank={args.rank}] loading vLLM with model: {model_path}", flush=True)

    # Imported lazily so the error messaging is clearer if vllm isn't in this env.
    from vllm import LLM, SamplingParams  # noqa: WPS433

    tp_size = int(llm_cfg.get("tensor_parallel_size", args.tensor_parallel_size))
    max_model_len = int(llm_cfg.get("max_model_len", args.max_model_len))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=float(llm_cfg.get("gpu_memory_utilization", 0.85)),
        dtype=str(llm_cfg.get("dtype", "bfloat16")),
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=float(llm_cfg.get("temperature", 0.2)),
        top_p=float(llm_cfg.get("top_p", 0.9)),
        max_tokens=int(llm_cfg.get("max_new_tokens", 700)),
    )

    # Build conversations as prompt strings via the tokenizer's chat template.
    tokenizer = llm.get_tokenizer()

    prompts: list[str] = []
    bundle_records: list[tuple[Path, dict[str, Any]]] = []
    for path in bundle_paths:
        with open(path, "r") as f:
            bundle = json.load(f)
        compact = compress_bundle_for_prompt(bundle)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    case_id=compact["case_id"],
                    pred_label=compact["predicted_outcome"],
                    conf=compact["confidence"],
                    gold_label=compact["actual_outcome"],
                    bundle_json=json.dumps(compact, indent=2),
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        bundle_records.append((path, bundle))

    print(f"[phase5 rank={args.rank}] generating explanations for {len(prompts)} cases...", flush=True)

    outputs = llm.generate(prompts, sampling)

    index_records: list[dict[str, Any]] = []
    for (path, bundle), output in zip(bundle_records, outputs):
        text = output.outputs[0].text.strip() if output.outputs else ""
        record = {
            "case_node_index": bundle.get("case_node_index"),
            "case_id": bundle.get("case_id"),
            "file_name": bundle.get("file_name"),
            "predicted_label": bundle.get("predicted_label"),
            "target_label": bundle.get("target_label"),
            "confidence": bundle.get("confidence"),
            "explanation": text,
        }
        out_path = output_root / f"explanation_{bundle.get('case_node_index')}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        index_records.append(
            {
                "case_node_index": record["case_node_index"],
                "case_id": record["case_id"],
                "predicted_label": record["predicted_label"],
                "target_label": record["target_label"],
                "explanation_path": str(out_path),
            }
        )

    with open(output_root / f"index_rank{args.rank}.json", "w") as f:
        json.dump(index_records, f, indent=2, ensure_ascii=False)

    print(f"[phase5 rank={args.rank}] done. {len(index_records)} explanations written.", flush=True)


if __name__ == "__main__":
    main()
