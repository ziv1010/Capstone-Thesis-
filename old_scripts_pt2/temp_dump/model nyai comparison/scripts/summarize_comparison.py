#!/usr/bin/env python3
"""Summarize one-document OpenNyai vs replication outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc_id", required=True)
    parser.add_argument("--workspace_root", required=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def normalize_entity_text(text: str) -> str:
    normalized = normalize_text(text).lower()
    return normalized.replace("dist.", "dist")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_open_nyai_entities(open_combined: dict) -> list[dict]:
    entities = []
    seen = set()
    for annotation in open_combined["raw_result"].get("annotations", []):
        for entity in annotation.get("entities", []):
            key = (
                int(entity["start"]),
                int(entity["end"]),
                str(entity["labels"][0]).strip().upper(),
                str(entity["text"]),
            )
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                {
                    "start": key[0],
                    "end": key[1],
                    "label": key[2],
                    "text": key[3],
                }
            )
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
    return entities


def collect_rr_rows_from_open_nyai(open_rr: dict) -> list[dict]:
    rows = []
    for index, sentence in enumerate(open_rr.get("sentences", []), start=1):
        rows.append(
            {
                "index": index,
                "text": str(sentence.get("text", "")),
                "normalized_text": normalize_text(sentence.get("text", "")),
                "label": str(sentence.get("rhetorical_role", "")).strip().upper(),
            }
        )
    return rows


def collect_rr_rows_from_replication(replication_rr: list[dict]) -> list[dict]:
    if not replication_rr:
        return []
    annotations = replication_rr[0].get("annotations", [])
    if not annotations:
        return []
    rows = []
    for index, sentence in enumerate(annotations[0].get("result", []), start=1):
        value = sentence.get("value", {})
        labels = value.get("labels", []) or [""]
        rows.append(
            {
                "index": index,
                "start": int(value.get("start", 0)),
                "end": int(value.get("end", 0)),
                "text": str(value.get("text", "")),
                "normalized_text": normalize_text(value.get("text", "")),
                "label": str(labels[0]).strip().upper(),
            }
        )
    return rows


def occurrence_keyed_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    counts: defaultdict[str, int] = defaultdict(int)
    keyed = {}
    for row in rows:
        counts[row["normalized_text"]] += 1
        keyed[(row["normalized_text"], counts[row["normalized_text"]])] = row
    return keyed


def build_markdown(summary: dict) -> str:
    lines = []
    lines.append(f"# Comparison Summary: {summary['doc_id']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- OpenNyai wrapper: NER + RR + summarizer pipeline output")
    lines.append("- Replication path: original NER repo path and original rhetorical-role repo path")
    lines.append("- This comparison focuses on NER and rhetorical roles. The summarizer exists only on the OpenNyai side.")
    lines.append("")
    lines.append("## NER")
    lines.append("")
    ner = summary["ner"]
    lines.append(f"- OpenNyai entities: {ner['open_nyai_count']}")
    lines.append(f"- Replication NER entities: {ner['replication_count']}")
    lines.append(f"- Shared entities: {ner['shared_count']}")
    lines.append(f"- Only in OpenNyai: {ner['open_only_count']}")
    lines.append(f"- Only in replication: {ner['replication_only_count']}")
    lines.append(
        "- Normalized text+label overlap "
        f"(ignores span offsets and casing): {ner['normalized_shared_count']} shared, "
        f"{ner['normalized_open_only_count']} OpenNyai-only, "
        f"{ner['normalized_replication_only_count']} replication-only"
    )
    if ner["open_only_examples"]:
        lines.append("")
        lines.append("OpenNyai-only examples:")
        for item in ner["open_only_examples"][:10]:
            lines.append(f"- `{item['label']}` {item['start']}-{item['end']}: {item['text']}")
    if ner["replication_only_examples"]:
        lines.append("")
        lines.append("Replication-only examples:")
        for item in ner["replication_only_examples"][:10]:
            lines.append(f"- `{item['label']}` {item['start']}-{item['end']}: {item['text']}")
    if ner["normalized_open_only_examples"]:
        lines.append("")
        lines.append("Normalized OpenNyai-only examples:")
        for item in ner["normalized_open_only_examples"][:10]:
            lines.append(f"- `{item['label']}`: {item['text']}")
    if ner["normalized_replication_only_examples"]:
        lines.append("")
        lines.append("Normalized replication-only examples:")
        for item in ner["normalized_replication_only_examples"][:10]:
            lines.append(f"- `{item['label']}`: {item['text']}")
    lines.append("")
    lines.append("## Rhetorical Role")
    lines.append("")
    rr = summary["rr"]
    lines.append(f"- OpenNyai sentence count: {rr['open_nyai_sentence_count']}")
    lines.append(f"- Replication RR sentence count: {rr['replication_sentence_count']}")
    lines.append(f"- Alignment strategy: {rr['alignment_strategy']}")
    lines.append(f"- Comparable sentences: {rr['comparable_sentence_count']}")
    lines.append(f"- Matching labels: {rr['matching_label_count']}")
    lines.append(f"- Differing labels: {rr['differing_label_count']}")
    lines.append(f"- OpenNyai-only sentences after alignment: {rr['open_only_sentence_count']}")
    lines.append(f"- Replication-only sentences after alignment: {rr['replication_only_sentence_count']}")
    if rr["differing_sentences"]:
        lines.append("")
        lines.append("Label differences:")
        for item in rr["differing_sentences"][:15]:
            lines.append(
                f"- `{item['open_nyai_label']}` vs `{item['replication_label']}`: {item['text']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    outputs_root = workspace_root / "outputs"

    open_combined = load_json(outputs_root / "open_nyai" / "combined" / f"{args.doc_id}.json")
    open_rr = load_json(outputs_root / "open_nyai" / "rhetorical_roles" / f"{args.doc_id}.json")
    replication_ner = load_json(outputs_root / "replication_ner" / f"{args.doc_id}.json")
    replication_rr = load_json(outputs_root / "replication_rr" / f"{args.doc_id}.predictions.json")

    open_entities = collect_open_nyai_entities(open_combined)
    replication_entities = replication_ner.get("entities", [])

    open_entity_set = {
        (int(item["start"]), int(item["end"]), str(item["label"]).strip().upper(), str(item["text"]))
        for item in open_entities
    }
    replication_entity_set = {
        (int(item["start"]), int(item["end"]), str(item["label"]).strip().upper(), str(item["text"]))
        for item in replication_entities
    }
    open_entity_normalized_set = {
        (str(item["label"]).strip().upper(), normalize_entity_text(item["text"])) for item in open_entities
    }
    replication_entity_normalized_set = {
        (str(item["label"]).strip().upper(), normalize_entity_text(item["text"]))
        for item in replication_entities
    }
    shared_entities = sorted(open_entity_set & replication_entity_set)
    open_only_entities = sorted(open_entity_set - replication_entity_set)
    replication_only_entities = sorted(replication_entity_set - open_entity_set)
    normalized_shared_entities = sorted(open_entity_normalized_set & replication_entity_normalized_set)
    normalized_open_only_entities = sorted(open_entity_normalized_set - replication_entity_normalized_set)
    normalized_replication_only_entities = sorted(
        replication_entity_normalized_set - open_entity_normalized_set
    )

    open_rr_rows = collect_rr_rows_from_open_nyai(open_rr)
    replication_rr_rows = collect_rr_rows_from_replication(replication_rr)

    if len(open_rr_rows) == len(replication_rr_rows) and all(
        left["normalized_text"] == right["normalized_text"]
        for left, right in zip(open_rr_rows, replication_rr_rows)
    ):
        alignment_strategy = "index"
        comparable_pairs = list(zip(open_rr_rows, replication_rr_rows))
        open_only_rows = []
        replication_only_rows = []
    else:
        alignment_strategy = "normalized_text_occurrence"
        open_keyed = occurrence_keyed_rows(open_rr_rows)
        replication_keyed = occurrence_keyed_rows(replication_rr_rows)
        common_keys = sorted(set(open_keyed) & set(replication_keyed))
        comparable_pairs = [(open_keyed[key], replication_keyed[key]) for key in common_keys]
        open_only_rows = [open_keyed[key] for key in sorted(set(open_keyed) - set(replication_keyed))]
        replication_only_rows = [replication_keyed[key] for key in sorted(set(replication_keyed) - set(open_keyed))]

    differing_sentences = []
    matching_label_count = 0
    for open_row, replication_row in comparable_pairs:
        if open_row["label"] == replication_row["label"]:
            matching_label_count += 1
            continue
        differing_sentences.append(
            {
                "text": open_row["text"],
                "open_nyai_label": open_row["label"],
                "replication_label": replication_row["label"],
            }
        )

    summary = {
        "doc_id": args.doc_id,
        "paths": {
            "workspace_root": str(workspace_root),
            "open_nyai_combined": str(outputs_root / "open_nyai" / "combined" / f"{args.doc_id}.json"),
            "open_nyai_rhetorical_roles": str(outputs_root / "open_nyai" / "rhetorical_roles" / f"{args.doc_id}.json"),
            "replication_ner": str(outputs_root / "replication_ner" / f"{args.doc_id}.json"),
            "replication_rr": str(outputs_root / "replication_rr" / f"{args.doc_id}.predictions.json"),
        },
        "ner": {
            "open_nyai_count": len(open_entity_set),
            "replication_count": len(replication_entity_set),
            "shared_count": len(shared_entities),
            "open_only_count": len(open_only_entities),
            "replication_only_count": len(replication_only_entities),
            "normalized_shared_count": len(normalized_shared_entities),
            "normalized_open_only_count": len(normalized_open_only_entities),
            "normalized_replication_only_count": len(normalized_replication_only_entities),
            "open_label_counts": dict(Counter(item[2] for item in open_entity_set)),
            "replication_label_counts": dict(Counter(item[2] for item in replication_entity_set)),
            "open_only_examples": [
                {"start": item[0], "end": item[1], "label": item[2], "text": item[3]}
                for item in open_only_entities[:25]
            ],
            "replication_only_examples": [
                {"start": item[0], "end": item[1], "label": item[2], "text": item[3]}
                for item in replication_only_entities[:25]
            ],
            "normalized_open_only_examples": [
                {"label": item[0], "text": item[1]} for item in normalized_open_only_entities[:25]
            ],
            "normalized_replication_only_examples": [
                {"label": item[0], "text": item[1]}
                for item in normalized_replication_only_entities[:25]
            ],
        },
        "rr": {
            "alignment_strategy": alignment_strategy,
            "open_nyai_sentence_count": len(open_rr_rows),
            "replication_sentence_count": len(replication_rr_rows),
            "comparable_sentence_count": len(comparable_pairs),
            "matching_label_count": matching_label_count,
            "differing_label_count": len(differing_sentences),
            "open_only_sentence_count": len(open_only_rows),
            "replication_only_sentence_count": len(replication_only_rows),
            "differing_sentences": differing_sentences[:50],
            "open_only_sentences": open_only_rows[:20],
            "replication_only_sentences": replication_only_rows[:20],
        },
    }

    reports_dir = outputs_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "comparison_summary.json"
    md_path = reports_dir / "comparison_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
