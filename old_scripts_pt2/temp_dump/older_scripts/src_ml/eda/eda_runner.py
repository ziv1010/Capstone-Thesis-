from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src_ml.common.io import load_cases_dataframe
from src_ml.common.labels import derive_model_labels
from src_ml.common.serialization import save_json
from src_ml.common.text_utils import char_count, safe_list, safe_text, word_count
from src_ml.eda.eda_plots import plot_bar_series, plot_histogram
from src_ml.eda.eda_report import build_eda_report, write_eda_report


def _value_counts_from_list_column(series: pd.Series, top_k: int) -> pd.Series:
    counts: dict[str, int] = {}
    for value in series:
        for item in safe_list(value):
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return pd.Series(dtype="int64")
    out = pd.Series(counts).sort_values(ascending=False)
    return out.head(top_k)


def _missing_rate_text(series: pd.Series) -> float:
    is_missing = series.fillna("").map(lambda x: len(safe_text(x)) == 0)
    return float(is_missing.mean() * 100.0)


def _missing_rate_list(series: pd.Series) -> float:
    is_missing = series.map(lambda x: len(safe_list(x)) == 0)
    return float(is_missing.mean() * 100.0)


def run_eda(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    output_cfg = config["outputs"]
    eda_cfg = config["eda"]

    out_root = Path(output_cfg["root"])
    tables_dir = out_root / "eda" / "tables"
    figures_dir = out_root / "eda" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_cases_dataframe(
        path=dataset_cfg["jsonl_path"],
        limit=dataset_cfg.get("limit"),
        chunk_size=int(dataset_cfg.get("chunk_size", 4096)),
    )
    if df.empty:
        raise ValueError("No rows loaded for EDA")

    labeled_df, _, _ = derive_model_labels(df, config["labels"], logger=logger)

    logger.info("EDA loaded rows=%d labeled_rows=%d", len(df), len(labeled_df))

    class_dist = labeled_df["y_name"].value_counts().sort_values(ascending=False)
    year_dist = df["year"].dropna().astype(int).value_counts().sort_index()
    court_dist = df["court"].fillna("UNKNOWN").value_counts().sort_values(ascending=False)
    case_type_dist = df["case_type"].fillna("UNKNOWN").value_counts().sort_values(ascending=False)

    missing_rows = [
        ("judge_names", _missing_rate_list(df["judge_names"])),
        ("statutes", _missing_rate_list(df["statutes"])),
        ("provisions", _missing_rate_list(df["provisions"])),
        ("precedents", _missing_rate_list(df["precedents"])),
        ("facts_text", _missing_rate_text(df["facts_text"])),
        ("arguments_petitioner", _missing_rate_text(df["arguments_petitioner"])),
        ("arguments_respondent", _missing_rate_text(df["arguments_respondent"])),
        ("ml_input_text", _missing_rate_text(df["ml_input_text"])),
    ]
    missing_df = pd.DataFrame(missing_rows, columns=["field", "missing_pct"])

    text_stats_rows: list[dict[str, Any]] = []
    for col in ["facts_text", "arguments_petitioner", "arguments_respondent", "ml_input_text"]:
        chars = df[col].fillna("").map(char_count)
        words = df[col].fillna("").map(word_count)
        text_stats_rows.append(
            {
                "field": col,
                "chars_mean": float(chars.mean()),
                "chars_median": float(chars.median()),
                "chars_p95": float(chars.quantile(0.95)),
                "words_mean": float(words.mean()),
                "words_median": float(words.median()),
                "words_p95": float(words.quantile(0.95)),
            }
        )
    text_stats_df = pd.DataFrame(text_stats_rows)

    top_k = int(eda_cfg.get("top_k", 20))
    top_statutes = _value_counts_from_list_column(df["statutes"], top_k=top_k)
    top_provisions = _value_counts_from_list_column(df["provisions"], top_k=top_k)
    top_precedents = _value_counts_from_list_column(df["precedents"], top_k=top_k)
    top_courts = court_dist.head(top_k)

    leakage_true_pct = float(df["ml_leakage_flag"].astype(bool).mean() * 100.0)
    decision_text_nonempty_pct = float(
        df["decision_text"].fillna("").map(lambda x: len(safe_text(x)) > 0).mean() * 100.0
    )

    # Save tables
    class_dist.to_csv(tables_dir / "class_distribution.csv", header=["count"])
    year_dist.to_csv(tables_dir / "cases_by_year.csv", header=["count"])
    court_dist.to_csv(tables_dir / "cases_by_court.csv", header=["count"])
    case_type_dist.to_csv(tables_dir / "cases_by_case_type.csv", header=["count"])
    missing_df.to_csv(tables_dir / "missingness.csv", index=False)
    text_stats_df.to_csv(tables_dir / "text_length_stats.csv", index=False)
    top_statutes.to_csv(tables_dir / "top_statutes.csv", header=["count"])
    top_provisions.to_csv(tables_dir / "top_provisions.csv", header=["count"])
    top_precedents.to_csv(tables_dir / "top_precedents.csv", header=["count"])
    top_courts.to_csv(tables_dir / "top_courts.csv", header=["count"])

    # Save figures
    plot_bar_series(
        class_dist,
        path=figures_dir / "class_distribution.png",
        title="Class Distribution",
        xlabel="Label",
        ylabel="Count",
        rotate_xticks=20,
    )
    plot_bar_series(
        top_courts,
        path=figures_dir / "top_courts.png",
        title="Top Courts",
        xlabel="Court",
        ylabel="Count",
    )
    plot_bar_series(
        year_dist,
        path=figures_dir / "cases_by_year.png",
        title="Cases by Year",
        xlabel="Year",
        ylabel="Count",
        rotate_xticks=0,
    )
    plot_histogram(
        df["ml_input_text"].fillna("").map(word_count),
        path=figures_dir / "ml_input_text_wordcount_hist.png",
        title="ml.input_text Word Count Distribution",
        xlabel="Word count",
        bins=int(eda_cfg.get("hist_bins", 30)),
    )
    plot_bar_series(
        top_statutes,
        path=figures_dir / "top_statutes.png",
        title="Top Statutes",
        xlabel="Statute",
        ylabel="Count",
    )
    plot_bar_series(
        top_provisions,
        path=figures_dir / "top_provisions.png",
        title="Top Provisions",
        xlabel="Provision",
        ylabel="Count",
    )
    plot_bar_series(
        top_precedents,
        path=figures_dir / "top_precedents.png",
        title="Top Precedents",
        xlabel="Precedent",
        ylabel="Count",
    )

    summary = {
        "n_cases": int(len(df)),
        "n_labeled_cases": int(len(labeled_df)),
        "leakage_true_pct": leakage_true_pct,
        "decision_text_nonempty_pct": decision_text_nonempty_pct,
        "table_files": sorted([str(p.relative_to(out_root)) for p in tables_dir.glob("*.csv")]),
        "figure_files": sorted([str(p.relative_to(out_root)) for p in figures_dir.glob("*.png")]),
    }

    report_text = build_eda_report(summary)
    write_eda_report(report_text, out_root / "eda" / "eda_report.md")
    save_json(summary, out_root / "eda" / "eda_summary.json")

    logger.info("EDA completed | tables=%d figures=%d", len(summary["table_files"]), len(summary["figure_files"]))
    return summary
