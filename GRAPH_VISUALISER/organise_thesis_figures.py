#!/usr/bin/env python3
"""
organise_thesis_figures.py — Collect all thesis-ready PNGs into one place.

Copies (not moves) the best figures into outputs/thesis_figures/ and
writes an INDEX.md listing each figure, its source, and its thesis relevance.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = {
    # --- from plots_full/ (raw dataset, no graph needed) ---
    "01_case_counts_full.png":              ROOT / "outputs/plots_full/case_counts_full.png",
    "02_outcome_distribution_full.png":     ROOT / "outputs/plots_full/outcome_distribution_full.png",
    "03_outcome_rate_full.png":             ROOT / "outputs/plots_full/outcome_rate_full.png",
    "04_entity_type_totals_full.png":       ROOT / "outputs/plots_full/entity_type_totals_full.png",
    "05_entity_per_case_full.png":          ROOT / "outputs/plots_full/entity_per_case_full.png",
    "06_sentence_count_distribution_full.png": ROOT / "outputs/plots_full/sentence_count_distribution_full.png",
    "07_rhetorical_role_breakdown_full.png":ROOT / "outputs/plots_full/rhetorical_role_breakdown_full.png",
    "08_labelling_coverage_full.png":       ROOT / "outputs/plots_full/labelling_coverage_full.png",

    # --- from outputs/plots/ (sample graph) ---
    "09_node_type_distribution.png":        ROOT / "outputs/plots/node_type_distribution.png",
    "10_case_distribution_by_bucket.png":   ROOT / "outputs/plots/case_distribution_by_bucket.png",
    "11_degree_distribution.png":           ROOT / "outputs/plots/degree_distribution.png",
    "12_top_hub_entities.png":              ROOT / "outputs/plots/top_hub_entities.png",
    "13_cross_bucket_bridges.png":          ROOT / "outputs/plots/cross_bucket_bridges.png",
    "14_connectivity_score_distributions.png": ROOT / "outputs/plots/connectivity_score_distributions.png",
    "15_entity_sharing_heatmap.png":        ROOT / "outputs/plots/entity_sharing_heatmap.png",
    "16_outcome_distribution_graph.png":    ROOT / "outputs/plots/outcome_distribution.png",

    # --- from outputs/graph_stats/ (full graph) ---
    "17_per_bucket_entity_composition.png": ROOT / "outputs/graph_stats/per_bucket_entity_composition.png",
    "18_case_degree_boxplot.png":           ROOT / "outputs/graph_stats/case_degree_boxplot.png",
    "19_degree_dist_by_type.png":           ROOT / "outputs/graph_stats/degree_dist_by_type.png",
    "20_cross_bucket_sharing_matrix.png":   ROOT / "outputs/graph_stats/cross_bucket_sharing_matrix.png",
    "21_bridge_entity_distribution.png":    ROOT / "outputs/graph_stats/bridge_entity_distribution.png",
    "22_top_hubs_by_degree_per_type.png":   ROOT / "outputs/graph_stats/top_hubs_by_degree_per_type.png",
    "23_entity_bucket_span.png":            ROOT / "outputs/graph_stats/entity_bucket_span.png",
    "24_top_pagerank_nodes.png":            ROOT / "outputs/graph_stats/top_pagerank_nodes.png",
    "25_top_pagerank_by_type.png":          ROOT / "outputs/graph_stats/top_pagerank_by_type.png",
    "26_pagerank_vs_degree.png":            ROOT / "outputs/graph_stats/pagerank_vs_degree.png",

    # --- from entity_analysis/outputs/figures_readable/ ---
    "27_cross_bucket_network_by_type_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/cross_bucket_network_by_type.png",
    "28_cross_bucket_network_by_bucket_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/cross_bucket_network_by_bucket.png",
    "29_cross_bucket_top_pagerank_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/cross_bucket_top_by_pagerank.png",
    "30_cross_bucket_top_bridge_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/cross_bucket_top_by_bridge.png",
    "31_within_family_network_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/within_family_matrimonial_network.png",
    "32_within_land_network_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/within_land_property_network.png",
    "33_within_motor_network_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/within_motor_accidents_network.png",
    "34_within_sexual_network_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/within_sexual_offences_network.png",
    "35_within_finfraud_network_readable.png":
        ROOT / "entity_analysis/outputs/figures_readable/within_fin_fraud_network.png",

    # --- from outputs/thesis_extras/ (new figures) ---
    "36_within_bucket_network_fingerprints.png":
        ROOT / "outputs/thesis_extras/within_bucket_network_fingerprints.png",
    "37_degree_ccdf_loglog.png":            ROOT / "outputs/thesis_extras/degree_ccdf_loglog.png",
    "38_per_bucket_richness_and_degree.png":ROOT / "outputs/thesis_extras/per_bucket_richness_and_degree.png",
    "39_connectivity_violin.png":           ROOT / "outputs/thesis_extras/connectivity_violin.png",
    "40_top_hubs_degree_and_bridging.png":  ROOT / "outputs/thesis_extras/top_hubs_degree_and_bridging.png",
    "41_shared_vs_local_nodes_per_bucket.png":
        ROOT / "outputs/thesis_extras/shared_vs_local_nodes_per_bucket.png",
    "42_pagerank_vs_degree_top30_labeled.png":
        ROOT / "outputs/thesis_extras/pagerank_vs_degree_top30_labeled.png",
    "43_outcome_skew_per_bucket.png":       ROOT / "outputs/thesis_extras/outcome_skew_per_bucket.png",
    "44_cross_domain_network_fingerprint.png":
        ROOT / "outputs/thesis_extras/cross_domain_network_fingerprint.png",
}

INDEX_ROWS = [
    # (dest_key, section, description, recommended)
    ("01_case_counts_full.png",              "§5.3 Per-domain",       "Case counts per domain (full dataset)", True),
    ("02_outcome_distribution_full.png",     "§5.3 Per-domain",       "Outcome distribution — full dataset (win/loss/unknown)", True),
    ("03_outcome_rate_full.png",             "§5.3 / §5.4",           "Win/loss rate (%) per domain — labelled only", True),
    ("04_entity_type_totals_full.png",       "§5.2 Node types",       "Entity mention counts by type and domain", True),
    ("05_entity_per_case_full.png",          "§5.3 Entity richness",  "Entity richness (mentions/case) per domain — boxplot", True),
    ("06_sentence_count_distribution_full.png","§5.3 Case length",    "Sentence count distribution per domain", True),
    ("07_rhetorical_role_breakdown_full.png","§5.3 Rhetoric",         "Rhetorical role composition per domain — stacked bar", True),
    ("08_labelling_coverage_full.png",       "§5.3",                  "Outcome labelling coverage per domain", False),
    ("09_node_type_distribution.png",        "§5.2 Node types",       "Node-type composition bar chart (sample graph)", True),
    ("10_case_distribution_by_bucket.png",   "§5.2",                  "Cases per domain (sample graph)", False),
    ("11_degree_distribution.png",           "§5.2 Degree",           "Degree distribution log-log — all nodes vs cases", True),
    ("12_top_hub_entities.png",              "§5.2 Hubs",             "Top shared-entity hubs by degree (sample)", True),
    ("13_cross_bucket_bridges.png",          "§5.2 Cross-domain",     "Cross-domain bridge entities (sample)", True),
    ("14_connectivity_score_distributions.png","§5.2 Connectivity",   "Connectivity score boxplot per domain", True),
    ("15_entity_sharing_heatmap.png",        "§5.2 Sharing",          "Entity sharing heatmap across domains", True),
    ("16_outcome_distribution_graph.png",    "§5.3",                  "Outcome distribution in graph subset", False),
    ("17_per_bucket_entity_composition.png", "§5.2 / §5.3",           "Unique shared-entity nodes per type and domain (FULL graph)", True),
    ("18_case_degree_boxplot.png",           "§5.2 Degree",           "Case degree distribution per domain — boxplot (full graph)", True),
    ("19_degree_dist_by_type.png",           "§5.2 Degree",           "Degree distribution per node type 3×3 grid (full graph)", True),
    ("20_cross_bucket_sharing_matrix.png",   "§5.2 Cross-domain",     "Cross-bucket entity sharing heatmap (full graph)", True),
    ("21_bridge_entity_distribution.png",    "§5.2 Cross-domain",     "Bridge entities by number of domains spanned (full graph)", True),
    ("22_top_hubs_by_degree_per_type.png",   "§5.2 Hubs",             "Top-15 hubs per entity type — 2×3 grid (full graph)", True),
    ("23_entity_bucket_span.png",            "§5.2 Cross-domain",     "Entities stacked by domain-span count per type", False),
    ("24_top_pagerank_nodes.png",            "§5.2 PageRank",         "Top-20 nodes by PageRank (all types)", True),
    ("25_top_pagerank_by_type.png",          "§5.2 PageRank",         "Top-10 PageRank per entity type — 2×3 grid", True),
    ("26_pagerank_vs_degree.png",            "§5.2 PageRank",         "PageRank vs. degree scatter by entity type (full graph)", True),
    ("27_cross_bucket_network_by_type_readable.png","§5.2 Cross-domain","Cross-domain entity network coloured by type (readable)", True),
    ("28_cross_bucket_network_by_bucket_readable.png","§5.2",          "Cross-domain entity network coloured by source domain", False),
    ("29_cross_bucket_top_pagerank_readable.png","§5.2 PageRank",     "Top cross-domain entities by PageRank (bar, readable)", True),
    ("30_cross_bucket_top_bridge_readable.png","§5.2 Cross-domain",   "Top bridge entities by domain-span score (readable)", True),
    ("31_within_family_network_readable.png","§5.3 Per-domain",       "Family/matrimonial entity co-occurrence network (top-25)", True),
    ("32_within_land_network_readable.png",  "§5.3 Per-domain",       "Land/property entity co-occurrence network (top-25)", True),
    ("33_within_motor_network_readable.png", "§5.3 Per-domain",       "Motor accidents entity co-occurrence network (top-25)", True),
    ("34_within_sexual_network_readable.png","§5.3 Per-domain",       "Sexual offences entity co-occurrence network (top-25)", True),
    ("35_within_finfraud_network_readable.png","§5.3 Per-domain",     "Financial fraud entity co-occurrence network (top-25)", True),
    ("36_within_bucket_network_fingerprints.png","§5.3 Per-domain",   "2×2 composite fingerprints — 4 domains side-by-side (NEW)", True),
    ("37_degree_ccdf_loglog.png",            "§5.2 Degree",           "Degree CCDF log-log — power-law tail (NEW)", True),
    ("38_per_bucket_richness_and_degree.png","§5.3 Per-domain",       "Shared-entity richness + mean/median degree per domain (NEW)", True),
    ("39_connectivity_violin.png",           "§5.2 Connectivity",     "Connectivity score violin plot per domain (NEW — richer than box)", True),
    ("40_top_hubs_degree_and_bridging.png",  "§5.2 Hubs",             "Top-20 hubs: degree + domains bridged side-by-side (NEW)", True),
    ("41_shared_vs_local_nodes_per_bucket.png","§5.2 Node types",     "Shared-entity vs. local-party node ratio per domain (NEW)", True),
    ("42_pagerank_vs_degree_top30_labeled.png","§5.2 PageRank",       "PageRank vs. degree — top-30 labeled scatter (NEW)", True),
    ("43_outcome_skew_per_bucket.png",       "§5.4 Outcome",          "Outcome count + win-rate per domain from graph subset (NEW)", True),
    ("44_cross_domain_network_fingerprint.png","§5.2 Cross-domain",   "Cross-domain entity fingerprint with shared bridge nodes (NEW)", True),
]

def main():
    dst = ROOT / "outputs" / "thesis_figures"
    dst.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for dest_name, src_path in SRC.items():
        if src_path.exists():
            shutil.copy2(src_path, dst / dest_name)
            copied.append(dest_name)
        else:
            missing.append((dest_name, str(src_path)))

    # write INDEX.md
    lines = [
        "# Thesis Figures — index",
        "",
        "All figures are copies from their source directories.",
        "Files numbered 01–08 → raw dataset; 09–16 → sample graph; 17–26 → full graph;",
        "27–35 → entity co-occurrence networks; 36–44 → new extras.",
        "",
        f"**{len(copied)} figures copied. {len(missing)} missing.**",
        "",
        "| # | File | Section | Description | ★ Recommended |",
        "|---|------|---------|-------------|---------------|",
    ]
    for fname, section, desc, rec in INDEX_ROWS:
        star = "✓" if rec else ""
        exists = "✓" if (dst / fname).exists() else "✗ MISSING"
        lines.append(f"| {fname[:2]} | `{fname}` | {section} | {desc} | {star} |")

    if missing:
        lines += ["", "## Missing files", ""]
        for n, p in missing:
            lines.append(f"- `{n}` → `{p}`")

    (dst / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"\nCopied {len(copied)} figures → {dst}/")
    if missing:
        print(f"Missing ({len(missing)}):")
        for n, p in missing:
            print(f"  {n} ← {p}")
    print("INDEX.md written.")


if __name__ == "__main__":
    main()
