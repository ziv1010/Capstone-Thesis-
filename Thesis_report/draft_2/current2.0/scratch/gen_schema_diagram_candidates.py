from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "candidates"

COLORS = {
    "case": "#DBEAFE",
    "case_edge": "#2563EB",
    "text": "#DCFCE7",
    "text_edge": "#15803D",
    "local": "#FCE7F3",
    "local_edge": "#BE185D",
    "shared": "#FEF3C7",
    "shared_edge": "#D97706",
    "line": "#334155",
    "muted": "#64748B",
    "purple": "#7C3AED",
    "bg": "#FFFFFF",
    "group": "#F8FAFC",
}


def setup_canvas(width=14, height=8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def group_box(ax, xy, width, height, label, edge="#CBD5E1", fill="#F8FAFC"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.2,
        linestyle=(0, (4, 4)),
        zorder=0,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.22,
        y + height - 0.28,
        label,
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
        fontweight="bold",
    )


def node(ax, x, y, label, kind, w=1.45, h=0.62, circle=False, fontsize=10.5):
    fill = COLORS[kind]
    edge = COLORS[f"{kind}_edge"] if f"{kind}_edge" in COLORS else COLORS["line"]
    if circle:
        patch = Circle(
            (x, y),
            radius=w / 2,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
            zorder=3,
        )
    else:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.16",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4,
            zorder=3,
        )
    patch.set_path_effects(
        [pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=0.13), pe.Normal()]
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        fontweight="bold" if kind == "case" else "normal",
        zorder=4,
        linespacing=1.05,
    )


def arrow(
    ax,
    start,
    end,
    label=None,
    color=None,
    rad=0.0,
    lw=1.8,
    text_shift=(0, 0),
    label_bg=None,
    label_color=None,
):
    color = color or COLORS["line"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=16,
        shrinkB=16,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2 + text_shift[0]
        my = (start[1] + end[1]) / 2 + text_shift[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=8.4,
            color=label_color or color,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=label_bg or "white",
                ec="none",
                alpha=0.96,
            ),
            zorder=5,
        )


def legend(ax, x, y):
    entries = [
        ("Case", "case"),
        ("Text span", "text"),
        ("Case-local identity", "local"),
        ("Shared legal entity", "shared"),
    ]
    for i, (label, kind) in enumerate(entries):
        yy = y - i * 0.35
        ax.add_patch(
            Circle(
                (x, yy),
                0.085,
                facecolor=COLORS[kind],
                edgecolor=COLORS.get(f"{kind}_edge", COLORS["line"]),
                linewidth=1.0,
                zorder=6,
            )
        )
        ax.text(x + 0.18, yy, label, va="center", ha="left", fontsize=8.5, color="#334155")


def draw_single_case():
    fig, ax = setup_canvas(14, 8)

    group_box(ax, (0.45, 1.45), 4.45, 5.8, "Text nodes: embedded legal discourse")
    group_box(ax, (9.2, 1.45), 4.35, 5.8, "Case-local identity nodes")
    group_box(ax, (4.55, 0.35), 5.1, 1.55, "Shared legal nodes: cross-case conduits", fill="#FFFBEB")

    case = (7.0, 4.45)
    node(ax, *case, "Case\nlabel + dates\nbucket metadata", "case", w=1.5, circle=True, fontsize=9.6)

    text_nodes = {
        "Preamble": (2.05, 6.45),
        "Facts": (1.8, 5.28),
        "Arguments": (2.12, 4.1),
        "Petitioner\nArguments": (3.1, 2.88),
        "Respondent\nArguments": (1.45, 2.32),
        "Other Lawyer\nArguments": (3.28, 1.74),
    }
    for label, pos in text_nodes.items():
        node(ax, *pos, label, "text", w=1.55, h=0.62, fontsize=9.2)
        if label in {"Preamble", "Facts", "Arguments"}:
            arrow(ax, case, pos, "has_*", color=COLORS["text_edge"], lw=1.55)
        else:
            arrow(ax, case, pos, "has_party_args", color=COLORS["text_edge"], lw=1.35)

    local_nodes = {
        "Court": (10.55, 6.45),
        "Judge": (12.12, 5.45),
        "Petitioner": (10.65, 4.18),
        "Respondent": (12.1, 3.2),
        "Counsel": (10.6, 2.1),
    }
    for label, pos in local_nodes.items():
        node(ax, *pos, label, "local", w=1.42, h=0.58, fontsize=9.4)
        arrow(ax, case, pos, "case-local", color=COLORS["local_edge"], lw=1.45)

    shared_nodes = {
        "Precedent": (5.15, 1.03),
        "Statute": (7.0, 1.03),
        "Provision": (8.85, 1.03),
    }
    for label, pos in shared_nodes.items():
        node(ax, *pos, label, "shared", w=1.38, h=0.58, fontsize=9.5)

    args = text_nodes["Arguments"]
    arrow(ax, args, shared_nodes["Precedent"], "cites_precedent", color=COLORS["shared_edge"], rad=-0.08, lw=1.55)
    arrow(ax, args, shared_nodes["Statute"], "cites_statute", color=COLORS["shared_edge"], rad=0.02, lw=1.55, text_shift=(0.25, 0.1))
    arrow(ax, args, shared_nodes["Provision"], "cites_provision", color=COLORS["shared_edge"], rad=0.09, lw=1.55)
    arrow(ax, shared_nodes["Provision"], shared_nodes["Statute"], "belongs_to_statute", color=COLORS["shared_edge"], lw=1.35, text_shift=(0, 0.34))

    arrow(ax, local_nodes["Counsel"], text_nodes["Petitioner\nArguments"], "argues_in", color=COLORS["local_edge"], rad=0.18, lw=1.25)
    arrow(ax, local_nodes["Petitioner"], text_nodes["Petitioner\nArguments"], "party_in", color=COLORS["local_edge"], rad=0.12, lw=1.2)
    arrow(ax, local_nodes["Respondent"], text_nodes["Respondent\nArguments"], "party_in", color=COLORS["local_edge"], rad=-0.12, lw=1.2)

    ax.text(
        4.7,
        0.45,
        "Temporal gate: precedent_year < case_year; statute_year <= case_year",
        fontsize=8.8,
        color="#92400E",
        ha="left",
        va="bottom",
    )
    legend(ax, 11.62, 0.98)
    ax.text(
        0.52,
        7.68,
        "Single-case star schema",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.52,
        7.32,
        "A case anchors text evidence and private identities; only legal citations leave the local star.",
        fontsize=9.8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    fig.savefig(OUT / "candidate_single_case_schema_v2.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_cross_case():
    fig, ax = setup_canvas(14, 8)

    group_box(ax, (0.45, 4.45), 5.15, 2.75, "Case A local star: private identities")
    group_box(ax, (0.45, 0.85), 5.15, 2.75, "Case B local star: private identities")
    group_box(ax, (7.05, 1.1), 6.35, 6.1, "Shared legal layer: reusable across cases", fill="#FFFBEB")

    case_a = (2.0, 5.88)
    args_a = (4.25, 5.88)
    court_a = (0.95, 6.65)
    judge_a = (0.95, 5.13)

    case_b = (2.0, 2.28)
    args_b = (4.25, 2.28)
    court_b = (0.95, 3.05)
    judge_b = (0.95, 1.55)

    shared = {
        "Statute\nIPC": (9.0, 4.95),
        "Provision\nSec. 420": (11.55, 4.95),
        "Precedent\nX v. Y, 2010": (9.0, 3.1),
    }

    for pos, label in [
        (case_a, "Case A\n2015"),
        (case_b, "Case B\n2022"),
    ]:
        node(ax, *pos, label, "case", w=1.15, circle=True, fontsize=9.4)

    for pos, label in [(args_a, "Arguments A"), (args_b, "Arguments B")]:
        node(ax, *pos, label, "text", w=1.52, h=0.62, fontsize=9.4)

    for pos, label in [
        (court_a, "Court A"),
        (judge_a, "Judge A"),
        (court_b, "Court B"),
        (judge_b, "Judge B"),
    ]:
        node(ax, *pos, label, "local", w=1.22, h=0.55, fontsize=9.0)

    for label, pos in shared.items():
        node(ax, *pos, label, "shared", w=1.65, h=0.7, fontsize=9.2)

    arrow(ax, case_a, args_a, "has_arguments", color=COLORS["text_edge"], lw=1.65)
    arrow(ax, case_b, args_b, "has_arguments", color=COLORS["text_edge"], lw=1.65)
    arrow(ax, case_a, court_a, "heard_in", color=COLORS["local_edge"], lw=1.25)
    arrow(ax, case_a, judge_a, "decided_by", color=COLORS["local_edge"], lw=1.25)
    arrow(ax, case_b, court_b, "heard_in", color=COLORS["local_edge"], lw=1.25)
    arrow(ax, case_b, judge_b, "decided_by", color=COLORS["local_edge"], lw=1.25)

    arrow(ax, args_a, shared["Statute\nIPC"], "cites", color=COLORS["shared_edge"], rad=-0.10, lw=1.75)
    arrow(ax, args_b, shared["Statute\nIPC"], "cites", color=COLORS["shared_edge"], rad=0.10, lw=1.75)
    arrow(ax, args_a, shared["Precedent\nX v. Y, 2010"], "cites", color=COLORS["shared_edge"], rad=-0.04, lw=1.55, text_shift=(-0.15, -0.1))
    arrow(ax, args_b, shared["Precedent\nX v. Y, 2010"], "cites", color=COLORS["shared_edge"], rad=0.05, lw=1.55, text_shift=(-0.1, 0.1))
    arrow(ax, shared["Provision\nSec. 420"], shared["Statute\nIPC"], "belongs_to_statute", color=COLORS["shared_edge"], rad=0.0, lw=1.35, text_shift=(0, 0.42))

    ax.plot([0.92, 0.92], [3.65, 4.38], color="#DC2626", linewidth=1.5, linestyle=(0, (4, 4)))
    ax.text(
        1.08,
        4.02,
        "Courts and judges are duplicated per case\n(no shared docket-bias shortcut)",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#991B1B",
    )

    arrow(
        ax,
        (args_a[0] + 0.42, args_a[1] - 0.55),
        (args_b[0] + 0.42, args_b[1] + 0.55),
        "message passing only through shared legal nodes",
        color=COLORS["purple"],
        rad=0.42,
        lw=1.7,
        text_shift=(-0.15, 0.0),
    )

    ax.text(
        7.28,
        1.42,
        "Temporal validity: Case A cannot link to future precedents; Case B may share older law.",
        fontsize=8.8,
        color="#92400E",
        ha="left",
        va="bottom",
    )
    legend(ax, 11.15, 0.72)
    ax.text(
        0.52,
        7.68,
        "Cross-case sharing policy",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.52,
        7.32,
        "Only statutes, provisions, and precedents are global; identity entities remain local.",
        fontsize=9.8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    fig.savefig(OUT / "candidate_cross_case_schema_v2.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_single_case_clean():
    fig, ax = setup_canvas(14, 8.4)

    ax.text(
        0.65,
        7.82,
        "Single-case heterogeneous star",
        fontsize=15.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.65,
        7.42,
        "The case node anchors text evidence and private identity entities; only legal citations connect outward to shared law.",
        fontsize=9.8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    group_box(ax, (0.65, 1.45), 4.2, 5.35, "Text nodes", fill="#F7FBF6")
    group_box(ax, (9.15, 1.45), 4.2, 5.35, "Private identity nodes", fill="#FFF7F5")
    group_box(ax, (4.6, 0.28), 4.9, 1.62, "Shared legal nodes", fill="#FFFBEB")

    case = (7.0, 4.35)
    node(ax, *case, "Case\nlabel, dates,\nbucket", "case", w=1.35, circle=True, fontsize=9.7)

    text_nodes = [
        ("Preamble", (2.35, 5.95)),
        ("Facts", (2.35, 4.8)),
        ("Arguments", (2.35, 3.65)),
        ("Party-specific\nArguments", (2.35, 2.5)),
    ]
    for label, pos in text_nodes:
        node(ax, *pos, label, "text", w=1.7, h=0.62, fontsize=9.4)
        arrow(ax, case, pos, color=COLORS["text_edge"], lw=1.45)

    local_nodes = [
        ("Court", (11.28, 5.95)),
        ("Judge", (11.28, 4.8)),
        ("Parties", (11.28, 3.65)),
        ("Counsel", (11.28, 2.5)),
    ]
    for label, pos in local_nodes:
        node(ax, *pos, label, "local", w=1.55, h=0.62, fontsize=9.4)
        arrow(ax, case, pos, color=COLORS["local_edge"], lw=1.35)

    precedent = (5.35, 1.02)
    statute = (7.0, 1.02)
    provision = (8.65, 1.02)
    for label, pos in [("Precedent", precedent), ("Statute", statute), ("Provision", provision)]:
        node(ax, *pos, label, "shared", w=1.42, h=0.58, fontsize=9.4)

    args = text_nodes[2][1]
    arrow(ax, args, precedent, "cites", color=COLORS["shared_edge"], rad=-0.06, lw=1.55, text_shift=(-0.08, 0.1))
    arrow(ax, args, statute, "cites", color=COLORS["shared_edge"], rad=0.02, lw=1.55, text_shift=(0.0, 0.14))
    arrow(ax, args, provision, "cites", color=COLORS["shared_edge"], rad=0.08, lw=1.55, text_shift=(0.18, 0.1))
    arrow(ax, provision, statute, "belongs to", color=COLORS["shared_edge"], lw=1.2, text_shift=(0, 0.28))

    ax.text(2.35, 1.68, "BGE-M3 embeddings", fontsize=8.9, color=COLORS["text_edge"], ha="center")
    ax.text(11.28, 1.68, "kept local to avoid docket shortcuts", fontsize=8.9, color=COLORS["local_edge"], ha="center")
    ax.text(
        7.0,
        0.35,
        "temporal gate: precedent_year < case_year; statute_year <= case_year",
        fontsize=8.3,
        color="#92400E",
        ha="center",
    )

    # Compact visual legend.
    ax.plot([10.25, 10.85], [7.48, 7.48], color=COLORS["text_edge"], lw=2)
    ax.text(10.95, 7.48, "case to text", fontsize=8.7, color="#334155", va="center")
    ax.plot([10.25, 10.85], [7.12, 7.12], color=COLORS["local_edge"], lw=2)
    ax.text(10.95, 7.12, "case to local identity", fontsize=8.7, color="#334155", va="center")
    ax.plot([10.25, 10.85], [6.76, 6.76], color=COLORS["shared_edge"], lw=2)
    ax.text(10.95, 6.76, "argument to shared law", fontsize=8.7, color="#334155", va="center")

    fig.savefig(
        OUT / "candidate_single_case_schema_clean.png",
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def draw_cross_case_clean():
    fig, ax = setup_canvas(14, 8.4)

    ax.text(
        0.65,
        7.82,
        "Cross-case sharing policy",
        fontsize=15.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.65,
        7.42,
        "Legal entities are global conduits for message passing; courts, judges, parties, and counsel remain private to each case.",
        fontsize=9.8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    group_box(ax, (0.65, 4.42), 5.25, 2.55, "Case A local subgraph", fill="#F8FAFC")
    group_box(ax, (0.65, 1.15), 5.25, 2.55, "Case B local subgraph", fill="#F8FAFC")
    group_box(ax, (7.15, 1.15), 6.15, 5.82, "Shared legal subgraph", fill="#FFFBEB")

    case_a = (2.25, 5.64)
    args_a = (4.45, 5.64)
    case_b = (2.25, 2.37)
    args_b = (4.45, 2.37)
    local_a = [("Court A", (1.15, 6.25)), ("Judge A", (1.15, 5.03))]
    local_b = [("Court B", (1.15, 2.98)), ("Judge B", (1.15, 1.76))]
    statute = (9.3, 4.72)
    provision = (11.55, 4.72)
    precedent = (9.3, 2.72)

    node(ax, *case_a, "Case A\n2015", "case", w=1.13, circle=True, fontsize=9.3)
    node(ax, *case_b, "Case B\n2022", "case", w=1.13, circle=True, fontsize=9.3)
    node(ax, *args_a, "Arguments A", "text", w=1.55, h=0.62, fontsize=9.4)
    node(ax, *args_b, "Arguments B", "text", w=1.55, h=0.62, fontsize=9.4)

    for label, pos in local_a + local_b:
        node(ax, *pos, label, "local", w=1.15, h=0.52, fontsize=8.9)
    for label, pos in [("Statute\nIPC", statute), ("Provision\nSec. 420", provision), ("Precedent\nX v. Y, 2010", precedent)]:
        node(ax, *pos, label, "shared", w=1.55, h=0.66, fontsize=9.2)

    for start, end in [(case_a, args_a), (case_b, args_b)]:
        arrow(ax, start, end, color=COLORS["text_edge"], lw=1.55)
    for start, end in [(case_a, local_a[0][1]), (case_a, local_a[1][1]), (case_b, local_b[0][1]), (case_b, local_b[1][1])]:
        arrow(ax, start, end, color=COLORS["local_edge"], lw=1.2)

    arrow(ax, args_a, statute, "cites", color=COLORS["shared_edge"], rad=-0.10, lw=1.65)
    arrow(ax, args_b, statute, "cites", color=COLORS["shared_edge"], rad=0.10, lw=1.65)
    arrow(ax, args_a, precedent, "cites", color=COLORS["shared_edge"], rad=-0.04, lw=1.45, text_shift=(-0.15, -0.08))
    arrow(ax, args_b, precedent, "cites", color=COLORS["shared_edge"], rad=0.04, lw=1.45, text_shift=(-0.15, 0.08))
    arrow(ax, provision, statute, "belongs to", color=COLORS["shared_edge"], lw=1.25, text_shift=(0, 0.32))

    ax.plot([1.15, 1.15], [3.78, 4.34], color="#DC2626", lw=1.6, linestyle=(0, (4, 4)))
    ax.text(
        1.36,
        4.06,
        "identity nodes are not merged across cases",
        fontsize=8.8,
        color="#991B1B",
        ha="left",
        va="center",
    )

    ax.plot([6.48, 6.48], [1.25, 6.85], color="#94A3B8", lw=1.2, linestyle=(0, (3, 5)))
    ax.text(
        6.62,
        6.48,
        "cross-case edges\nonly through law",
        fontsize=8.8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    ax.text(
        8.0,
        1.42,
        "temporal rule blocks future citations before graph construction",
        fontsize=8.6,
        color="#92400E",
        ha="left",
    )

    ax.plot([10.4, 10.95], [7.44, 7.44], color=COLORS["text_edge"], lw=2)
    ax.text(11.05, 7.44, "case to argument", fontsize=8.7, color="#334155", va="center")
    ax.plot([10.4, 10.95], [7.08, 7.08], color=COLORS["local_edge"], lw=2)
    ax.text(11.05, 7.08, "private identity", fontsize=8.7, color="#334155", va="center")
    ax.plot([10.4, 10.95], [6.72, 6.72], color=COLORS["shared_edge"], lw=2)
    ax.text(11.05, 6.72, "shared legal citation", fontsize=8.7, color="#334155", va="center")

    fig.savefig(
        OUT / "candidate_cross_case_schema_clean.png",
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def panel(ax, x, y, w, h, title, color, fill):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.25,
        linestyle=(0, (4, 5)),
        alpha=0.92,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.24,
        y + h - 0.26,
        title,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=color, lw=1.0, alpha=0.96),
        zorder=6,
    )


def draw_relation_key(ax, x, y, entries):
    row_gap = 0.32
    box_h = 0.42 + row_gap * len(entries)
    box_w = 3.05
    key_box = FancyBboxPatch(
        (x - 0.18, y - box_h + 0.06),
        box_w,
        box_h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        zorder=7,
    )
    key_box.set_path_effects(
        [pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.08), pe.Normal()]
    )
    ax.add_patch(key_box)
    for i, (label, color) in enumerate(entries):
        yy = y - 0.18 - i * row_gap
        ax.plot(
            [x, x + 0.46],
            [yy, yy],
            color=color,
            lw=3.0,
            solid_capstyle="round",
            zorder=8,
        )
        ax.text(
            x + 0.6,
            yy,
            label,
            ha="left",
            va="center",
            fontsize=8.4,
            color="#334155",
            zorder=8,
        )


def draw_single_case_v3():
    fig, ax = setup_canvas(14, 8.15)

    ax.text(
        0.62,
        7.78,
        "Single-case heterogeneous star",
        fontsize=15.2,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.62,
        7.4,
        "A case collects private text and identity evidence; shared legal nodes are the only outward conduits.",
        fontsize=9.5,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    draw_relation_key(
        ax,
        10.45,
        7.78,
        [
            ("case -> text", COLORS["text_edge"]),
            ("case -> private identity", COLORS["local_edge"]),
            ("arguments -> shared law", COLORS["shared_edge"]),
        ],
    )

    panel(ax, 0.55, 1.18, 4.25, 5.4, "TEXT EVIDENCE", COLORS["text_edge"], "#F0FDF4")
    panel(ax, 9.2, 1.18, 4.25, 5.4, "PRIVATE IDENTITY", COLORS["local_edge"], "#FDF2F8")
    panel(ax, 4.78, 0.22, 4.65, 1.58, "SHARED LEGAL LAYER", COLORS["shared_edge"], "#FFFBEB")

    case = (7.0, 4.0)
    node(ax, *case, "Case\nlabel + dates\nbucket", "case", w=1.32, circle=True, fontsize=9.3)

    text_nodes = [
        ("Preamble", (2.45, 5.65)),
        ("Facts", (2.45, 4.58)),
        ("Arguments", (2.45, 3.5)),
        ("Party Args", (2.45, 2.43)),
    ]
    for label, pos in text_nodes:
        node(ax, *pos, label, "text", w=1.54, h=0.56, fontsize=9.1)
        arrow(ax, case, pos, color=COLORS["text_edge"], lw=1.35)

    identity_nodes = [
        ("Court", (11.35, 5.65)),
        ("Judge", (11.35, 4.58)),
        ("Parties", (11.35, 3.5)),
        ("Counsel", (11.35, 2.43)),
    ]
    for label, pos in identity_nodes:
        node(ax, *pos, label, "local", w=1.54, h=0.56, fontsize=9.1)
        arrow(ax, case, pos, color=COLORS["local_edge"], lw=1.3)

    law_nodes = [
        ("Precedent", (5.62, 0.92)),
        ("Statute", (7.1, 0.92)),
        ("Provision", (8.58, 0.92)),
    ]
    for label, pos in law_nodes:
        node(ax, *pos, label, "shared", w=1.26, h=0.52, fontsize=8.9)

    args = text_nodes[2][1]
    precedent = law_nodes[0][1]
    statute = law_nodes[1][1]
    provision = law_nodes[2][1]
    arrow(
        ax,
        args,
        precedent,
        "cites",
        color=COLORS["shared_edge"],
        rad=-0.04,
        lw=1.65,
        text_shift=(-0.18, 0.06),
        label_bg="#FFF7ED",
        label_color=COLORS["shared_edge"],
    )
    arrow(
        ax,
        args,
        statute,
        "cites",
        color=COLORS["shared_edge"],
        rad=0.02,
        lw=1.65,
        text_shift=(-0.02, 0.08),
        label_bg="#FFF7ED",
        label_color=COLORS["shared_edge"],
    )
    arrow(
        ax,
        args,
        provision,
        "cites",
        color=COLORS["shared_edge"],
        rad=0.08,
        lw=1.65,
        text_shift=(0.18, 0.08),
        label_bg="#FFF7ED",
        label_color=COLORS["shared_edge"],
    )
    arrow(
        ax,
        provision,
        statute,
        "belongs to",
        color=COLORS["shared_edge"],
        lw=1.2,
        text_shift=(0, 0.26),
        label_bg="#FFF7ED",
        label_color=COLORS["shared_edge"],
    )

    ax.text(2.45, 1.46, "BGE-M3 embeddings", fontsize=8.6, color=COLORS["text_edge"], ha="center")
    ax.text(11.35, 1.46, "local only: no docket-bias shortcut", fontsize=8.6, color=COLORS["local_edge"], ha="center")
    ax.text(
        7.1,
        0.34,
        "temporal gate: precedent_year < case_year; statute_year <= case_year",
        fontsize=7.9,
        color="#92400E",
        ha="center",
    )

    fig.savefig(
        OUT / "candidate_single_case_schema_v3.png",
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def draw_cross_case_v3():
    fig, ax = setup_canvas(14, 8.15)

    ax.text(
        0.62,
        7.78,
        "Cross-case sharing policy",
        fontsize=15.2,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="top",
    )
    ax.text(
        0.62,
        7.4,
        "Identity entities stay inside each case; statutes, provisions, and precedents are shared across cases.",
        fontsize=9.5,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    draw_relation_key(
        ax,
        10.45,
        7.78,
        [
            ("case -> arguments", COLORS["text_edge"]),
            ("private identity", COLORS["local_edge"]),
            ("shared legal citation", COLORS["shared_edge"]),
        ],
    )

    panel(ax, 0.55, 4.05, 5.18, 2.55, "CASE A LOCAL STAR", COLORS["case_edge"], "#F8FAFC")
    panel(ax, 0.55, 1.03, 5.18, 2.55, "CASE B LOCAL STAR", COLORS["case_edge"], "#F8FAFC")
    panel(ax, 7.35, 1.03, 6.1, 5.57, "GLOBAL SHARED LAW", COLORS["shared_edge"], "#FFFBEB")

    ax.plot([6.45, 6.45], [1.1, 6.55], color="#94A3B8", lw=1.3, linestyle=(0, (3, 5)))
    ax.text(
        6.62,
        6.25,
        "cross-case passage\nonly through law",
        fontsize=8.7,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    case_a = (2.18, 5.25)
    args_a = (4.35, 5.25)
    case_b = (2.18, 2.23)
    args_b = (4.35, 2.23)

    node(ax, *case_a, "Case A\n2015", "case", w=1.05, circle=True, fontsize=9.0)
    node(ax, *args_a, "Arguments A", "text", w=1.48, h=0.58, fontsize=9.0)
    node(ax, 1.08, 5.9, "Court A", "local", w=1.14, h=0.48, fontsize=8.5)
    node(ax, 1.08, 4.62, "Judge A", "local", w=1.14, h=0.48, fontsize=8.5)

    node(ax, *case_b, "Case B\n2022", "case", w=1.05, circle=True, fontsize=9.0)
    node(ax, *args_b, "Arguments B", "text", w=1.48, h=0.58, fontsize=9.0)
    node(ax, 1.08, 2.88, "Court B", "local", w=1.14, h=0.48, fontsize=8.5)
    node(ax, 1.08, 1.6, "Judge B", "local", w=1.14, h=0.48, fontsize=8.5)

    statute = (9.55, 4.48)
    provision = (11.72, 4.48)
    precedent = (9.55, 2.72)
    node(ax, *statute, "Statute\nIPC", "shared", w=1.45, h=0.64, fontsize=8.9)
    node(ax, *provision, "Provision\nSec. 420", "shared", w=1.48, h=0.64, fontsize=8.9)
    node(ax, *precedent, "Precedent\nX v. Y, 2010", "shared", w=1.56, h=0.64, fontsize=8.7)

    arrow(ax, case_a, args_a, color=COLORS["text_edge"], lw=1.45)
    arrow(ax, case_b, args_b, color=COLORS["text_edge"], lw=1.45)
    for start, end in [
        (case_a, (1.08, 5.9)),
        (case_a, (1.08, 4.62)),
        (case_b, (1.08, 2.88)),
        (case_b, (1.08, 1.6)),
    ]:
        arrow(ax, start, end, color=COLORS["local_edge"], lw=1.15)

    for start, end, shift, rad in [
        (args_a, statute, (-0.04, 0.12), -0.08),
        (args_a, precedent, (-0.13, -0.05), 0.02),
        (args_b, statute, (-0.18, 0.03), 0.08),
        (args_b, precedent, (-0.04, 0.08), 0.03),
    ]:
        arrow(
            ax,
            start,
            end,
            "cites",
            color=COLORS["shared_edge"],
            rad=rad,
            lw=1.58,
            text_shift=shift,
            label_bg="#FFF7ED",
            label_color=COLORS["shared_edge"],
        )
    arrow(
        ax,
        provision,
        statute,
        "belongs to",
        color=COLORS["shared_edge"],
        lw=1.25,
        text_shift=(0, 0.3),
        label_bg="#FFF7ED",
        label_color=COLORS["shared_edge"],
    )

    ax.plot([1.04, 1.04], [3.72, 3.96], color="#DC2626", lw=2, linestyle=(0, (4, 4)))
    ax.text(
        1.28,
        3.84,
        "identity nodes are duplicated, not merged",
        fontsize=8.6,
        color="#991B1B",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.14", fc="#FEF2F2", ec="none", alpha=0.96),
    )
    ax.text(
        8.05,
        1.34,
        "temporal rule blocks future citations before graph construction",
        fontsize=8.3,
        color="#92400E",
        ha="left",
    )

    fig.savefig(
        OUT / "candidate_cross_case_schema_v3.png",
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_single_case_clean()
    draw_cross_case_clean()
    draw_single_case_v3()
    draw_cross_case_v3()
    print(f"Generated candidates in {OUT}")


if __name__ == "__main__":
    main()
