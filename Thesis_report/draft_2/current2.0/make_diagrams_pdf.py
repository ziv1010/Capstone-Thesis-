r"""Generate dependency-free PDF diagrams for thesis_presentation.tex.

The execution environment used for editing this deck does not have beamer,
matplotlib, graphviz, or pillow installed.  This script writes simple vector
PDFs directly with the Python standard library so the generated images can be
included by pdflatex through \includegraphics.
"""

from __future__ import annotations

import math
from pathlib import Path


OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#113A5C"
GOLD = "#C78B2B"
INK = "#21313F"
SOFT = "#EEF3F7"
ACCENT = "#DCE8F2"
PALE_GOLD = "#FFF3D6"
PALE_BLUE = "#F4F8FB"
GREY = "#7A8A99"
LIGHT_GREY = "#D5DEE6"
WHITE = "#FFFFFF"


def _rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_width(text: str, size: float) -> float:
    # Helvetica approximation, good enough for centered diagram labels.
    return len(text) * size * 0.48


def _wrap(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= max_chars:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


class PdfCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = INK,
        lw: float = 1.5,
    ) -> None:
        fr, fg, fb = _rgb(fill)
        sr, sg, sb = _rgb(stroke)
        self.ops.append(
            f"q {fr:.4f} {fg:.4f} {fb:.4f} rg {sr:.4f} {sg:.4f} {sb:.4f} RG "
            f"{lw:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re B Q"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = INK,
        lw: float = 1.2,
        dashed: bool = False,
    ) -> None:
        r, g, b = _rgb(color)
        dash = "[8 6] 0 d" if dashed else "[] 0 d"
        self.ops.append(
            f"q {r:.4f} {g:.4f} {b:.4f} RG {lw:.2f} w {dash} "
            f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S Q"
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = INK,
        lw: float = 1.4,
        size: float = 10,
        dashed: bool = False,
    ) -> None:
        self.line(x1, y1, x2, y2, color=color, lw=lw, dashed=dashed)
        angle = math.atan2(y2 - y1, x2 - x1)
        left = angle + math.radians(154)
        right = angle - math.radians(154)
        p1 = (x2, y2)
        p2 = (x2 + size * math.cos(left), y2 + size * math.sin(left))
        p3 = (x2 + size * math.cos(right), y2 + size * math.sin(right))
        r, g, b = _rgb(color)
        self.ops.append(
            f"q {r:.4f} {g:.4f} {b:.4f} rg "
            f"{p1[0]:.2f} {p1[1]:.2f} m {p2[0]:.2f} {p2[1]:.2f} l "
            f"{p3[0]:.2f} {p3[1]:.2f} l h f Q"
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 12,
        color: str = INK,
        bold: bool = False,
        center: bool = True,
    ) -> None:
        r, g, b = _rgb(color)
        font = "F2" if bold else "F1"
        tx = x - _text_width(text, size) / 2 if center else x
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {r:.4f} {g:.4f} {b:.4f} rg "
            f"{tx:.2f} {y:.2f} Td ({_pdf_text(text)}) Tj ET"
        )

    def multiline(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 12,
        color: str = INK,
        bold: bool = False,
        center: bool = True,
        leading: float | None = None,
        max_chars: int | None = None,
    ) -> None:
        lines = _wrap(text, max_chars) if max_chars else text.split("\n")
        leading = leading or size * 1.22
        start = y + (len(lines) - 1) * leading / 2
        for idx, line in enumerate(lines):
            self.text(x, start - idx * leading, line, size=size, color=color, bold=bold, center=center)

    def box(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        label: str,
        fill: str,
        stroke: str,
        size: float = 12,
        color: str = INK,
        bold: bool = False,
        max_chars: int | None = None,
        lw: float = 1.5,
    ) -> None:
        self.rect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, lw=lw)
        self.multiline(cx, cy - size * 0.34, label, size=size, color=color, bold=bold, max_chars=max_chars)

    def save(self, path: Path) -> None:
        content = "\n".join(self.ops).encode("latin-1")
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("latin-1")
        )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objects.append(b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream")

        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(data))
            data.extend(f"{i} 0 obj\n".encode("ascii"))
            data.extend(obj)
            data.extend(b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        data.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            data.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        data.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(data)


def draw_pipeline() -> Path:
    c = PdfCanvas(1600, 900)
    c.text(800, 850, "End-to-End Data and Graph Construction Pipeline", 28, BLUE, bold=True)
    c.text(800, 815, "Each stage writes an auditable artefact; that artefact is the input to the next stage.", 16, INK)

    stages = [
        ("1. Document Collection", "Raw PDFs and judgment text\nbucketed by legal domain"),
        ("2. OpenNyAI Enrichment", "NER, rhetorical roles,\nand structured summaries"),
        ("3. Mistral Outcome Labelling", "win / loss / procedural\nwith confidence and rationale"),
        ("4. Multi-Hearing Merge", "one dispute mapped to\none final labelled case"),
        ("5. Leakage-Safe Cleaning", "decision text, RPC/RLC roles,\nand outcome phrases removed"),
        ("6. Graph Build + Embeddings", "PyG HeteroData graph cache\nplus bge-m3 node features"),
    ]
    xs_top = [250, 800, 1350]
    xs_bottom = [1350, 800, 250]
    y_stage_top, y_out_top = 700, 565
    y_stage_bot, y_out_bot = 360, 225
    sw, sh = 320, 78
    ow, oh = 320, 86

    c.rect(40, 105, 1520, 690, WHITE, LIGHT_GREY, lw=1.0)
    c.box(145, 775, 160, 36, "STAGE", ACCENT, BLUE, 14, BLUE, True)
    c.box(350, 775, 220, 36, "CACHED OUTPUT", PALE_GOLD, GOLD, 14, GOLD, True)

    for i, x in enumerate(xs_top):
        stage, output = stages[i]
        c.box(x, y_stage_top, sw, sh, stage, ACCENT, BLUE, 17, BLUE, True, max_chars=24)
        c.arrow(x, y_stage_top - sh / 2 - 8, x, y_out_top + oh / 2 + 8, GOLD, lw=3, size=13)
        c.box(x, y_out_top, ow, oh, output, PALE_GOLD, GOLD, 14, INK, max_chars=30)
    c.arrow(xs_top[0] + ow / 2 + 15, y_out_top, xs_top[1] - sw / 2 - 15, y_stage_top, BLUE, lw=2.4, size=13)
    c.arrow(xs_top[1] + ow / 2 + 15, y_out_top, xs_top[2] - sw / 2 - 15, y_stage_top, BLUE, lw=2.4, size=13)
    c.arrow(xs_top[2], y_out_top - oh / 2 - 18, xs_bottom[0], y_stage_bot + sh / 2 + 18, BLUE, lw=2.4, size=13)

    for i, x in enumerate(xs_bottom):
        stage, output = stages[3 + i]
        c.box(x, y_stage_bot, sw, sh, stage, ACCENT, BLUE, 17, BLUE, True, max_chars=24)
        c.arrow(x, y_stage_bot - sh / 2 - 8, x, y_out_bot + oh / 2 + 8, GOLD, lw=3, size=13)
        c.box(x, y_out_bot, ow, oh, output, PALE_GOLD, GOLD, 14, INK, max_chars=31)
    c.arrow(xs_bottom[0] - ow / 2 - 15, y_out_bot, xs_bottom[1] + sw / 2 + 15, y_stage_bot, BLUE, lw=2.4, size=13)
    c.arrow(xs_bottom[1] - ow / 2 - 15, y_out_bot, xs_bottom[2] + sw / 2 + 15, y_stage_bot, BLUE, lw=2.4, size=13)

    c.text(800, 65, "Format: stage -> output artefact -> next stage", 18, BLUE, bold=True)
    out = OUT / "pipeline_flow_v3.pdf"
    c.save(out)
    return out


def draw_single_case() -> Path:
    c = PdfCanvas(1600, 1050)
    c.text(800, 1000, "Single-Case Reasoning Graph: 17 Node Types and Schema Edges", 27, BLUE, bold=True)
    c.text(800, 966, "One case anchor connects local text, local identity entities, and shared legal authorities.", 15, INK)

    # Node positions.
    pos = {
        "case": (780, 535),
        "preamble": (285, 810),
        "facts": (285, 690),
        "arguments": (285, 535),
        "petitioner_arguments": (285, 380),
        "respondent_arguments": (285, 260),
        "other_lawyer_arguments": (285, 140),
        "petitioner": (665, 180),
        "respondent": (895, 180),
        "petitioner_lawyer": (555, 320),
        "defence_lawyer": (1005, 320),
        "lawyer": (780, 75),
        "court": (620, 805),
        "judge": (940, 805),
        "provision": (1185, 720),
        "statute": (1360, 560),
        "precedent": (1185, 400),
    }
    labels = {
        "case": "CASE",
        "preamble": "Preamble",
        "facts": "Facts",
        "arguments": "Arguments",
        "petitioner_arguments": "Petitioner\nArguments",
        "respondent_arguments": "Respondent\nArguments",
        "other_lawyer_arguments": "Other Lawyer\nArguments",
        "petitioner": "Petitioner",
        "respondent": "Respondent",
        "petitioner_lawyer": "Petitioner\nLawyer",
        "defence_lawyer": "Defence\nLawyer",
        "lawyer": "Lawyer",
        "court": "Court",
        "judge": "Judge",
        "provision": "Provision",
        "statute": "Statute",
        "precedent": "Precedent",
    }
    text_nodes = {"preamble", "facts", "arguments", "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments"}
    shared_nodes = {"provision", "statute", "precedent"}

    c.rect(90, 92, 390, 770, "#FFFBF4", GOLD, lw=1.0)
    c.text(285, 875, "Text and argument nodes", 16, GOLD, bold=True)
    c.rect(505, 45, 545, 340, PALE_BLUE, BLUE, lw=1.0)
    c.text(778, 405, "Local identity entities", 16, BLUE, bold=True)
    c.rect(1080, 335, 385, 450, "#FFFBF4", GOLD, lw=1.0)
    c.text(1272, 805, "Shared authority nodes", 16, GOLD, bold=True)

    def edge(src: str, dst: str, color: str, dashed: bool = False, lw: float = 1.35) -> None:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length
        src_trim = 66 if src == "case" else 82
        dst_trim = 82 if dst != "case" else 66
        c.arrow(
            x1 + ux * src_trim,
            y1 + uy * src_trim,
            x2 - ux * dst_trim,
            y2 - uy * dst_trim,
            color=color,
            lw=lw,
            size=10,
            dashed=dashed,
        )

    # Case incidence edges: all case -> text/entity links.
    for dst in [
        "preamble",
        "facts",
        "arguments",
        "petitioner_arguments",
        "respondent_arguments",
        "other_lawyer_arguments",
        "petitioner",
        "respondent",
        "petitioner_lawyer",
        "defence_lawyer",
        "lawyer",
        "court",
        "judge",
    ]:
        edge("case", dst, BLUE, lw=1.25)

    # Argument ownership edges.
    for src, dst in [
        ("petitioner", "petitioner_arguments"),
        ("petitioner_lawyer", "petitioner_arguments"),
        ("respondent", "respondent_arguments"),
        ("defence_lawyer", "respondent_arguments"),
        ("lawyer", "other_lawyer_arguments"),
    ]:
        edge(src, dst, GREY, lw=1.15)

    # Citation edges from argument nodes to legal authorities.
    for src in ["arguments", "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments"]:
        for dst in ["provision", "statute", "precedent"]:
            edge(src, dst, GOLD, dashed=(src != "arguments"), lw=1.05)
    edge("provision", "statute", GOLD, lw=1.5)

    # Draw nodes after edges.
    for name, (x, y) in pos.items():
        if name == "case":
            c.box(x, y, 132, 70, labels[name], BLUE, "#0A2236", 18, WHITE, True)
        elif name in text_nodes:
            c.box(x, y, 210, 60, labels[name], PALE_GOLD, GOLD, 13, INK, True, max_chars=18)
        elif name in shared_nodes:
            c.box(x, y, 160, 62, labels[name], "#FFEAA7", GOLD, 14, INK, True)
        else:
            c.box(x, y, 158, 58, labels[name], ACCENT, BLUE, 13, INK, max_chars=18)

    # Relation labels and legend.
    c.box(128, 35, 150, 34, "case edges", ACCENT, BLUE, 11, BLUE, True)
    c.box(295, 35, 160, 34, "authority cites", "#FFEAA7", GOLD, 11, GOLD, True)
    c.box(475, 35, 170, 34, "ownership links", "#F1F3F5", GREY, 11, GREY, True)
    c.text(1160, 220, "Dashed gold: role-specific citation edges", 12, GOLD, bold=True, center=False)
    c.text(1160, 190, "Provision -> Statute preserves legal hierarchy", 12, GOLD, bold=True, center=False)
    out = OUT / "single_case_graph_v3.pdf"
    c.save(out)
    return out


def draw_cross_case() -> Path:
    c = PdfCanvas(1600, 900)
    c.text(800, 850, "Cross-Case Sharing and Leakage Control", 28, BLUE, bold=True)
    c.text(800, 815, "Only legal authorities are shared globally; identities stay inside each case cluster.", 16, INK)

    c.rect(65, 610, 1470, 145, "#FFF8E8", GOLD, lw=1.8)
    c.text(155, 720, "SHARED LEGAL AUTHORITY LAYER", 14, GOLD, bold=True, center=False)
    shared = [
        ("Statute\nIPC", 330, 665),
        ("Provision\nSection 482", 600, 665),
        ("Precedent\nSC case", 875, 665),
        ("Statute\nCrPC", 1145, 665),
        ("Provision\nSection 166", 1390, 665),
    ]
    for label, x, y in shared:
        c.box(x, y, 170, 64, label, "#FFEAA7", GOLD, 13, INK, True, max_chars=16)

    cluster_centres = [(310, 390), (800, 390), (1290, 390)]
    for idx, (cx, cy) in enumerate(cluster_centres, start=1):
        c.rect(cx - 200, cy - 150, 400, 260, SOFT, "#9CAAB6", lw=1.2)
        c.text(cx - 178, cy + 84, f"CASE {idx} LOCAL CLUSTER", 12, GREY, bold=True, center=False)
        c.box(cx, cy + 10, 128, 58, f"Case {idx}", BLUE, "#0A2236", 15, WHITE, True)
        locals_ = [
            ("Court", cx - 122, cy + 56),
            ("Judge", cx + 122, cy + 56),
            ("Petitioner", cx - 120, cy - 60),
            ("Respondent", cx + 120, cy - 60),
            ("Lawyers", cx, cy - 105),
        ]
        for label, x, y in locals_:
            c.box(x, y, 112, 42, label, ACCENT, BLUE, 10.5, INK)
            c.line(cx, cy + 10, x, y, BLUE, lw=0.8)

    # Shared authority links.
    links = [
        (cluster_centres[0], shared[0]), (cluster_centres[0], shared[1]), (cluster_centres[0], shared[2]),
        (cluster_centres[1], shared[0]), (cluster_centres[1], shared[2]), (cluster_centres[1], shared[3]),
        (cluster_centres[2], shared[1]), (cluster_centres[2], shared[2]), (cluster_centres[2], shared[4]),
    ]
    for (cx, cy), (_, sx, sy) in links:
        c.arrow(cx, cy + 140, sx, sy - 42, GOLD, lw=1.55, size=10)

    c.box(800, 555, 520, 44, "Cross-case messages pass through statutes, provisions, and precedents only", WHITE, WHITE, 13, BLUE, True)

    # Leakage gate strip.
    c.rect(65, 55, 1470, 105, ACCENT, BLUE, lw=1.6)
    c.text(110, 126, "LEAKAGE GATES BEFORE GRAPH BUILD", 14, BLUE, bold=True, center=False)
    gates = [
        "drop decision fields",
        "remove RPC/RLC sentences",
        "mask outcome phrases",
        "temporal authority gating",
        "keep identity nodes local",
    ]
    x = 250
    for gate in gates:
        c.box(x, 88, 230, 42, gate, WHITE, BLUE, 11, INK, max_chars=24)
        x += 270
    c.text(95, 190, "Not shared across cases: parties, lawyers, judges, courts, orgs, GPEs, dates, case numbers", 13, GREY, bold=True, center=False)

    out = OUT / "cross_case_v3.pdf"
    c.save(out)
    return out


if __name__ == "__main__":
    for path in (draw_pipeline(), draw_single_case(), draw_cross_case()):
        print(path)
