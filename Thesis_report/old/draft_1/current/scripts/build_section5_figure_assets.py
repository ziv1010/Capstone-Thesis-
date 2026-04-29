from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-")
THESIS_FIG_DIR = ROOT / "Thesis_report" / "current" / "figures"
READABLE_DIR = ROOT / "GRAPH_VISUALISER" / "entity_analysis" / "outputs" / "figures_readable"


def copy_cross_domain_topology() -> None:
    src = READABLE_DIR / "cross_bucket_network_by_type.png"
    dst = THESIS_FIG_DIR / "cross_bucket_network_by_type_readable.png"
    shutil.copyfile(src, dst)
    print(f"copied {src.name} -> {dst}")


def build_bucket_fingerprint_composite() -> None:
    panel_paths = [
        READABLE_DIR / "within_family_matrimonial_timed_mistral_network.png",
        READABLE_DIR / "within_sexual_offences_timed_mistral_network.png",
        READABLE_DIR / "within_land_property_timed_mistral_network.png",
        READABLE_DIR / "within_motor_accidents_timed_mistral_network.png",
    ]
    output_path = THESIS_FIG_DIR / "within_bucket_network_fingerprints.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(panel_paths[0]),
        "-i", str(panel_paths[1]),
        "-i", str(panel_paths[2]),
        "-i", str(panel_paths[3]),
        "-filter_complex",
        (
            "[0:v]scale=1152:720[v0];"
            "[1:v]scale=1152:720[v1];"
            "[2:v]scale=1152:720[v2];"
            "[3:v]scale=1152:720[v3];"
            "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|1152_0|0_720|1152_720[out]"
        ),
        "-map", "[out]",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"wrote {output_path}")


def main() -> None:
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    copy_cross_domain_topology()
    build_bucket_fingerprint_composite()


if __name__ == "__main__":
    main()
