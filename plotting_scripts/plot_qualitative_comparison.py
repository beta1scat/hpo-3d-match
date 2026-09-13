"""Plot qualitative comparison of 2D and 3D pose estimation results.

Compares Baseline (Default) vs. Oracle (HPO Optimized) across 5 uniformly spaced
unseen test scenes per object model.

Row 1: Default 2D Overlay
Row 2: Default 3D Raw Sensor Point Cloud
Row 3: Oracle 2D Overlay
Row 4: Oracle 3D Raw Sensor Point Cloud

Layout: 4 Rows x 5 Columns
Quality: >= 600 DPI vector PDF & high-resolution PNG
Output: figures/chapter3/ and visualizations/comparison/
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure plotting_scripts directory is in python search path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_config import FIGURES_DIR, setup_plt_style

# Model configurations
MODEL_CONFIGS = {
    "star": {
        "fig_name": "3_4_star_compare",
        "display_name": "Star",
        "scene_list_file": "scene_list_star.txt",
        "default_scenes": [1, 12, 154, 160, 170],
    },
    "screw_black": {
        "fig_name": "3_5_screw_compare",
        "display_name": "Screw_Black",
        "scene_list_file": "scene_list_screw_black.txt",
        "default_scenes": [294, 298, 305, 307, 310],
    },
    "bracket_planar": {
        "fig_name": "3_6_bracket_compare",
        "display_name": "Bracket_Planar",
        "scene_list_file": "scene_list_bracket_planar.txt",
        "default_scenes": [451, 458, 466, 472, 479],
    },
}

ROW_LABELS_ZH = [
    "默认参数 2D 投影叠加",
    "默认参数 3D 点云对齐",
    "优化参数 2D 投影叠加",
    "优化参数 3D 点云对齐",
]

ROW_LABELS_EN = [
    "Default 2D Overlay",
    "Default 3D Alignment",
    "Oracle 2D Overlay",
    "Oracle 3D Alignment",
]


def load_training_scenes(manifest_csv: Path, model_name: str) -> set[int]:
    """Load training/validation scene IDs to exclude from test comparison."""
    excluded = set()
    if not manifest_csv.exists():
        return excluded

    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_name") == model_name:
                try:
                    excluded.add(int(row["image_id"]))
                except (ValueError, KeyError):
                    pass
    return excluded


def load_all_scenes_from_list(list_path: Path) -> list[int]:
    """Load ordered scene IDs from a scene list file."""
    if not list_path.exists():
        raise FileNotFoundError(f"Scene list file not found: {list_path}")

    scenes = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.search(r"scene_(\d+)", line)
            if m:
                scenes.append(int(m.group(1)))
            else:
                try:
                    scenes.append(int(line))
                except ValueError:
                    continue
    return sorted(list(set(scenes)))


def select_evenly_spaced_scenes(
    all_scenes: Sequence[int],
    excluded_scenes: set[int],
    target_count: int = 5,
) -> list[int]:
    """Select target_count scenes at uniform intervals excluding training scenes.

    Formula: candidate[int(i * len(candidate) / target_count)]
    For 10 scenes, this picks indices 0, 2, 4, 6, 8 (e.g. 1, 3, 5, 7, 9).
    """
    candidates = [s for s in sorted(all_scenes) if s not in excluded_scenes]
    if len(candidates) < target_count:
        print(f"[!] Warning: Candidates count ({len(candidates)}) < target ({target_count}). Using all candidates.")
        return candidates

    L = len(candidates)
    selected = [candidates[int(i * L / target_count)] for i in range(target_count)]
    return selected


def _find_img(directory: Path, model_name: str, scene_id: int, suffix: str) -> Path:
    candidates = [
        directory / f"native_{model_name}_scene{scene_id:04d}_{suffix}.png",
        directory / f"seq_sub_native_{model_name}_scene{scene_id:04d}_{suffix}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def find_scene_images(
    vis_dir: Path,
    model_name: str,
    scene_id: int,
) -> Tuple[Path, Path, Path, Path]:
    """Locate the 4 required images for a given scene.

    Returns:
        (default_2d, default_3d, oracle_2d, oracle_3d)
    """
    default_dir = vis_dir / f"native_default_{model_name}"
    oracle_dir = vis_dir / f"native_{model_name}_oracle"

    def_2d = _find_img(default_dir, model_name, scene_id, "2d_overlay")
    def_3d = _find_img(default_dir, model_name, scene_id, "3d_raw_pointcloud")
    orc_2d = _find_img(oracle_dir, model_name, scene_id, "2d_overlay")
    orc_3d = _find_img(oracle_dir, model_name, scene_id, "3d_raw_pointcloud")

    missing = [p for p in (def_2d, def_3d, orc_2d, orc_3d) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing images for model '{model_name}' scene {scene_id}: {missing}"
        )

    return def_2d, def_3d, orc_2d, orc_3d


def plot_comparison_grid(
    model_name: str,
    scene_ids: Sequence[int],
    vis_dir: Path,
    out_dir: Path,
    dpi: int = 600,
    use_chinese: bool = True,
) -> Tuple[Path, Path]:
    """Compose and save a 4x5 comparison figure at high DPI."""
    cfg = MODEL_CONFIGS[model_name]
    row_labels = ROW_LABELS_ZH if use_chinese else ROW_LABELS_EN

    # 1. Pre-load images and determine aspect ratios
    images_grid = []
    for row_idx in range(4):
        images_grid.append([])

    for scene_id in scene_ids:
        def_2d, def_3d, orc_2d, orc_3d = find_scene_images(vis_dir, model_name, scene_id)
        images_grid[0].append(iio.imread(def_2d))
        images_grid[1].append(iio.imread(def_3d))
        images_grid[2].append(iio.imread(orc_2d))
        images_grid[3].append(iio.imread(orc_3d))

    # Calculate row height ratios based on the first column image dimensions
    h_2d, w_2d = images_grid[0][0].shape[:2]
    h_3d, w_3d = images_grid[1][0].shape[:2]

    ratio_2d = h_2d / w_2d
    ratio_3d = h_3d / w_3d
    height_ratios = [ratio_2d, ratio_3d, ratio_2d, ratio_3d]

    # 2. Setup figure layout
    setup_plt_style()
    num_cols = len(scene_ids)
    col_width_inch = 2.1
    fig_width = col_width_inch * num_cols + 2.0  # Add margin for left row labels
    fig_height = col_width_inch * sum(height_ratios) + 0.6

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = GridSpec(
        4,
        num_cols,
        figure=fig,
        height_ratios=height_ratios,
        left=0.18,
        right=0.98,
        top=0.93,
        bottom=0.03,
        wspace=0.04,
        hspace=0.04,
    )

    for row in range(4):
        for col in range(num_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(images_grid[row][col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Top Column Headers
            if row == 0:
                header_text = f"场景 {scene_ids[col]}" if use_chinese else f"Scene {scene_ids[col]}"
                ax.set_title(header_text, fontsize=21.0, fontweight="bold", pad=10)

            # Left Row Labels
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=20.0,
                    fontweight="bold",
                    color="#1F3864",
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{cfg['fig_name']}.pdf"
    png_path = out_dir / f"{cfg['fig_name']}.png"

    fig.savefig(pdf_path, format="pdf", dpi=dpi)
    fig.savefig(png_path, format="png", dpi=dpi)
    plt.close(fig)

    # Mirror to visualizations/comparison
    comp_dir = vis_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_png = comp_dir / f"{cfg['fig_name']}.png"
    comp_pdf = comp_dir / f"{cfg['fig_name']}.pdf"
    try:
        import shutil
        shutil.copyfile(png_path, comp_png)
        shutil.copyfile(pdf_path, comp_pdf)
    except Exception:
        pass

    print(f"[+] Successfully generated {model_name} comparison figure (DPI={dpi}):")
    print(f"    - PDF: {pdf_path}")
    print(f"    - PNG: {png_path}")
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot qualitative 2D/3D comparison figures for thesis.")
    parser.add_argument(
        "--model",
        choices=["star", "screw_black", "bracket_planar"],
        help="Target object model name",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate comparison figures for all 3 models",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        help="Explicit comma-separated scene IDs (overrides automatic spacing)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI quality (default: 600)",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=CURRENT_DIR.parent / "visualizations",
        help="Path to visualizations directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Output directory (default: figures/chapter3/)",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="Label language (default: zh)",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        print("[!] Error: Specify --model <name> or --all.")
        return 1

    models = list(MODEL_CONFIGS.keys()) if args.all else [args.model]
    base_dir = CURRENT_DIR.parent
    manifest_csv = base_dir / "data" / "manifests" / "itodd_manual_annotated" / "bop_manifest.csv"
    scene_lists_dir = base_dir / "data" / "base_package" / "models" / "scene_lists"

    print("==================================================")
    print(f"Generating 2D/3D Qualitative Comparison Figures (DPI={args.dpi})")
    print(f"Output Directory: {args.out_dir.resolve()}")
    print("==================================================")

    for model in models:
        cfg = MODEL_CONFIGS[model]
        print(f"\n[*] Processing model: {model} ({cfg['display_name']})")

        if args.scenes:
            selected_scenes = [int(s.strip()) for s in args.scenes.split(",") if s.strip()]
            print(f"    - Using explicitly specified scenes: {selected_scenes}")
        else:
            selected_scenes = cfg["default_scenes"]
            print(f"    - Using fixed candidate list ({len(selected_scenes)} scenes): {selected_scenes}")

        plot_comparison_grid(
            model_name=model,
            scene_ids=selected_scenes,
            vis_dir=args.vis_dir,
            out_dir=args.out_dir,
            dpi=args.dpi,
            use_chinese=(args.lang == "zh"),
        )

    print("\n==================================================")
    print("All qualitative comparison figures generated successfully!")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
