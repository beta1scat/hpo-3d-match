"""Plot Figure 3-1: Deterministic RANSAC tabletop plane removal effect comparison.

Generates the publication-quality Figure 3-1 for Chapter 3 of the PhD thesis:
  (a) Raw scene point cloud with tabletop inliers highlighted in coral red.
  (b) Filtered workpiece target point cloud with tabletop cleanly removed.

Output:
  - figures/chapter3/3_1_ransac_tabletop_filter.pdf (Vector PDF)
  - figures/chapter3/3_1_ransac_tabletop_filter.png (High-resolution 600 DPI PNG)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

# Ensure plotting_scripts directory is in Python search path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_config import FIGURES_DIR, save_fig, setup_plt_style

DEFAULT_INPUT_DIR = CURRENT_DIR.parent / "data" / "exported_ply" / "ransac_filtered"


def find_default_images(input_dir: Path) -> tuple[Path, Path]:
    """Locate the tabletop red-marked image and tabletop removed image."""
    red_cands = sorted(input_dir.glob("*table_red.png"))
    clean_cands = sorted(input_dir.glob("*table_removed.png"))

    if not red_cands:
        raise FileNotFoundError(f"Cannot find '*table_red.png' in {input_dir}")
    if not clean_cands:
        raise FileNotFoundError(f"Cannot find '*table_removed.png' in {input_dir}")

    # Prioritize scene 8 if available
    img_red = None
    for p in red_cands:
        if "scene_0008" in p.name:
            img_red = p
            break
    if img_red is None:
        img_red = red_cands[0]

    img_clean = None
    for p in clean_cands:
        if "scene_0008" in p.name:
            img_clean = p
            break
    if img_clean is None:
        img_clean = clean_cands[0]

    return img_red, img_clean


def crop_or_pad_to_equal_height(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure both images have the identical vertical height for visual symmetry."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)

    def pad_vertical(img: np.ndarray, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == th:
            return img
        pad_total = th - h
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        if img.ndim == 3:
            return np.pad(
                img,
                ((pad_top, pad_bot), (0, 0), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        return np.pad(
            img,
            ((pad_top, pad_bot), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    return pad_vertical(img1, target_h), pad_vertical(img2, target_h)


def generate_figure(
    img_red_path: Path,
    img_clean_path: Path,
    out_name: str = "3_1_ransac_tabletop_filter",
    dpi: int = 600,
    fontsize: float = 20.5,
) -> None:
    """Generate side-by-side comparison figure matching PhD thesis formatting.

    Note on typography and scaling:
      In the LaTeX thesis (chapters/chapter3.tex), this figure is inserted via:
        \\includegraphics[width=0.92\\textwidth]{chapter3/3_1_ransac_tabletop_filter.pdf}
      where \\textwidth = 15.6 cm, so the figure width on the printed page is 14.35 cm (5.65 in).
      Because the figure canvas is ~11.0 in wide, LaTeX scales it down by a factor of S ≈ 0.515.
      To ensure the subfigure caption font on the printed page matches standard thesis
      typography "比正文小一号" (五号, 10.5 pt, where body text is 小四 12 pt),
      the matplotlib font size is set to fontsize ≈ 10.5 pt / 0.515 ≈ 20.5 pt.
      The text is neatly broken into two balanced lines to prevent inter-subplot collision.
    """
    setup_plt_style()

    print(f"[*] Reading (a) Tabletop image: {img_red_path}")
    print(f"[*] Reading (b) Cleaned image:  {img_clean_path}")

    img_a = iio.imread(img_red_path)
    img_b = iio.imread(img_clean_path)

    # Normalize heights for clean alignment
    img_a, img_b = crop_or_pad_to_equal_height(img_a, img_b)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    # Subplot (a): Raw scene with tabletop plane highlighted
    axes[0].imshow(img_a)
    axes[0].axis("off")
    axes[0].set_title(
        "(a) 包含大面积承重平面的原始场景点云\n（红色为台面内点）",
        y=-0.26,
        fontsize=fontsize,
        fontfamily=["SimSun", "Times New Roman"],
        linespacing=1.25,
    )

    # Subplot (b): Clean workpiece point cloud after RANSAC separation
    axes[1].imshow(img_b)
    axes[1].axis("off")
    axes[1].set_title(
        "(b) 经确定性 RANSAC 平面分离后的\n工件目标纯净点云",
        y=-0.26,
        fontsize=fontsize,
        fontfamily=["SimSun", "Times New Roman"],
        linespacing=1.25,
    )

    plt.tight_layout(pad=0.8)

    # Save vector PDF and high-resolution PNG
    pdf_out = FIGURES_DIR / f"{out_name}.pdf"
    png_out = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(pdf_out, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_out, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"[+] Successfully generated Figure 3-1:")
    print(f"    - PDF: {pdf_out}")
    print(f"    - PNG: {png_out} ({dpi} DPI)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 3 Figure 3-1 (RANSAC Tabletop Removal)."
    )
    parser.add_argument(
        "--img-table",
        type=Path,
        default=None,
        help="Path to image with tabletop highlighted in red",
    )
    parser.add_argument(
        "--img-clean",
        type=Path,
        default=None,
        help="Path to image with tabletop removed",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing exported screenshot PNGs",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="3_1_ransac_tabletop_filter",
        help="Base output filename without extension",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI resolution for rasterized image",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=20.5,
        help="Matplotlib font size (default 20.5pt, matching 10.5pt Wuhao on thesis page)",
    )
    args = parser.parse_args()

    if args.img_table and args.img_clean:
        img_red = args.img_table
        img_clean = args.img_clean
    else:
        img_red, img_clean = find_default_images(args.input_dir)

    generate_figure(img_red, img_clean, out_name=args.out_name, dpi=args.dpi, fontsize=args.fontsize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
