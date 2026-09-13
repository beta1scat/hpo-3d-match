"""Common plotting configuration and styling for Chapter 3 figures.

Follows Chinese PhD thesis typography:
  - Chinese font: SimSun (宋体)
  - English/Math font: Times New Roman / STIX
  - Academic Palette: Thesis blue-gray theme
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# Output directory for chapter 3 figures
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures" / "chapter3"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Results root directory
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Academic Color Palette (strictly matching main.tex definitions)
THEME_BLUE = "#2F5597"       # themeblue
THEME_BLUE_DARK = "#1F3864"  # themebluedark
THEME_BLUE_LIGHT = "#B4C7E7" # themebluelight
THEME_GRAY = "#595959"       # themegray
THEME_GRAY_LIGHT = "#D9D9D9" # themegraylight
THEME_GREEN = "#548235"      # themegreen
THEME_ORANGE = "#C68642"     # themeorange
THEME_RED = "#C0504D"        # themered

# Setup Matplotlib style with thesis scale compensation
def setup_plt_style(scale: float = 2.0):
    """Configure matplotlib rcParams.

    Note on thesis scaling:
      In the LaTeX thesis, figures (typically figsize width 11~14 in) are inserted at width ≈ 0.9~1.0\\textwidth
      (\\textwidth = 15.6 cm ≈ 6.14 in), which scales the entire figure canvas down by a factor of S ≈ 0.45~0.52.
      To ensure typography on the printed page adheres to thesis standards:
        - Titles & Labels: 五号 (10.5 pt) -> matplotlib fontsize ≈ 10.5 * 2.0 = 21.0 pt
        - Ticks & Legends: 小五号 (9.0~9.5 pt) -> matplotlib fontsize ≈ 9.0~9.5 * 2.0 = 18~19 pt
      scale defaults to 2.0 to provide 1:1 perceptual match to body / caption typography.
    """
    matplotlib.rcParams['font.family'] = ['SimSun', 'Times New Roman']
    matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.labelsize'] = 10.0 * scale   # 20.0 pt -> ~10.0 pt on page
    matplotlib.rcParams['axes.titlesize'] = 10.5 * scale   # 21.0 pt -> ~10.5 pt on page (五号)
    matplotlib.rcParams['xtick.labelsize'] = 9.0 * scale   # 18.0 pt -> ~9.0 pt on page
    matplotlib.rcParams['ytick.labelsize'] = 9.0 * scale   # 18.0 pt -> ~9.0 pt on page
    matplotlib.rcParams['legend.fontsize'] = 9.0 * scale   # 18.0 pt -> ~9.0 pt on page
    matplotlib.rcParams['figure.titlesize'] = 11.5 * scale # 23.0 pt
    matplotlib.rcParams['lines.linewidth'] = 1.6 * 1.25    # 2.0 pt
    matplotlib.rcParams['lines.markersize'] = 5.0 * 1.25   # 6.25 pt
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['grid.alpha'] = 0.5
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.pad_inches'] = 0.04

def save_fig(fig: plt.Figure, name: str):
    """Save figure as vector PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    print(f"[+] Saved figure to {pdf_path} and {png_path}")
