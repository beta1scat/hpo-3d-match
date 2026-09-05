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

# Setup Matplotlib style
def setup_plt_style():
    matplotlib.rcParams['font.family'] = ['SimSun', 'Times New Roman']
    matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.labelsize'] = 10.5
    matplotlib.rcParams['axes.titlesize'] = 11.5
    matplotlib.rcParams['xtick.labelsize'] = 9.5
    matplotlib.rcParams['ytick.labelsize'] = 9.5
    matplotlib.rcParams['legend.fontsize'] = 9.0
    matplotlib.rcParams['figure.titlesize'] = 12.0
    matplotlib.rcParams['lines.linewidth'] = 1.6
    matplotlib.rcParams['lines.markersize'] = 5.0
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['grid.alpha'] = 0.5
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.pad_inches'] = 0.05

def save_fig(fig: plt.Figure, name: str):
    """Save figure as vector PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    print(f"[+] Saved figure to {pdf_path} and {png_path}")
