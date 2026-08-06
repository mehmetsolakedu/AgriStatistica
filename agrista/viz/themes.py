"""
Agrista Tema Sistemi — bilimsel yayın kalitesinde grafik stilleri.
"""

THEMES = {
    "agrista": {
        "style": "whitegrid",
        "rc": {"axes.edgecolor": "#333333", "figure.dpi": 100,
               "savefig.dpi": 150},
        "palette": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
                    "#3B1F2B", "#44BBA4"],
        "dpi": 150,
    },
    "yayin": {
        "style": "white",
        "rc": {"font.family": "serif", "axes.grid": False,
               "axes.spines.top": False, "axes.spines.right": False,
               "figure.dpi": 150, "savefig.dpi": 300},
        "palette": ["#1B1B1B", "#5A5A5A", "#8C8C8C", "#B0B0B0",
                    "#2E86AB", "#C73E1D"],
        "dpi": 300,
    },
    "minimal": {
        "style": "ticks",
        "rc": {"axes.grid": False, "figure.facecolor": "white",
               "savefig.dpi": 150},
        "palette": ["#2E86AB", "#44BBA4", "#F18F01", "#A23B72"],
        "dpi": 150,
    },
    "karanlik": {
        "style": "darkgrid",
        "rc": {"figure.facecolor": "#121212", "axes.facecolor": "#1E1E1E",
               "text.color": "#E0E0E0", "axes.labelcolor": "#E0E0E0",
               "xtick.color": "#B0B0B0", "ytick.color": "#B0B0B0",
               "savefig.dpi": 150},
        "palette": ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"],
        "dpi": 150,
    },
}


def apply_theme(name: str) -> dict:
    """Tema adı doğrula ve içerik sözlüğünü döndür."""
    if name not in THEMES:
        raise ValueError(f"Bilinmeyen tema: {name}. "
                         f"Seçenekler: {sorted(THEMES)}")
    return THEMES[name]
