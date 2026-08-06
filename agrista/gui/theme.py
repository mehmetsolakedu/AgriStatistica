"""Agrista GUI tema sistemi — açık ve koyu QSS temaları."""

ACCENT = "#2E86AB"

LIGHT_QSS = f"""
QMainWindow {{ background: #f7f9fa; }}
QMenuBar {{ background: #ffffff; color: #1b1b1b; }}
QMenuBar::item:selected {{ background: {ACCENT}; color: white; }}
QMenu {{ background: #ffffff; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QTableView {{ background: #ffffff; gridline-color: #dfe3e8; }}
QTabWidget::pane {{ border: 1px solid #dfe3e8; }}
QPushButton {{ background: {ACCENT}; color: white; border: none;
              border-radius: 4px; padding: 6px 14px; }}
QStatusBar {{ background: #eef2f4; }}
"""

DARK_QSS = f"""
QMainWindow {{ background: #1e1e1e; }}
QMenuBar {{ background: #252526; color: #e0e0e0; }}
QMenuBar::item:selected {{ background: {ACCENT}; color: white; }}
QMenu {{ background: #2d2d30; color: #e0e0e0; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: #6f6f6f; }}
QTableView {{ background: #1e1e1e; color: #e0e0e0;
             gridline-color: #333333; }}
QTextEdit, QPlainTextEdit {{ background: #252526; color: #e0e0e0; }}
QTabWidget::pane {{ border: 1px solid #333333; }}
QPushButton {{ background: {ACCENT}; color: white; border: none;
              border-radius: 4px; padding: 6px 14px; }}
QStatusBar {{ background: #252526; color: #e0e0e0; }}
"""


def tema_qss(ad: str) -> str:
    """Tema adı → QSS metni."""
    temalar = {"açık": LIGHT_QSS, "koyu": DARK_QSS}
    if ad not in temalar:
        raise ValueError(f"Bilinmeyen tema: {ad}")
    return temalar[ad]
