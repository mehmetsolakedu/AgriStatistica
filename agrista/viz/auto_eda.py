"""
Agrista Auto-EDA — değişken türüne göre akıllı grafik önerisi ve rapor.
"""

from __future__ import annotations

import os

import pandas as pd

from agrista.viz.plotter import AgristaPlotter


def infer_column_types(df: pd.DataFrame) -> dict:
    """Her sütun için tür çıkarımı: sayisal/kategorik/tarih/metin."""
    turler = {}
    for kol in df.columns:
        s = df[kol]
        if pd.api.types.is_datetime64_any_dtype(s):
            turler[kol] = "tarih"
        elif pd.api.types.is_numeric_dtype(s):
            turler[kol] = "sayisal"
        elif s.nunique(dropna=True) <= max(2, min(20, len(s) // 10)):
            turler[kol] = "kategorik"
        else:
            turler[kol] = "metin"
    return turler


def chart_suggestion(df: pd.DataFrame) -> list:
    """Deterministik grafik öneri listesi (tür kurallarına göre)."""
    if df.empty:
        raise ValueError("Öneri için boş olmayan veri gerekli")
    turler = infer_column_types(df)
    sayisal = [c for c, t in turler.items() if t == "sayisal"]
    kategorik = [c for c, t in turler.items() if t == "kategorik"]
    tarih = [c for c, t in turler.items() if t == "tarih"]

    oneriler = []

    def ekle(tip, kolonlar, gerekce):
        oneriler.append({"type": tip, "columns": list(kolonlar),
                         "reason": gerekce})

    for c in sayisal[:4]:
        ekle("histogram", [c], "Tek sayısal değişken dağılımı")
        ekle("qq_plot", [c], "Normallik değerlendirmesi")
    for i in range(min(len(sayisal), 3)):
        for j in range(i + 1, min(len(sayisal), 3)):
            ekle("scatter", [sayisal[i], sayisal[j]], "İki sayısal ilişki")
            ekle("hexbin", [sayisal[i], sayisal[j]], "Yoğunluk görünümü")
    for k in kategorik[:2]:
        for c in sayisal[:3]:
            ekle("violin_plot", [k, c], "Sayısal × kategorik dağılım")
    if tarih and sayisal:
        ekle("line_chart", [tarih[0], sayisal[0]], "Zaman serisi")
    if len(sayisal) >= 3:
        ekle("correlation_heatmap", sayisal[:6], "Korelasyon matrisi")
        ekle("pair_grid", sayisal[:4], "Çok değişkenli matris")
    for k in kategorik[:2]:
        ekle("bar_chart", [k], "Kategori frekansları")
    return oneriler


def _ciz(p: AgristaPlotter, df: pd.DataFrame, oneri: dict, yol: str):
    """Bir öneriyi çizip kaydeder."""
    tip, kol = oneri["type"], oneri["columns"]
    if tip == "histogram":
        fig = p.histogram(df[kol[0]].dropna(), title=f"{kol[0]} Dağılımı")
    elif tip == "qq_plot":
        fig = p.qq_plot(df[kol[0]].dropna(), title=f"{kol[0]} Q-Q")
    elif tip == "scatter":
        fig = p.scatter(df[kol[0]], df[kol[1]], title=f"{kol[0]} × {kol[1]}",
                        xlabel=kol[0], ylabel=kol[1])
    elif tip == "hexbin":
        fig = p.hexbin_plot(df[kol[0]], df[kol[1]],
                            title=f"{kol[0]} × {kol[1]} Yoğunluk")
    elif tip == "violin_plot":
        fig = p.violin_plot(df, x_col=kol[0], y_col=kol[1],
                            title=f"{kol[1]} ~ {kol[0]}")
    elif tip == "line_chart":
        fig = p.line_chart(list(df[kol[0]]), list(df[kol[1]]),
                           title="Zaman Serisi", xlabel=kol[0], ylabel=kol[1])
    elif tip == "correlation_heatmap":
        fig = p.correlation_heatmap(df[kol])
    elif tip == "pair_grid":
        fig = p.pair_grid(df, cols=kol)
    elif tip == "bar_chart":
        say = df[kol[0]].value_counts()
        fig = p.bar_chart([str(x) for x in say.index], list(say.values),
                          title=f"{kol[0]} Frekansları", xlabel=kol[0])
    else:
        return None
    p.save(yol, fig=fig)
    AgristaPlotter.close()
    return yol


def auto_eda(df: pd.DataFrame, output_dir: str) -> dict:
    """Tam keşif raporu: öneriler + PNG'ler + report.html."""
    if df.empty:
        raise ValueError("Auto-EDA için boş olmayan veri gerekli")
    os.makedirs(output_dir, exist_ok=True)
    oneriler = chart_suggestion(df)
    p = AgristaPlotter()
    yollar = []
    for i, oneri in enumerate(oneriler):
        yol = os.path.join(output_dir, f"{i + 1:02d}_{oneri['type']}.png")
        if _ciz(p, df, oneri, yol):
            yollar.append(yol)
    html = ["<!DOCTYPE html><html lang=\"tr\"><head>"
            "<meta charset=\"utf-8\"><title>Agrista Auto-EDA</title>"
            "<style>body{font-family:sans-serif;background:#f4f6f8;"
            "margin:24px}img{max-width:900px;background:#fff;border:1px "
            "solid #dfe3e8;border-radius:8px;margin:8px 0;padding:8px}"
            "h1{color:#2E86AB}</style></head><body>",
            "<h1>Agrista Otomatik Keşif Raporu</h1>",
            f"<p>{len(df)} satır, {len(df.columns)} sütun, "
            f"{len(oneriler)} öneri.</p>"]
    for yol in yollar:
        html.append(f"<img src=\"{os.path.basename(yol)}\"/>")
    html.append("</body></html>")
    rapor_yolu = os.path.join(output_dir, "report.html")
    with open(rapor_yolu, "w", encoding="utf-8") as fh:
        fh.write("".join(html))
    return {"html_path": rapor_yolu, "figures": yollar,
            "suggestions": oneriler}
