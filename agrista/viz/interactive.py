"""
Agrista Etkileşimli Görselleştirme — Plotly tabanlı grafikler ve dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RENKLER = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4",
           "#3B1F2B"]


def _dogrula(df: pd.DataFrame, cols: list):
    eksik = [c for c in cols if c is not None and c not in df.columns]
    if eksik:
        raise ValueError(f"Sütun bulunamadı: {eksik}")


def interactive_scatter(df: pd.DataFrame, x: str, y: str,
                        color: str = None,
                        trendline: bool = False) -> go.Figure:
    """Etkileşimli saçılım grafiği."""
    _dogrula(df, [x, y, color])
    fig = px.scatter(df, x=x, y=y, color=color, trendline="ols"
                     if trendline else None, color_discrete_sequence=RENKLER,
                     template="plotly_white")
    fig.update_layout(title="Etkileşimli Saçılım Grafiği")
    return fig


def interactive_line(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli çizgi grafiği."""
    _dogrula(df, [x, y])
    fig = px.line(df, x=x, y=y, color_discrete_sequence=RENKLER,
                  template="plotly_white")
    fig.update_layout(title="Etkileşimli Çizgi Grafiği")
    return fig


def interactive_bar(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli çubuk grafiği."""
    _dogrula(df, [x, y])
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=RENKLER,
                 template="plotly_white")
    fig.update_layout(title="Etkileşimli Çubuk Grafiği")
    return fig


def interactive_heatmap(df: pd.DataFrame,
                        columns: list = None) -> go.Figure:
    """Etkileşimli korelasyon ısı haritası."""
    sayisal = df[columns] if columns else df.select_dtypes(
        include=[np.number])
    if sayisal.shape[1] < 2:
        raise ValueError("Isı haritası için en az 2 sayısal sütun gerekli")
    corr = sayisal.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=list(corr.columns),
                                    y=list(corr.columns), colorscale="RdBu",
                                    zmin=-1, zmax=1))
    fig.update_layout(title="Korelasyon Isı Haritası",
                      template="plotly_white")
    return fig


def interactive_box(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli kutu grafiği."""
    _dogrula(df, [x, y])
    fig = px.box(df, x=x, y=y, color_discrete_sequence=RENKLER,
                 template="plotly_white")
    fig.update_layout(title="Etkileşimli Kutu Grafiği")
    return fig


def interactive_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Etkileşimli histogram."""
    _dogrula(df, [column])
    fig = px.histogram(df, x=column, color_discrete_sequence=RENKLER,
                       template="plotly_white")
    fig.update_layout(title=f"{column} Dağılımı")
    return fig


def build_dashboard(df: pd.DataFrame, output_path: str,
                    target: str = None,
                    title: str = "Agrista Keşif Paneli") -> dict:
    """Tek HTML keşif paneli: KPI kartları + histogramlar + korelasyon."""
    if df.empty:
        raise ValueError("Dashboard için boş olmayan veri gerekli")
    sayisal = df.select_dtypes(include=[np.number])
    sayisal_kolonlar = list(sayisal.columns)[:6]
    if not sayisal_kolonlar:
        raise ValueError("Dashboard için en az 1 sayısal sütun gerekli")

    n_satir, n_sutun = df.shape
    eksik_oran = float(df.isna().sum().sum() / max(n_satir * n_sutun, 1))

    n_hist = len(sayisal_kolonlar)
    fig = make_subplots(rows=2, cols=max(n_hist, 2),
                        specs=[[{"type": "xy"}] * max(n_hist, 2),
                               [{"type": "xy"}, {"type": "table"}]
                               + [None] * (max(n_hist, 2) - 2)],
                        subplot_titles=tuple(sayisal_kolonlar)
                        + ("Korelasyon", "KPI Özet"),
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    n_figures = 0
    for i, kol in enumerate(sayisal_kolonlar):
        fig.add_trace(go.Histogram(x=df[kol].dropna(), name=kol,
                                   marker_color=RENKLER[i % len(RENKLER)],
                                   showlegend=False),
                      row=1, col=i + 1)
        n_figures += 1

    if len(sayisal_kolonlar) >= 2:
        corr = sayisal[sayisal_kolonlar].corr()
        fig.add_trace(go.Heatmap(z=corr.values, x=sayisal_kolonlar,
                                 y=sayisal_kolonlar, colorscale="RdBu",
                                 zmin=-1, zmax=1, showscale=True),
                      row=2, col=1)
        n_figures += 1

    kpi_metin = (f"Satır: {n_satir}<br>Sütun: {n_sutun}<br>"
                 f"Eksik oranı: {eksik_oran:.2%}")
    fig.add_trace(go.Table(header=dict(values=["Özet"]),
                           cells=dict(values=[[kpi_metin]])),
                  row=2, col=2)
    n_figures += 1

    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Hedef sütun bulunamadı: {target}")
        kategori = df.select_dtypes(exclude=[np.number]).columns
        if len(kategori) > 0:
            fig = interactive_box(df, x=str(kategori[0]), y=target)
            n_figures += 1

    fig.update_layout(title_text=title, height=760,
                      template="plotly_white")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return {"path": output_path, "n_figures": int(n_figures),
            "n_rows": int(n_satir)}
