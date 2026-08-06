"""
Agrista Etkileşimli Görselleştirme — Plotly tabanlı grafikler ve dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
