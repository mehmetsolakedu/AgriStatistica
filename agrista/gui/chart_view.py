"""Agrista GUI grafik paneli — gömülü matplotlib canvas."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from agrista.viz import AgristaPlotter


def _bar_ortalama(plotter, df, d):
    say = df.groupby(d["kategori"], observed=True)[d["deger"]].mean()
    return plotter.bar_chart([str(k) for k in say.index], list(say.values),
                             xlabel=d["kategori"],
                             ylabel=f"Ortalama {d['deger']}")


def _pasta_toplam(plotter, df, d):
    say = df.groupby(d["kategori"], observed=True)[d["deger"]].sum()
    return plotter.pie_chart([str(k) for k in say.index], list(say.values))


CHART_TYPES = {
    "Histogram": {"alanlar": ["degisken"], "ciz":
                  lambda p, df, d: p.histogram(df[d["degisken"]].dropna())},
    "Kutu Grafiği": {"alanlar": ["grup", "yanit"], "ciz":
                     lambda p, df, d: p.boxplot(df, d["grup"], d["yanit"])},
    "Saçılım": {"alanlar": ["x", "y"], "ciz":
                lambda p, df, d: p.scatter(df[d["x"]], df[d["y"]])},
    "Violin": {"alanlar": ["grup", "yanit"], "ciz":
               lambda p, df, d: p.violin_plot(df, x_col=d["grup"],
                                              y_col=d["yanit"])},
    "Raincloud": {"alanlar": ["grup", "yanit"], "ciz":
                  lambda p, df, d: p.raincloud_plot(df, y_col=d["yanit"],
                                                    group_col=d["grup"])},
    "Ridge": {"alanlar": ["grup", "yanit"], "ciz":
              lambda p, df, d: p.ridge_plot(df, value_col=d["yanit"],
                                            group_col=d["grup"])},
    "Çubuk (ortalama)": {"alanlar": ["kategori", "deger"],
                         "ciz": _bar_ortalama},
    "Çizgi": {"alanlar": ["x", "y"], "ciz":
              lambda p, df, d: p.line_chart(list(df[d["x"]]),
                                            list(df[d["y"]]))},
    "Pasta": {"alanlar": ["kategori", "deger"], "ciz": _pasta_toplam},
    "Q-Q": {"alanlar": ["degisken"], "ciz":
            lambda p, df, d: p.qq_plot(df[d["degisken"]].dropna())},
    "Korelasyon Isı Haritası": {"alanlar": [], "ciz":
                                lambda p, df, d: p.correlation_heatmap(df)},
    "Hata Çubuğu": {"alanlar": ["grup", "yanit"], "ciz":
                    lambda p, df, d: p.error_bar(df, d["grup"], d["yanit"])},
}


class ChartPanel(QWidget):
    """Grafik tipi seçimi + gömülü canvas."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.df = df
        self.plotter = AgristaPlotter()
        self._son_figur = None

        duzen = QVBoxLayout(self)
        self.bilgi = QLabel("Bir grafik tipi seçin ve alanları doldurun.")
        duzen.addWidget(self.bilgi)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 5)))
        duzen.addWidget(self.canvas, stretch=1)

    def set_dataframe(self, df):
        self.df = df

    def uret(self, tip: str, degerler: dict) -> Figure:
        """Grafik üretir, canvas'a çizer ve figürü döndürür."""
        if tip not in CHART_TYPES:
            raise ValueError(f"Bilinmeyen grafik tipi: {tip}")
        if self.df is None or self.df.empty:
            raise ValueError("Grafik için önce veri yükleyin")
        spec = CHART_TYPES[tip]
        eksik = [a for a in spec["alanlar"] if not degerler.get(a)]
        if eksik:
            raise ValueError(f"Eksik alanlar: {eksik}")
        fig = spec["ciz"](self.plotter, self.df, degerler)
        self._son_figur = fig
        self.canvas.figure = fig
        self.canvas.draw()
        self.bilgi.setText(f"{tip} grafiği üretildi.")
        return fig

    def kaydet(self, path: str):
        """Son üretilen grafiği dosyaya kaydeder."""
        if self._son_figur is None:
            raise ValueError("Kaydedilecek grafik yok")
        self.plotter.save(path, fig=self._son_figur)
