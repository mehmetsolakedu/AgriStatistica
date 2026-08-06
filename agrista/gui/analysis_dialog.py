"""Agrista GUI analiz diyaloğu — parametre şemasından otomatik form."""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLineEdit)


class AnalysisDialog(QDialog):
    """AnalysisSpec parametrelerinden üretilen girdi formu."""

    def __init__(self, spec, df, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.df = df
        self.setWindowTitle(spec.label)
        self.setMinimumWidth(420)
        self._alanlar = {}
        form = QFormLayout(self)
        for prm in spec.params:
            form.addRow(prm.label + ":", self._alan_kur(prm))
        dugmeler = QDialogButtonBox(QDialogButtonBox.Ok |
                                    QDialogButtonBox.Cancel)
        dugmeler.accepted.connect(self.accept)
        dugmeler.rejected.connect(self.reject)
        form.addRow(dugmeler)

    def _alan_kur(self, prm):
        if prm.kind == "column":
            kutu = QComboBox()
            kutu.addItems([str(c) for c in self.df.columns])
            if prm.default is not None:
                kutu.setCurrentText(str(prm.default))
            widget = kutu
        elif prm.kind == "columns":
            widget = QLineEdit(str(prm.default or ""))
        elif prm.kind == "numeric":
            kutu = QDoubleSpinBox()
            kutu.setRange(-1e12, 1e12)
            kutu.setDecimals(6)
            kutu.setValue(float(prm.default or 0.0))
            widget = kutu
        elif prm.kind == "choice":
            kutu = QComboBox()
            kutu.addItems([str(c) for c in prm.choices])
            if prm.default is not None:
                kutu.setCurrentText(str(prm.default))
            widget = kutu
        else:
            raise ValueError(f"Bilinmeyen parametre türü: {prm.kind}")
        self._alanlar[prm.name] = (prm, widget)
        return widget

    def widget(self, ad: str):
        return self._alanlar[ad][1]

    def degerler(self) -> dict:
        sonuc = {}
        for ad, (prm, widget) in self._alanlar.items():
            if prm.kind in ("column", "choice"):
                sonuc[ad] = widget.currentText()
            elif prm.kind == "columns":
                sonuc[ad] = widget.text().strip()
            elif prm.kind == "numeric":
                sonuc[ad] = float(widget.value())
        return sonuc
