"""Agrista GUI çekirdek testleri (model, tema, format_result)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")


def _df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})


class TestDataFrameModel:
    def test_satir_sutun(self):
        from agrista.gui.data_model import DataFrameModel
        m = DataFrameModel(_df())
        assert m.rowCount() == 3
        assert m.columnCount() == 2

    def test_baslik_dtype(self):
        from agrista.gui.data_model import DataFrameModel
        from PySide6.QtCore import Qt
        m = DataFrameModel(_df())
        assert "float" in m.headerData(0, Qt.Horizontal)
        assert m.headerData(0, Qt.Vertical) == "1"

    def test_hucre_degeri(self):
        from agrista.gui.data_model import DataFrameModel
        from PySide6.QtCore import Qt  # noqa: F401
        m = DataFrameModel(_df())
        assert m.data(m.index(2, 1)) == "z"

    def test_satir_limiti(self):
        from agrista.gui.data_model import DataFrameModel, MAX_ROWS
        buyuk = pd.DataFrame({"x": np.arange(MAX_ROWS + 50, dtype=float)})
        assert DataFrameModel(buyuk).rowCount() == MAX_ROWS

    def test_bos_model(self):
        from agrista.gui.data_model import DataFrameModel
        m = DataFrameModel()
        assert m.rowCount() == 0 and m.columnCount() == 0


class TestTema:
    def test_temalar_tanimli(self):
        from agrista.gui.theme import LIGHT_QSS, DARK_QSS
        assert "QMainWindow" in LIGHT_QSS
        assert "QMainWindow" in DARK_QSS

    def test_tema_secici(self):
        from agrista.gui.theme import tema_qss
        assert tema_qss("açık") != tema_qss("koyu")
        with pytest.raises(ValueError):
            tema_qss("olmayan")


class TestKayitVeFormat:
    def test_kayit_baslangicta_bos(self):
        from agrista.gui.registry import REGISTRY
        assert isinstance(REGISTRY, list)

    def test_param_ve_spec(self):
        from agrista.gui.registry import AnalysisSpec, Param
        p = Param(name="kolon", label="Kolon", kind="column")
        s = AnalysisSpec(key="k", menu_category="m", label="l",
                         run=lambda df, p: {}, params=[p])
        assert s.params[0].kind == "column"

    def test_format_result_dict(self):
        from agrista.gui.registry import format_result
        metin = format_result({"a": 1.5, "b": {"c": "x"}})
        assert "a: 1.5" in metin and "c: x" in metin

    def test_format_result_liste_ve_skaler(self):
        from agrista.gui.registry import format_result
        assert "1" in format_result([1, 2])
        assert format_result(3.25) == "3.25"
