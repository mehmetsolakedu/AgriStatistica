"""Agrista GUI veri modeli — pandas DataFrame → Qt tablo."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt

MAX_ROWS = 10000


class DataFrameModel(QAbstractTableModel):
    """Salt-okunur pandas tablo modeli (ilk MAX_ROWS satır)."""

    def __init__(self, df: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._df = df.head(MAX_ROWS).reset_index(drop=True) \
            if df is not None else pd.DataFrame()

    def rowCount(self, parent=None) -> int:
        return len(self._df)

    def columnCount(self, parent=None) -> int:
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            kolon = self._df.columns[section]
            return f"{kolon} ({self._df[kolon].dtype})"
        return str(section + 1)

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.head(MAX_ROWS).reset_index(drop=True)
        self.endResetModel()
