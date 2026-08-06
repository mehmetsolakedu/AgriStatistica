"""agrista.data modülü testleri."""

import numpy as np
import pandas as pd
import pytest

from agrista.data import AgristaData, load_csv, load_json


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "urun": ["buğday", "arpa", "mısır", "buğday"],
        "verim": [4.5, 3.8, 7.2, np.nan],
        "alan": [10.0, 20.0, 30.0, 40.0],
    })


class TestAgristaData:
    def test_metadata_extraction(self, sample_df):
        data = AgristaData(sample_df)
        data._extract_metadata()
        assert data.metadata["rows"] == 4
        assert "verim" in data.metadata["numeric_columns"]
        assert "urun" in data.metadata["categorical_columns"]
        assert data.metadata["null_counts"]["verim"] == 1

    def test_setter_validates_type(self):
        data = AgristaData()
        with pytest.raises(TypeError):
            data.dataframe = [1, 2, 3]

    def test_filter(self, sample_df):
        data = AgristaData(sample_df)
        filtered = data.filter(urun="buğday")
        assert len(filtered.dataframe) == 2

    def test_filter_missing_column_raises(self, sample_df):
        data = AgristaData(sample_df)
        with pytest.raises(KeyError):
            data.filter(olmayan="x")

    def test_select_columns(self, sample_df):
        data = AgristaData(sample_df)
        subset = data.select_columns(["verim", "alan"])
        assert list(subset.dataframe.columns) == ["verim", "alan"]

    def test_drop_nulls(self, sample_df):
        data = AgristaData(sample_df)
        assert len(data.drop_nulls().dataframe) == 3
        assert len(data.drop_nulls(columns=["alan"]).dataframe) == 4

    def test_rename_columns(self, sample_df):
        data = AgristaData(sample_df)
        renamed = data.rename_columns({"verim": "yield"})
        assert "yield" in renamed.dataframe.columns

    def test_unloaded_raises(self):
        data = AgristaData()
        with pytest.raises(ValueError):
            data.head()
        with pytest.raises(ValueError):
            data.info()

    def test_get_column_missing_raises(self, sample_df):
        data = AgristaData(sample_df)
        with pytest.raises(KeyError):
            data.get_column("yok")


class TestLoaders:
    def test_load_csv_roundtrip(self, sample_df, tmp_path):
        path = tmp_path / "veri.csv"
        sample_df.to_csv(path, index=False)
        data = load_csv(path)
        assert len(data.dataframe) == 4
        assert list(data.dataframe.columns) == list(sample_df.columns)

    def test_load_json_roundtrip(self, sample_df, tmp_path):
        df = sample_df.fillna(0)
        path = tmp_path / "veri.json"
        df.to_json(path, orient="records")
        data = load_json(path)
        assert len(data.dataframe) == 4

    def test_load_csv_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/olmayan/dosya.csv")
