# Plan: Masaüstü Wave 1 — Analiz Koşucu (Kayıt + Otomatik Form)

**Spec:** `docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md` (§6)

**Goal:** 16 analizi bildirimsel kayda bağlamak, parametre şemasından
otomatik form diyaloğu üretmek ve ana pencereye uçtan uca entegre etmek.

**Architecture:** `registry.py` içine adaptörler + REGISTRY doldurulur;
yeni `analysis_dialog.py` şemadan form üretir; `MainWindow.analiz_calistir`
diyaloğu açıp sonucu panelde gösterir. Menü öğeleri kayıtla eşleşince
otomatik etkinleşir (menü kurulumu kayıttan okur).

**Tech Stack:** PySide6, pandas, mevcut analiz fonksiyonları.
Ön koşul: Plan 1 tamamlanmış.

**Global Constraints (spec'ten aynen):**
1. PySide6 opsiyonel ekstra; başka yeni bağımlılık yok; ağ kodu stdlib.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. GUI Türkçe; mevcut analiz fonksiyonları değişmez.
4. Menü 21 kategoriyle birebir; kayıtsız öğeler devre dışı "(planlanıyor)".
5. İmza konusu bu planda yok.
6. TDD zorunlu; widget testleri pytest-qt + offscreen.
7. Sürüm bu planda değişmez.
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/gui/registry.py` | adaptörler + 16 REGISTRY öğesi |
| `agrista/gui/analysis_dialog.py` (yeni) | `AnalysisDialog` otomatik form |
| `agrista/gui/main_window.py` | `analiz_calistir` entegrasyonu |
| `tests/test_gui_registry.py` (yeni) | kayıt + adaptör + diyalog testleri |

## Kayıt Eşlemesi (menü etiketleri CLI yapısından birebir)

| # | Kategori | Etiket | Fonksiyon |
|---|---|---|---|
| 1 | 📊 Betimsel İstatistikler | Betimsel özet tablosu | `descriptive_stats` |
| 2 | 📊 Betimsel İstatistikler | Frekans tabloları | `frequencies` |
| 3 | 📊 Betimsel İstatistikler | Çapraz tablolar (ki-kare) | `crosstabs` |
| 4 | 📊 Betimsel İstatistikler | Oran istatistikleri | `ratio_statistics` |
| 5 | 📊 Betimsel İstatistikler | Normallik testi (Shapiro-Wilk) | `normality_test` |
| 6 | ⚖️ Ortalamaların Karşılaştırılması | Tek örneklem t-testi | `one_sample_t_test` |
| 7 | ⚖️ Ortalamaların Karşılaştırılması | Bağımsız iki örneklem t-testi | `t_test` (grup bölme adaptörü) |
| 8 | ⚖️ Ortalamaların Karşılaştırılması | Eşleştirilmiş iki örneklem t-testi | `paired_t_test` |
| 9 | ⚖️ Ortalamaların Karşılaştırılması | Tek yönlü ANOVA + Tukey HSD | `anova_one_way` + `posthoc_tukey` |
| 10 | ⚖️ Ortalamaların Karşılaştırılması | Genel Doğrusal Model (Tek Değişkenli) | `glm_univariate` |
| 11 | ⚖️ Ortalamaların Karşılaştırılması | Tekrarlı Ölçümler (GLM) | `glm_repeated_measures` |
| 12 | 🔗 Korelasyon | İki değişkenli korelasyon (Pearson/Spearman) | `correlation_analysis` |
| 13 | 📈 Regresyon | GEE (Genelleştirilmiş Tahmin Denklemleri) | `gee_model` |
| 14 | 🎯 ROC Eğrisi | ROC / AUC (Youden eşik) | `roc_curve` |
| 15 | ⏳ Yaşam Analizi (Survival) | Kaplan-Meier sağkalım tablosu | `kaplan_meier` |
| 16 | 🌿 Uzman Branş Modülleri | Bitki Koruma — AUDPC (hastalık ilerlemesi) | `protection.audpc` |

---

## Task 1: Adaptörler + REGISTRY (TDD)

**Files:** Test: `tests/test_gui_registry.py` (Create) · Modify: `agrista/gui/registry.py`
**Interfaces:** her adaptör `(df: DataFrame, p: dict) -> dict`; `REGISTRY` 16 öğe.

- [ ] **RED** — `tests/test_gui_registry.py` oluştur:

```python
"""Agrista GUI analiz kaydı ve adaptör testleri."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 60
    grup = rng.choice(["A", "B"], n)
    etki = np.where(grup == "A", 0.0, 2.0)
    return pd.DataFrame({
        "grup": grup,
        "verim": 10 + etki + rng.normal(0, 1, n),
        "nem": rng.uniform(20, 80, n),
        "once": rng.normal(5, 1, n),
        "sonra": rng.normal(6, 1, n),
        "zaman": np.tile([1, 2, 3, 4], 15).astype(float),
        "siddet": rng.uniform(0, 100, n),
        "gercek": rng.integers(0, 2, n),
        "skor": rng.uniform(0, 1, n),
        "olay": rng.integers(0, 2, n),
        "sure": rng.exponential(10, n),
    })


class TestKayitButunlugu:
    def test_onalti_analiz(self):
        from agrista.gui.registry import REGISTRY
        assert len(REGISTRY) == 16

    def test_menu_denkligi(self):
        """Kayıttaki her (kategori, etiket) CLI menüsünde var olmalı."""
        from agrista.cli import _build_menu_structure
        from agrista.gui.registry import REGISTRY
        cli = {baslik: {etiket for etiket, _ in islemler}
               for baslik, islemler in _build_menu_structure()}
        for spec in REGISTRY:
            assert spec.label in cli.get(spec.menu_category, set()), \
                f"Menüde yok: {spec.menu_category} / {spec.label}"

    def test_key_benzersiz(self):
        from agrista.gui.registry import REGISTRY
        keyler = [s.key for s in REGISTRY]
        assert len(keyler) == len(set(keyler))


class TestAdaptorler:
    def _calistir(self, key, df, p):
        from agrista.gui.registry import REGISTRY
        spec = next(s for s in REGISTRY if s.key == key)
        return spec.run(df, p)

    def test_betimsel(self, df):
        res = self._calistir("betimsel", df, {"kolonlar": "verim,nem"})
        assert "verim" in res and "count" in res["verim"]

    def test_frekans(self, df):
        res = self._calistir("frekans", df, {"kolonlar": "grup"})
        assert res  # dolu dict

    def test_capraz(self, df):
        res = self._calistir("capraz", df,
                             {"satir": "grup", "sutun": "gercek"})
        assert res

    def test_tek_orneklem(self, df):
        res = self._calistir("tek_orneklem_t", df,
                             {"kolon": "verim", "deger": 10.0})
        assert "p_value" in res

    def test_bagimsiz_t(self, df):
        res = self._calistir("bagimsiz_t", df,
                             {"yanit": "verim", "grup": "grup"})
        assert res["p_value"] < 0.05  # etki 2.0 tasarlandı

    def test_eslestirilmis_t(self, df):
        res = self._calistir("eslestirilmis_t", df,
                             {"once": "once", "sonra": "sonra"})
        assert "p_value" in res

    def test_anova_tukey(self, df):
        res = self._calistir("anova_tukey", df,
                             {"yanit": "verim", "grup": "grup"})
        assert "anova" in res and "tukey" in res

    def test_glm(self, df):
        res = self._calistir("glm_univariate", df,
                             {"yanit": "verim", "faktorler": "grup"})
        assert res["model"] == "GLM Univariate"

    def test_korelasyon(self, df):
        res = self._calistir("korelasyon", df, {})
        assert res

    def test_roc(self, df):
        res = self._calistir("roc", df,
                             {"gercek": "gercek", "skor": "skor"})
        assert "auc" in res

    def test_kaplan_meier(self, df):
        res = self._calistir("kaplan_meier", df,
                             {"zaman": "sure", "olay": "olay"})
        assert "survival" in res

    def test_audpc(self, df):
        res = self._calistir("audpc", df,
                             {"zaman": "zaman", "siddet": "siddet"})
        assert res

    def test_eksik_sutun_hatasi(self, df):
        with pytest.raises(ValueError):
            self._calistir("betimsel", df, {"kolonlar": "yok"})
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_registry.py -x -q`
      → `assert 0 == 16` (kayıt boş).
- [ ] **GREEN** — `agrista/gui/registry.py` dosyasına `REGISTRY`
      tanımından ÖNCE adaptörleri ekle, `REGISTRY` listesini doldur:

```python
def _kolonlar(p: dict, anahtar: str) -> list:
    ham = p.get(anahtar) or ""
    return [c.strip() for c in str(ham).split(",") if c.strip()]


def _betimsel(df, p):
    from agrista.analysis import descriptive_stats
    kol = _kolonlar(p, "kolonlar")
    return descriptive_stats(df[kol] if kol else df)


def _frekans(df, p):
    from agrista.analysis import frequencies
    kol = _kolonlar(p, "kolonlar")
    return frequencies(df, columns=kol or None)


def _capraz(df, p):
    from agrista.analysis import crosstabs
    return crosstabs(df, p["satir"], p["sutun"])


def _oran(df, p):
    from agrista.analysis import ratio_statistics
    return ratio_statistics(df, p["pay"], p["payda"])


def _normallik(df, p):
    from agrista.analysis import normality_test
    return normality_test(df[p["kolon"]])


def _tek_orneklem_t(df, p):
    from agrista.analysis import one_sample_t_test
    return one_sample_t_test(df[p["kolon"]], test_value=float(p["deger"]))


def _bagimsiz_t(df, p):
    from agrista.analysis import t_test
    duzeyler = sorted(df[p["grup"]].dropna().unique(), key=str)
    if len(düzeyler := duzeyler) < 2:
        raise ValueError("Grup sütunu en az 2 düzey içermeli")
    g1 = df.loc[df[p["grup"]] == duzeyler[0], p["yanit"]]
    g2 = df.loc[df[p["grup"]] == duzeyler[1], p["yanit"]]
    return t_test(g1, g2)


def _eslestirilmis_t(df, p):
    from agrista.analysis import paired_t_test
    return paired_t_test(df[p["once"]], df[p["sonra"]])


def _anova_tukey(df, p):
    from agrista.analysis import anova_one_way, posthoc_tukey
    gruplar = [g[p["yanit"]].dropna().values
               for _, g in df.groupby(p["grup"], observed=True)]
    if len(gruplar) < 2:
        raise ValueError("ANOVA için en az 2 grup gerekli")
    return {"anova": anova_one_way(*gruplar),
            "tukey": posthoc_tukey(df, p["yanit"], p["grup"])}


def _glm_univariate(df, p):
    from agrista.analysis import glm_univariate
    return glm_univariate(df, response=p["yanit"],
                          between_factors=_kolonlar(p, "faktorler"),
                          posthoc=None)


def _glm_tekrarli(df, p):
    from agrista.analysis import glm_repeated_measures
    return glm_repeated_measures(df, response_cols=_kolonlar(p, "kosullar"),
                                 subject_col=p["denek"])


def _korelasyon(df, p):
    from agrista.analysis import correlation_analysis
    return correlation_analysis(df, method=p.get("yontem", "pearson"))


def _gee(df, p):
    from agrista.analysis import gee_model
    return gee_model(df, response=p["yanit"],
                     covariates=_kolonlar(p, "degiskenler"),
                     group_col=p["grup"])


def _roc(df, p):
    from agrista.analysis import roc_curve
    return roc_curve(df[p["gercek"]], df[p["skor"]])


def _kaplan_meier(df, p):
    from agrista.survival import kaplan_meier
    return kaplan_meier(df[p["zaman"]], df[p["olay"]])


def _audpc(df, p):
    from agrista.protection import audpc
    return audpc(df[p["zaman"]], df[p["siddet"]])


B = "📊 Betimsel İstatistikler"
O = "⚖️ Ortalamaların Karşılaştırılması"

REGISTRY: list = [
    AnalysisSpec("betimsel", B, "Betimsel özet tablosu", _betimsel,
                 [Param("kolonlar", "Sütunlar (boş = tüm sayısal)",
                        "columns", required=False)]),
    AnalysisSpec("frekans", B, "Frekans tabloları", _frekans,
                 [Param("kolonlar", "Kategorik sütunlar (virgülle)",
                        "columns", required=False)]),
    AnalysisSpec("capraz", B, "Çapraz tablolar (ki-kare)", _capraz,
                 [Param("satir", "Satır değişkeni", "column"),
                  Param("sutun", "Sütun değişkeni", "column")]),
    AnalysisSpec("oran", B, "Oran istatistikleri", _oran,
                 [Param("pay", "Pay değişkeni", "column"),
                  Param("payda", "Payda değişkeni", "column")]),
    AnalysisSpec("normallik", B, "Normallik testi (Shapiro-Wilk)",
                 _normallik, [Param("kolon", "Sayısal sütun", "column")]),
    AnalysisSpec("tek_orneklem_t", O, "Tek örneklem t-testi",
                 _tek_orneklem_t,
                 [Param("kolon", "Sayısal sütun", "column"),
                  Param("deger", "Karşılaştırılacak değer", "numeric",
                        default=0.0)]),
    AnalysisSpec("bagimsiz_t", O, "Bağımsız iki örneklem t-testi",
                 _bagimsiz_t,
                 [Param("yanit", "Yanıt değişkeni", "column"),
                  Param("grup", "Grup sütunu", "column")]),
    AnalysisSpec("eslestirilmis_t", O, "Eşleştirilmiş iki örneklem t-testi",
                 _eslestirilmis_t,
                 [Param("once", "Önce ölçümü", "column"),
                  Param("sonra", "Sonra ölçümü", "column")]),
    AnalysisSpec("anova_tukey", O, "Tek yönlü ANOVA + Tukey HSD",
                 _anova_tukey,
                 [Param("yanit", "Yanıt değişkeni", "column"),
                  Param("grup", "Grup sütunu", "column")]),
    AnalysisSpec("glm_univariate", O,
                 "Genel Doğrusal Model (Tek Değişkenli)", _glm_univariate,
                 [Param("yanit", "Bağımlı değişken", "column"),
                  Param("faktorler", "Faktörler (virgülle)", "columns")]),
    AnalysisSpec("glm_tekrarli", O, "Tekrarlı Ölçümler (GLM)",
                 _glm_tekrarli,
                 [Param("kosullar", "Ölçüm sütunları (virgülle)", "columns"),
                  Param("denek", "Denek sütunu", "column")]),
    AnalysisSpec("korelasyon", "🔗 Korelasyon",
                 "İki değişkenli korelasyon (Pearson/Spearman)",
                 _korelasyon,
                 [Param("yontem", "Yöntem", "choice", default="pearson",
                        choices=("pearson", "spearman"))]),
    AnalysisSpec("gee", "📈 Regresyon",
                 "GEE (Genelleştirilmiş Tahmin Denklemleri)", _gee,
                 [Param("yanit", "Bağımlı değişken", "column"),
                  Param("degiskenler", "Açıklayıcılar (virgülle)",
                        "columns"),
                  Param("grup", "Küme/grup sütunu", "column")]),
    AnalysisSpec("roc", "🎯 ROC Eğrisi", "ROC / AUC (Youden eşik)", _roc,
                 [Param("gercek", "Gerçek sınıf (0/1)", "column"),
                  Param("skor", "Skor sütunu", "column")]),
    AnalysisSpec("kaplan_meier", "⏳ Yaşam Analizi (Survival)",
                 "Kaplan-Meier sağkalım tablosu", _kaplan_meier,
                 [Param("zaman", "Zaman sütunu", "column"),
                  Param("olay", "Olay sütunu (0/1)", "column")]),
    AnalysisSpec("audpc", "🌿 Uzman Branş Modülleri",
                 "Bitki Koruma — AUDPC (hastalık ilerlemesi)", _audpc,
                 [Param("zaman", "Zaman sütunu", "column"),
                  Param("siddet", "Hastalık şiddeti sütunu", "column")]),
]
```

  DİKKAT: `_bagimsiz_t` içindeki `len(düzeyler := duzeyler)` satırı
  yanlış yazılmıştır — doğrusu yalnızca `if len(duzeyler) < 2:` olacak.
  Bu düzeltme GREEN adımının parçasıdır.
  `crosstabs` ve `ratio_statistics` imzaları uygulamadan önce
  `agrista/analysis/__init__.py` içinden doğrulanır (sıralı parametre
  adları farklıysa adaptör o adlara uydurulur; testler değişmez).
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_registry.py -q` → yeşil.
- [ ] Commit: `feat(gui): analiz kaydı — 16 analiz adaptörlerle bağlandı`

## Task 2: AnalysisDialog otomatik form (TDD)

**Files:** Test: `tests/test_gui_registry.py` (ek) · Create: `agrista/gui/analysis_dialog.py`
**Interfaces:** `AnalysisDialog(spec, df, parent=None)`, `degerler() -> dict`.

- [ ] **RED** — test dosyası sonuna ekle:

```python
class TestAnalysisDialog:
    def test_form_alanlari(self, qtbot, df):
        from agrista.gui.analysis_dialog import AnalysisDialog
        from agrista.gui.registry import REGISTRY
        spec = next(s for s in REGISTRY if s.key == "tek_orneklem_t")
        dlg = AnalysisDialog(spec, df)
        qtbot.addWidget(dlg)
        assert dlg.windowTitle() == "Tek örneklem t-testi"
        degerler = dlg.degerler()
        assert "kolon" in degerler and "deger" in degerler

    def test_kolon_secenekleri_veriden(self, qtbot, df):
        from agrista.gui.analysis_dialog import AnalysisDialog
        from agrista.gui.registry import REGISTRY
        spec = next(s for s in REGISTRY if s.key == "tek_orneklem_t")
        dlg = AnalysisDialog(spec, df)
        qtbot.addWidget(dlg)
        assert dlg.widget("kolon").count() == len(df.columns)
```

- [ ] Çalıştır → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/gui/analysis_dialog.py` oluştur:

```python
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
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_registry.py -q` → yeşil.
- [ ] Commit: `feat(gui): AnalysisDialog — şemadan otomatik form`

## Task 3: Ana pencere entegrasyonu + uçtan uca (TDD)

**Files:** Modify: `agrista/gui/main_window.py`, `tests/test_gui_window.py`
**Interfaces:** `MainWindow.analiz_calistir(spec)` diyalog açar.

- [ ] **RED** — `tests/test_gui_window.py` sonuna ekle:

```python
class TestAnalizAkisi:
    def test_bagli_ogeler_etkin(self, pencere):
        bar = pencere.menuBar()
        kategori = [a.menu() for a in bar.actions()
                    if a.text() == "📊 Betimsel İstatistikler"][0]
        adlar = {o.text() for o in kategori.actions()}
        assert "Betimsel özet tablosu" in adlar
        bagli = [o for o in kategori.actions()
                 if o.text() == "Betimsel özet tablosu"][0]
        assert bagli.isEnabled()

    def test_analiz_uc_tan_uca(self, pencere, qtbot, tmp_path, monkeypatch):
        from agrista.gui.analysis_dialog import AnalysisDialog
        from agrista.gui.registry import REGISTRY
        pencere.open_file(_csv(tmp_path))
        spec = next(s for s in REGISTRY if s.key == "betimsel")
        monkeypatch.setattr(AnalysisDialog, "exec", lambda self: 1)
        monkeypatch.setattr(AnalysisDialog, "degerler",
                            lambda self: {"kolonlar": "x"})
        pencere.analiz_calistir(spec)
        assert "count" in pencere.sonuc_paneli.toPlainText()

    def test_verisiz_analiz_uyarisi(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        from agrista.gui.registry import REGISTRY
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, **k: None))
        spec = REGISTRY[0]
        pencere.analiz_calistir(spec)
        assert pencere.sonuc_paneli.toPlainText() == ""
```

- [ ] Çalıştır: `-k AnalizAkisi` → başarısız (placeholder davranış).
- [ ] **GREEN** — `main_window.py` içinde `analiz_calistir` metodunu
      değiştir ve import'lara ekle (`from PySide6.QtWidgets import QDialog,
      QMessageBox`; `from agrista.gui.analysis_dialog import AnalysisDialog`):

```python
    def analiz_calistir(self, spec):
        """Kayıtlı analizi diyalogla çalıştırır, sonucu panelde gösterir."""
        if self.df.empty:
            QMessageBox.warning(self, "Veri yok",
                                "Önce Dosya → Veri Aç ile veri yükleyin.")
            return
        dlg = AnalysisDialog(spec, self.df, self)
        if dlg.exec() == QDialog.Accepted:
            try:
                sonuc = spec.run(self.df, dlg.degerler())
            except ValueError as e:
                QMessageBox.critical(self, "Analiz hatası", str(e))
                return
            self.sonuc_goster(spec.label, sonuc)
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_window.py tests/test_gui_registry.py tests/test_gui_core.py -q` → yeşil.
- [ ] Tam doğrulama: `.venv/bin/python -m pytest tests/ -q` ve
      `.venv/bin/python -m flake8 agrista tests` → temiz.
- [ ] Commit: `feat(gui): analiz koşucu entegrasyonu — 16 analiz uçtan uca`
