# Plan: Masaüstü Wave 1 — Uygulama İçi Güncelleyici + Teslim Kapanışı

**Spec:** `docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md` (§9, §11)

**Goal:** Uygulama içi güncelleme denetimini (saf sürüm karşılaştırma +
ağ katmanı + menü akışı) tamamlamak ve Wave 1'i sürüm 0.4.0, README ve
docs/02 güncellemeleriyle kapatmak.

**Architecture:** `agrista/gui/updater.py`: saf fonksiyonlar
(`parse_version`, `compare_versions`, `build_update_info`) + ağ katmanı
(`fetch_latest`, `check_update`, stdlib urllib). Menü "Güncellemeleri
Denetle…" akışı sonucu iletişim kutusuyla gösterir; yeni sürüm varsa
platform varlık URL'si tarayıcıda açılır.

**Tech Stack:** stdlib (urllib, json, webbrowser), PySide6.
Ön koşul: Plan 1-4 tamamlanmış.

**Global Constraints (spec'ten aynen):**
1. Yeni bağımlılık yok; ağ erişimi stdlib.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. GUI Türkçe.
4. Menü denkliği korunur.
5. İmza gerektirmez.
6. TDD zorunlu; ağ kodu testlerde mock'lanır (gerçek ağ çağrısı yok).
7. Sürüm: 0.3.0 → 0.4.0 (`pyproject.toml`, `__version__`, banner,
   `version_option`, `TestCli::test_version`).
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/gui/updater.py` (yeni) | sürüm karşılaştırma + güncelleme denetimi |
| `agrista/gui/main_window.py` | `_guncelleme_denetle` gerçek akış |
| `tests/test_gui_updater.py` (yeni) | saf + mock'lu testler |
| `README.md`, `pyproject.toml`, `docs/02` | teslim güncellemeleri |

---

## Task 1: Saf güncelleyici fonksiyonları (TDD)

**Files:** Test: `tests/test_gui_updater.py` (Create) · Create: `agrista/gui/updater.py`
**Interfaces:** spec §9 imzaları.

- [ ] **RED** — `tests/test_gui_updater.py` oluştur:

```python
"""Agrista GUI güncelleyici testleri (ağ yok; mock'lu)."""
import json

import pytest


class TestSurumFonksiyonlari:
    def test_parse_version(self):
        from agrista.gui.updater import parse_version
        assert parse_version("0.4.0") == (0, 4, 0)
        assert parse_version("v0.4.0") == (0, 4, 0)

    def test_parse_gecersiz(self):
        from agrista.gui.updater import parse_version
        with pytest.raises(ValueError):
            parse_version("abc")

    def test_compare(self):
        from agrista.gui.updater import compare_versions
        assert compare_versions("0.3.0", "0.4.0") == -1
        assert compare_versions("0.4.0", "0.4.0") == 0
        assert compare_versions("1.0.0", "0.9.9") == 1

    def test_build_update_info_yeni_surum(self):
        from agrista.gui.updater import build_update_info
        payload = {"version": "0.5.0", "notes": "yeni",
                   "assets": {"macos": "u1", "windows": "u2"}}
        bilgi = build_update_info(payload, "0.4.0")
        assert bilgi["guncelleme_var"] is True
        assert bilgi["url"]["macos"] == "u1"

    def test_build_update_info_guncel(self):
        from agrista.gui.updater import build_update_info
        payload = {"version": "0.4.0", "notes": "",
                   "assets": {"macos": "u1", "windows": "u2"}}
        bilgi = build_update_info(payload, "0.4.0")
        assert bilgi["guncelleme_var"] is False


class TestAgKatmani:
    def test_fetch_latest_mock(self, monkeypatch):
        import agrista.gui.updater as up
        payload = {"version": "9.9.9", "notes": "",
                   "assets": {"macos": "m", "windows": "w"}}

        class SahteYanit:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        monkeypatch.setattr(up.urllib.request, "urlopen",
                            lambda *a, **k: SahteYanit())
        assert up.fetch_latest("http://ornek/latest.json")["version"] == "9.9.9"

    def test_check_update_mock(self, monkeypatch):
        import agrista.gui.updater as up
        monkeypatch.setattr(up, "fetch_latest", lambda url, timeout=5.0:
                            {"version": "9.9.9", "notes": "n",
                             "assets": {"macos": "m", "windows": "w"}})
        bilgi = up.check_update("0.4.0", url="http://ornek/latest.json")
        assert bilgi["guncelleme_var"] is True

    def test_check_update_ag_hatasi(self, monkeypatch):
        import agrista.gui.updater as up

        def patlat(*a, **k):
            raise OSError("ağ yok")

        monkeypatch.setattr(up, "fetch_latest", patlat)
        assert up.check_update("0.4.0", url="http://ornek") is None
```

- [ ] Çalıştır → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/gui/updater.py` oluştur:

```python
"""Agrista GUI güncelleyici — GitHub Releases latest.json denetimi."""

from __future__ import annotations

import json
import platform
import re
import urllib.request

DEFAULT_URL = ("https://github.com/mehmetsolakedu/AgriStatistica/"
               "releases/latest/download/latest.json")

_SURUM_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(surum: str) -> tuple:
    """'v0.4.0' → (0, 4, 0)."""
    m = _SURUM_RE.match(surum.strip())
    if not m:
        raise ValueError(f"Geçersiz sürüm: {surum}")
    return tuple(int(x) for x in m.groups())


def compare_versions(a: str, b: str) -> int:
    """-1 / 0 / 1."""
    ta, tb = parse_version(a), parse_version(b)
    return (ta > tb) - (ta < tb)


def build_update_info(payload: dict, current: str) -> dict:
    """latest.json içeriği + geçerli sürüm → güncelleme bilgisi."""
    en_yeni = payload["version"]
    bilgi = {
        "en_yeni": en_yeni,
        "notes": payload.get("notes", ""),
        "url": payload.get("assets", {}),
        "guncelleme_var": compare_versions(current, en_yeni) < 0,
    }
    sistem = "macos" if platform.system() == "Darwin" else "windows"
    bilgi["platform_url"] = bilgi["url"].get(sistem)
    return bilgi


def fetch_latest(url: str, timeout: float = 5.0) -> dict:
    """latest.json indirir (stdlib urllib)."""
    with urllib.request.urlopen(url, timeout=timeout) as yanit:
        return json.loads(yanit.read().decode("utf-8"))


def check_update(current: str, url: str = DEFAULT_URL):
    """Güncelleme denetimi; ağ hatasında None döner."""
    try:
        payload = fetch_latest(url)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return build_update_info(payload, current)
```

- [ ] Çalıştır → yeşil. Commit: `feat(gui): güncelleyici — sürüm denetimi (stdlib)`

## Task 2: Menü akışı entegrasyonu (TDD)

**Files:** Modify: `agrista/gui/main_window.py`, `tests/test_gui_window.py`
**Interfaces:** `MainWindow._guncelleme_denetle()` → bilgi kutusu.

- [ ] **RED** — `tests/test_gui_window.py` sonuna ekle:

```python
class TestGuncellemeMenusu:
    def test_denetim_ag_hatasi_bilgisi(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        import agrista.gui.updater as up
        mesajlar = []
        monkeypatch.setattr(up, "check_update", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda ebeveyn, baslik, metin:
                                         mesajlar.append(metin)))
        pencere._guncelleme_denetle()
        assert any("denetim" in m.lower() or "denetle" in m.lower()
                   for m in mesajlar)

    def test_denetim_yeni_surum(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        import agrista.gui.updater as up
        mesajlar = []
        monkeypatch.setattr(up, "check_update", lambda *a, **k:
                            {"en_yeni": "9.9.9", "notes": "n",
                             "url": {}, "platform_url": None,
                             "guncelleme_var": True})
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda ebeveyn, baslik, metin:
                                         mesajlar.append(metin)))
        pencere._guncelleme_denetle()
        assert any("9.9.9" in m for m in mesajlar)
```

- [ ] Çalıştır → başarısız (placeholder mesaj).
- [ ] **GREEN** — `main_window.py` içinde `_guncelleme_denetle` güncelle:

```python
    def _guncelleme_denetle(self):
        from agrista import __version__
        from agrista.gui import updater
        bilgi = updater.check_update(__version__)
        if bilgi is None:
            QMessageBox.information(
                self, "Güncellemeler",
                "Güncelleme denetimi yapılamadı (ağ bağlantısı yok "
                "veya sunucuya erişilemiyor).")
        elif bilgi["guncelleme_var"]:
            QMessageBox.information(
                self, "Güncellemeler",
                f"Yeni sürüm var: {bilgi['en_yeni']}\n"
                f"{bilgi['notes']}\nİndirme sayfası: Releases.")
        else:
            QMessageBox.information(
                self, "Güncellemeler",
                f"Agrista güncel ({__version__}).")
```

- [ ] Çalıştır → yeşil. Commit: `feat(gui): güncelleme denetimi menü akışı`

## Task 3: Teslim kapanışı — README, sürüm 0.4.0, docs/02

**Files:** Modify: `README.md`, `pyproject.toml`, `agrista/__init__.py`,
`agrista/cli/__init__.py`, `tests/test_viz_cli.py` (`TestCli::test_version`),
`docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md`

- [ ] Sürüm: `pyproject.toml` 0.4.0; `agrista/__init__.py` `__version__`
      0.4.0; CLI banner "v0.4.0"; `version_option(version="0.4.0")`;
      `TestCli::test_version` beklentisi 0.4.0.
- [ ] README'ye "## Masaüstü Uygulaması" bölümü ekle (Kurulum'dan sonra):

```markdown
## Masaüstü Uygulaması

Agrista, PySide6 tabanlı masaüstü uygulaması olarak da kullanılabilir:

```bash
pip install "agrista[gui]"
agrista-gui
```

Uygulama: 21 kategorili menü, veri tablosu görünümü, 16 bağlı analiz
(otomatik formlar), 12 grafik tipi (gömülü canvas, tema desteği),
uygulama içi güncelleme denetimi. macOS/Windows kurulum paketleri
GitHub Releases'ta yayınlanır (`Agrista-<sürüm>-macOS.dmg`,
`Agrista-<sürüm>-Setup.exe`).
```

- [ ] `docs/02` log'una "## Güncelleme 7 — Masaüstü Dağıtım (Wave 1)"
      bölümü: GUI paket dosyaları, 16 bağlı analiz, 12 grafik tipi,
      paketleme hattı, güncelleyici, test dosyaları.
- [ ] Son doğrulama (taze, tam): `.venv/bin/python -m pytest tests/ -q`
      ve `.venv/bin/python -m flake8 agrista tests packaging`.
- [ ] Commit: `chore: v0.4.0 — masaüstü dağıtım Wave 1 teslim güncellemeleri`
