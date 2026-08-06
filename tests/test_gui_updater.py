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
