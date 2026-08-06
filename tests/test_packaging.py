"""Paketleme hattı testleri (latest.json üretimi)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "packaging"))

from make_latest_json import build_latest_payload


class TestLatestPayload:
    def test_sema(self):
        p = build_latest_payload("0.4.0", "ilk sürüm",
                                 "https://x/Agrista.dmg",
                                 "https://x/Agrista-Setup.exe")
        assert p["version"] == "0.4.0"
        assert p["assets"]["macos"].endswith(".dmg")
        assert p["assets"]["windows"].endswith(".exe")
        assert p["notes"] == "ilk sürüm"

    def test_gecersiz_sürüm(self):
        import pytest
        with pytest.raises(ValueError):
            build_latest_payload("abc", "", "u1", "u2")

    def test_json_serileştirilebilir(self):
        p = build_latest_payload("0.4.0", "", "u1", "u2")
        assert json.loads(json.dumps(p))["version"] == "0.4.0"
