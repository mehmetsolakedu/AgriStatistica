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
