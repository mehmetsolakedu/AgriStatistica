"""Release `latest.json` üretici — güncelleme denetiminin veri kaynağı."""

from __future__ import annotations

import argparse
import json
import re

SURUM_RE = re.compile(r"^\d+\.\d+\.\d+$")


def build_latest_payload(version: str, notes: str, macos_url: str,
                         windows_url: str) -> dict:
    """latest.json içeriğini üretir (saf, doğrulamalı)."""
    if not SURUM_RE.match(version):
        raise ValueError(f"Geçersiz sürüm: {version}")
    return {
        "version": version,
        "notes": notes,
        "assets": {"macos": macos_url, "windows": windows_url},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="latest.json üret")
    ap.add_argument("--version", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--macos-url", required=True)
    ap.add_argument("--windows-url", required=True)
    ap.add_argument("--out", default="latest.json")
    a = ap.parse_args(argv)
    payload = build_latest_payload(a.version, a.notes, a.macos_url,
                                   a.windows_url)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
