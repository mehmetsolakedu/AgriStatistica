# -*- mode: python ; coding: utf-8 -*-
"""Agrista GUI PyInstaller spesifikasyonu (onedir, pencere modu)."""
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["../agrista/gui/main.py"],
    pathex=[".."],
    hiddenimports=collect_submodules("agrista"),
    datas=collect_data_files("matplotlib"),
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Agrista", console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Agrista")

if sys.platform == "darwin":
    BUNDLE(
        coll, name="Agrista.app",
        bundle_identifier="com.agrista.desktop",
        info_plist={"CFBundleShortVersionString": "0.4.0",
                    "NSHighResolutionCapable": True},
    )
