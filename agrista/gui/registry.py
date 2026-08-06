"""Agrista GUI analiz kaydı — bildirimsel analiz tanımları.

Her AnalysisSpec bir menü öğesini bir analiz fonksiyonuna bağlar;
formlar parametre şemasından otomatik üretilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional  # noqa: F401

import pandas as pd


@dataclass
class Param:
    """Analiz parametresi şeması.

    kind: "column" | "columns" | "numeric" | "choice"
    """
    name: str
    label: str
    kind: str
    required: bool = True
    default: object = None
    choices: tuple = ()


@dataclass
class AnalysisSpec:
    """Bir analiz öğesinin menü + çalışma tanımı."""
    key: str
    menu_category: str
    label: str
    run: Callable[[pd.DataFrame, dict], dict]
    params: list = field(default_factory=list)


REGISTRY: list = []


def format_result(obj, indent: int = 0) -> str:
    """dict/list sonuçlarını hiyerarşik metne çevirir."""
    girinti = "  " * indent
    if isinstance(obj, dict):
        satirlar = []
        for anahtar, deger in obj.items():
            if isinstance(deger, (dict, list)):
                satirlar.append(f"{girinti}{anahtar}:")
                satirlar.append(format_result(deger, indent + 1))
            else:
                satirlar.append(f"{girinti}{anahtar}: {_skaler(deger)}")
        return "\n".join(satirlar)
    if isinstance(obj, (list, tuple)):
        return "\n".join(f"{girinti}- {format_result(o, 0).strip()}"
                         if not isinstance(o, (dict, list))
                         else format_result(o, indent + 1) for o in obj)
    return f"{girinti}{_skaler(obj)}"


def _skaler(deger) -> str:
    if isinstance(deger, float):
        return f"{deger:.6g}"
    return str(deger)
