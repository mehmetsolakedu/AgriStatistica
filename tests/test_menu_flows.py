"""Etkileşimli menü akışlarının pytest'e entegrasyonu.

`menu_smoke_test.py` içindeki 58 akışın her biri ayrı bir parametrize
vakası olarak çalıştırılır; böylece menü işleyicileri pytest kapsam
raporuna dahil olur.
"""

import os
import sys

import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from menu_smoke_test import FLOWS, cli_main  # noqa: E402


@pytest.mark.parametrize(
    "menu,item,inputs,expect",
    FLOWS,
    ids=[f"{menu}-{item}" for menu, item, _, _ in FLOWS],
)
def test_menu_flow(menu, item, inputs, expect):
    result = CliRunner().invoke(cli_main, [], input=inputs)
    assert result.exit_code == 0, (
        f"{menu} / {item} çıkış kodu {result.exit_code}\n"
        f"{(result.output or '')[-800:]}")
    if expect:
        assert expect in result.output, (
            f"{menu} / {item} çıktısında '{expect}' bulunamadı\n"
            f"{result.output[-800:]}")
