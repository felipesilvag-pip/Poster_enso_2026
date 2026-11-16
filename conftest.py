"""Configuracion de pytest: pone la raiz del repo y ``src`` en ``sys.path``.

Evita instalar el paquete para correr los tests y permite ``import config`` y
``from enso_chile import ...`` desde ``tests/`` y desde los notebooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
