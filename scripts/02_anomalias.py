"""Paso 2: anomalias de temperatura de las cuatro capas de Norte Grande.

Por pixel y capa se ajusta por minimos cuadrados constante + tendencia lineal +
12 efectos mensuales sobre ``config.PERIOD_FULL`` (1993-01 a 2026-03); la
anomalia es el residuo. Los meses posteriores se proyectan en el script 07 con
este mismo ajuste congelado. Uso::

    python scripts/02_anomalias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import config  # noqa: E402
from enso_chile.anomalies import build_anomaly_cube  # noqa: E402
from enso_chile.cube import open_cube, save_cube, select_box, select_depths  # noqa: E402


def main() -> None:
    """Calcular y guardar el cubo de anomalias."""
    crudo = open_cube(config.CUBE_RAW)
    caja = select_box(crudo[list(config.VARIABLES)], config.BOXES[config.BOX])
    capas = select_depths(caja, config.DEPTHS_WORK).sel(time=slice(*config.PERIOD_FULL))

    cubo = build_anomaly_cube(capas, variables=config.VARIABLES, detrend=config.ANOMALY_DETREND,
                              seasonal="monthly", smooth=False)
    cubo.attrs["box"] = config.BOX
    save_cube(cubo, config.CUBE_ANOM, overwrite=True)

    anom = cubo["thetao_anom"]
    print(f"{config.CUBE_ANOM.name}: {dict(cubo.sizes)}")
    for z in anom.depth.values:
        n = int(anom.sel(depth=z).isel(time=0).notnull().sum())
        print(f"  {z:7.3f} m: {n} pixeles de agua")


if __name__ == "__main__":
    main()
