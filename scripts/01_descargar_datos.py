"""Paso 1: descargar los datos del poster.

- GLORYS12 (Copernicus Marine): temperatura potencial mensual de Norte Grande,
  0-160 m, 1993-01 a 2026-08. Requiere una cuenta gratuita de Copernicus Marine
  y haber ejecutado antes, una sola vez, ``copernicusmarine login``.
- Indices de TSM Nino 1+2, 3, 3.4 y 4 del CPC/NOAA (sin cuenta).

Si un archivo ya existe no se vuelve a descargar (usar ``--refrescar`` para los
indices). Uso::

    python scripts/01_descargar_datos.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import config  # noqa: E402
from enso_chile.cube import _without_distributed  # noqa: E402
from enso_chile.download import download_glorys_box  # noqa: E402
from enso_chile.indices import load_nino_regions  # noqa: E402


def main() -> None:
    """Descargar GLORYS12 y los indices Nino."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refrescar", action="store_true", help="volver a bajar los indices del CPC")
    args = parser.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"GLORYS12 -> {config.CUBE_RAW}")
    # copernicusmarine escribe con xarray, que puede importar dask.distributed y su
    # contexto SSL; en equipos con el almacen de certificados de Windows danado eso
    # aborta la escritura. El bloqueo es inocuo en el resto de los equipos.
    with _without_distributed():
        download_glorys_box(
            output_path=config.CUBE_RAW,
            box=config.BOXES[config.BOX],
            dataset_id=config.DATASET_ID,
            variables=config.VARIABLES,
            period=config.PERIOD_DOWNLOAD,
            depth=config.DEPTH_RANGE,
        )
    print(f"  {config.CUBE_RAW.stat().st_size / 2**20:.0f} MB")

    config.INDICES_CACHE.mkdir(parents=True, exist_ok=True)
    nino = load_nino_regions(cache_dir=config.INDICES_CACHE, refresh=args.refrescar)
    print(f"Indices Nino (CPC) -> {config.INDICES_CACHE}: {nino.index[0]:%Y-%m} a {nino.index[-1]:%Y-%m}")


if __name__ == "__main__":
    main()
