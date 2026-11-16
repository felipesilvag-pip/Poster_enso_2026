"""Tests de la descarga idempotente, sin tocar la red."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from enso_chile.download import download_glorys_box, subset_region


def _call(output_path: Path, **kwargs) -> Path:
    return subset_region(
        output_path=output_path,
        dataset_id=config.DATASET_ID,
        variables=config.VARIABLES,
        lon=(-74.0, -69.0),
        lat=(-27.0, -18.0),
        period=config.PERIOD_FULL,
        depth=config.DEPTH_RANGE,
        **kwargs,
    )


def test_no_descarga_si_el_archivo_existe(tmp_path) -> None:
    """Ante un archivo presente se devuelve su ruta sin llamar a la API.

    No se usa ``dry_run``: si el cortocircuito por existencia no funcionara,
    la llamada real a ``copernicusmarine`` haria fallar el test.
    """
    existente = tmp_path / "cubo.nc"
    existente.write_bytes(b"netcdf falso")

    assert _call(existente) == existente
    assert existente.read_bytes() == b"netcdf falso"


def test_dry_run_no_crea_el_archivo(tmp_path) -> None:
    """``dry_run`` devuelve la ruta prevista sin descargar ni crear nada."""
    destino = tmp_path / "nuevo.nc"

    assert _call(destino, dry_run=True) == destino
    assert not destino.exists()


def test_overwrite_con_dry_run_no_borra(tmp_path) -> None:
    """``overwrite`` no destruye el archivo antes de una descarga simulada."""
    existente = tmp_path / "cubo.nc"
    existente.write_bytes(b"datos")

    assert _call(existente, overwrite=True, dry_run=True) == existente
    assert existente.exists()


def test_box_prioriza_su_propia_profundidad(tmp_path) -> None:
    """Una caja con 'depth' propio se acepta igual que una sin el."""
    destino = tmp_path / "con_profundidad.nc"
    resultado = download_glorys_box(
        output_path=destino,
        box={**config.BOXES["norte_grande"], "depth": config.DEPTH_RANGE},
        dataset_id=config.DATASET_ID,
        variables=config.VARIABLES,
        period=config.PERIOD_FULL,
        dry_run=True,
    )

    assert resultado == destino


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("1993-01-01", "1993-01-01T00:00:00"), ("1993-01-01T12:00:00", "1993-01-01T12:00:00")],
)
def test_formato_de_fecha(entrada: str, esperado: str) -> None:
    """Las fechas sin hora se completan a ISO; las que ya la traen no cambian."""
    from enso_chile.download import _format_datetime

    assert _format_datetime(entrada) == esperado
