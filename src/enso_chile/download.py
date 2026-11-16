"""Descarga idempotente de subconjuntos de GLORYS12 desde Copernicus Marine.

Envoltorio sobre ``copernicusmarine.subset`` que no vuelve a descargar un
archivo ya presente: ante una ruta existente devuelve esa ruta y no toca la
red, salvo que se pida ``overwrite=True``. Las descargas de los scripts
originales se relanzaban completas con cada ejecucion de la celda, lo que
implicaba repetir gigabytes.

Las credenciales **no** se pasan por codigo: se usa el archivo de credenciales
del CLI, creado una sola vez con ``copernicusmarine login``.

Notes
-----
LEGACY: ``Glorys12_and_Index.py`` l.10-20 (prueba 1993) y l.78-106 (historico
1993-2026); ``Glorys12_and_Index.py`` l.1985-2060 y
``EarlySignal_and_Impact.py`` l.1550-1870 (Nino 3.4).

En los scripts originales la caja geografica se fijaba **en la descarga**, y el
cubo de anomalias heredaba esa caja. Aqui la descarga conserva la caja amplia y
el recorte a las cajas de analisis es responsabilidad de ``cube.py``: asi no
hay que volver a descargar al cambiar un limite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = ["DownloadSkipped", "subset_region", "download_glorys_box"]


class DownloadSkipped(RuntimeError):
    """Se intento descargar sin credenciales de Copernicus Marine disponibles."""


def _format_datetime(value: str) -> str:
    """Normalizar una fecha ``YYYY-MM-DD`` al formato ISO que espera la API."""
    return value if "T" in value else f"{value}T00:00:00"


def subset_region(
    output_path: Path,
    dataset_id: str,
    variables: Sequence[str],
    lon: tuple[float, float],
    lat: tuple[float, float],
    period: tuple[str, str],
    depth: tuple[float, float] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> Path:
    """Descargar un subconjunto del dataset, solo si el archivo no existe.

    Parameters
    ----------
    output_path : Path
        Ruta del NetCDF de salida. Si ya existe y ``overwrite`` es ``False``,
        no se descarga nada.
    dataset_id : str
        Identificador del producto, p. ej. ``config.DATASET_ID``.
    variables : sequence of str
        Variables del modelo a incluir.
    lon, lat : tuple of float
        Limites ``(minimo, maximo)`` en grados.
    period : tuple of str
        Par ``(inicio, fin)`` en formato ``'YYYY-MM-DD'`` o ISO completo.
    depth : tuple of float or None, optional
        Rango vertical ``(minimo, maximo)`` en metros. ``None`` para variables
        sin dimension de profundidad.
    overwrite : bool, optional
        Descargar de nuevo aunque el archivo exista.
    dry_run : bool, optional
        Devolver la ruta sin llamar a la API; util en tests y para revisar los
        parametros antes de gastar ancho de banda.
    **kwargs
        Argumentos adicionales que se pasan tal cual a
        ``copernicusmarine.subset``.

    Returns
    -------
    Path
        Ruta del archivo (existente o recien descargado).

    Raises
    ------
    DownloadSkipped
        Si ``copernicusmarine`` no esta disponible o no hay credenciales.

    Notes
    -----
    El tamanio del archivo no se valida: un NetCDF truncado por una descarga
    interrumpida se considera existente. Ante una descarga cortada, borrar el
    archivo o usar ``overwrite=True``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    if dry_run:
        return output_path

    try:
        import copernicusmarine
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise DownloadSkipped(
            "copernicusmarine no esta instalado en este entorno."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    request: dict[str, Any] = {
        "dataset_id": dataset_id,
        "variables": list(variables),
        "minimum_longitude": lon[0],
        "maximum_longitude": lon[1],
        "minimum_latitude": lat[0],
        "maximum_latitude": lat[1],
        "start_datetime": _format_datetime(period[0]),
        "end_datetime": _format_datetime(period[1]),
        "output_directory": str(output_path.parent),
        "output_filename": output_path.name,
    }
    if depth is not None:
        request["minimum_depth"] = depth[0]
        request["maximum_depth"] = depth[1]
    request.update(kwargs)

    copernicusmarine.subset(**request)
    return output_path


def download_glorys_box(
    output_path: Path,
    box: dict[str, tuple[float, float]],
    dataset_id: str,
    variables: Sequence[str],
    period: tuple[str, str],
    depth: tuple[float, float] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    """Descargar una de las cajas declaradas en ``config``.

    Atajo sobre :func:`subset_region` que acepta directamente una entrada de
    ``config.BOXES``, ``config.BOX_NINO34`` o ``config.BOX_HOVMOLLER``.

    Parameters
    ----------
    output_path : Path
        Ruta del NetCDF de salida.
    box : dict
        Diccionario con claves ``'lat'`` y ``'lon'``, y opcionalmente
        ``'depth'``, que tiene prioridad sobre el argumento ``depth``.
    dataset_id : str
        Identificador del producto.
    variables : sequence of str
        Variables del modelo a incluir.
    period : tuple of str
        Par ``(inicio, fin)``.
    depth : tuple of float or None, optional
        Rango vertical, si la caja no lo trae.
    overwrite : bool, optional
        Descargar de nuevo aunque el archivo exista.
    dry_run : bool, optional
        No llamar a la API.

    Returns
    -------
    Path
        Ruta del archivo.
    """
    return subset_region(
        output_path=output_path,
        dataset_id=dataset_id,
        variables=variables,
        lon=box["lon"],
        lat=box["lat"],
        period=period,
        depth=box.get("depth", depth),
        overwrite=overwrite,
        dry_run=dry_run,
    )
