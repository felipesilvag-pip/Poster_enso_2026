"""Apertura perezosa del cubo, recorte a las cajas y aplanado para los EOF.

Toda operacion abre el NetCDF con ``xr.open_dataset`` sin cargar datos: el
cubo crudo pesa 1.8 GB y nunca debe entrar completo en memoria. Los recortes y
selecciones siguen siendo perezosos; los datos se materializan solo al llamar
``.load()`` o al escribir a disco.

Las cajas de analisis se recortan **aqui**, no en la descarga. En los scripts
originales la caja se fijaba en ``copernicusmarine.subset`` y el cubo de
anomalias heredaba esa caja, de modo que cambiar un limite obligaba a volver a
descargar.

Notes
-----
LEGACY: ``COAST.py`` l.31-45 (apertura y seleccion de profundidades, sin
recorte espacial); ``Glorys12_and_Index.py`` l.837 y siguientes (recorte
redundante con ``.sel(slice)`` sobre un cubo ya recortado en la descarga);
``Entendiendo_EOF.py`` l.8-10 (usa el cubo completo).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

__all__ = [
    "open_cube",
    "save_cube",
    "select_box",
    "select_depths",
    "water_mask",
    "area_weights",
    "apply_area_weights",
    "flatten_space",
    "unflatten_space",
]


def open_cube(path: Path | str, **kwargs) -> xr.Dataset:
    """Abrir un cubo NetCDF de forma perezosa.

    Parameters
    ----------
    path : Path or str
        Ruta del archivo.
    **kwargs
        Argumentos adicionales para ``xarray.open_dataset``, p. ej.
        ``chunks={'time': 60}`` para forzar dask.

    Returns
    -------
    xarray.Dataset
        Cubo abierto sin cargar datos en memoria.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el cubo: {path}")
    kwargs.setdefault("decode_timedelta", False)
    return xr.open_dataset(path, **kwargs)


@contextmanager
def _without_distributed() -> Iterator[None]:
    """Evitar que ``xarray`` importe ``distributed`` al escribir un NetCDF.

    ``xarray.backends.locks._get_scheduler`` hace ``from dask.distributed
    import Client`` sin condicion en cada ``to_netcdf``, y captura el
    ``ImportError`` si no esta disponible. En esta maquina esa cadena de
    importaciones termina en ``tornado.netutil``, que crea un contexto TLS por
    defecto, y el almacen de certificados de Windows esta corrupto: toda
    escritura falla con ``SSLError: [ASN1: NOT_ENOUGH_DATA]``.

    Mientras dura el bloque se pone un centinela en ``sys.modules`` para que
    esa importacion falle de inmediato por la via que ``xarray`` ya maneja. Al
    salir se restaura ``sys.modules``. Si ``distributed`` ya estaba importado
    (por ejemplo, porque hay un cliente activo) no se toca nada.

    Notes
    -----
    Es un parche a un entorno roto, no a ``xarray``. El arreglo de fondo es
    reparar el almacen de certificados de Windows, lo que tambien devolveria el
    funcionamiento normal a las descargas HTTPS (ver
    ``indices._ssl_context``). Mientras eso no ocurra, este bloque es lo que
    permite guardar cubos.
    """
    blocked = "distributed" in sys.modules or "dask.distributed" in sys.modules
    if blocked:
        yield
        return

    sys.modules["distributed"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.modules.pop("distributed", None)


def save_cube(
    ds: xr.Dataset | xr.DataArray,
    path: Path | str,
    overwrite: bool = False,
    encoding: dict | None = None,
    compress: bool = True,
) -> Path:
    """Escribir un cubo a NetCDF, con compresion y sin pisar archivos.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Cubo a guardar.
    path : Path or str
        Ruta de salida.
    overwrite : bool, optional
        Permitir sobrescribir un archivo existente. Por defecto ``False``:
        regenerar un cubo no debe destruir el anterior por accidente.
    encoding : dict or None, optional
        Codificacion por variable para ``to_netcdf``. Si es ``None`` y
        ``compress`` es ``True``, se aplica ``zlib`` nivel 4 a cada variable.
    compress : bool, optional
        Comprimir con ``zlib`` cuando no se entrega ``encoding``.

    Returns
    -------
    Path
        Ruta escrita.

    Raises
    ------
    FileExistsError
        Si el archivo existe y ``overwrite`` es ``False``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} ya existe; usar overwrite=True para reemplazarlo."
        )

    dataset = ds.to_dataset() if isinstance(ds, xr.DataArray) else ds

    if encoding is None and compress:
        encoding = {
            var: {"zlib": True, "complevel": 4} for var in dataset.data_vars
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with _without_distributed():
        dataset.to_netcdf(path, encoding=encoding)
    return path


def select_box(
    obj: xr.Dataset | xr.DataArray,
    box: dict[str, tuple[float, float]],
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    exclude_north_edge: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Recortar a una caja geografica declarada en ``config``.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Cubo o campo a recortar.
    box : dict
        Diccionario con claves ``'lat'`` y ``'lon'``, cada una un par
        ``(minimo, maximo)``.
    lat_dim, lon_dim : str, optional
        Nombres de las dimensiones espaciales.
    exclude_north_edge : bool, optional
        Excluir la fila de latitud del limite **norte** de la caja. Es
        necesario para Norte Chico: ``.sel(slice)`` es inclusivo en ambos
        extremos, asi que Norte Grande ``(-27.0, -18.0)`` y Norte Chico
        ``(-32.5, -27.0)`` compartirian la fila de -27.0 exacta y los 61
        pixeles de esa latitud entrarian en los dos analisis. Con este
        argumento el limite de 27 grados sur queda solo en Norte Grande.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Objeto recortado, perezosamente.

    Raises
    ------
    ValueError
        Si el recorte deja la caja vacia, lo que suele indicar limites
        invertidos o una caja fuera del dominio del cubo.
    """
    lat_min, lat_max = box["lat"]
    lon_min, lon_max = box["lon"]

    out = obj.sel(
        {lat_dim: slice(lat_min, lat_max), lon_dim: slice(lon_min, lon_max)}
    )

    if exclude_north_edge:
        latitudes = np.asarray(out[lat_dim].values)
        out = out.sel({lat_dim: latitudes[latitudes < lat_max]})

    if out.sizes.get(lat_dim, 0) == 0 or out.sizes.get(lon_dim, 0) == 0:
        raise ValueError(
            f"El recorte lat={box['lat']}, lon={box['lon']} deja la caja vacia."
        )
    return out


def select_depths(
    obj: xr.Dataset | xr.DataArray,
    depths: tuple[float, ...] | list[float],
    depth_dim: str = "depth",
    tol: float = 0.01,
    round_decimals: int | None = 3,
) -> xr.Dataset | xr.DataArray:
    """Seleccionar niveles verticales exigiendo que existan en el modelo.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Cubo o campo con dimension de profundidad.
    depths : sequence of float
        Profundidades buscadas, en metros (``config.DEPTHS_WORK``).
    depth_dim : str, optional
        Nombre de la dimension vertical.
    tol : float, optional
        Tolerancia maxima admitida, en metros, entre la profundidad pedida y el
        nivel real del modelo.
    round_decimals : int or None, optional
        Redondear la coordenada vertical a este numero de decimales y pasarla a
        ``float64``. Con ``None`` se conserva tal cual.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Objeto con solo esos niveles, en el orden pedido.

    Raises
    ------
    ValueError
        Si alguna profundidad no tiene un nivel del modelo dentro de ``tol``.

    Notes
    -----
    Se falla en vez de usar ``method='nearest'`` en silencio: un nivel
    sustituido sin aviso cambiaria la profundidad de un resultado reportado.

    El redondeo resuelve un problema practico: GLORYS guarda ``depth`` en
    ``float32``, donde el nivel de superficie vale 0.49399998784, asi que
    ``.sel(depth=0.494)`` con el literal de ``config.DEPTHS_WORK`` falla con
    ``KeyError`` y obliga a arrastrar ``method='nearest'`` por todo el
    analisis. Se redondea el **valor real del modelo**, nunca el pedido: si se
    solicita 100 m con una tolerancia amplia, la coordenada queda en 92.326 y
    no en 100, de modo que la etiqueta nunca miente sobre el dato. Los valores
    originales quedan en el atributo ``depth_original`` de la coordenada.
    """
    available = np.asarray(obj[depth_dim].values, dtype=float)
    chosen: list[float] = []

    for target in depths:
        distance = np.abs(available - float(target))
        i = int(distance.argmin())
        if distance[i] > tol:
            raise ValueError(
                f"La profundidad {target} m no existe en el cubo "
                f"(el nivel mas cercano es {available[i]:.3f} m, "
                f"a {distance[i]:.3f} m > tol={tol})."
            )
        chosen.append(float(available[i]))

    out = obj.sel({depth_dim: chosen})

    if round_decimals is not None:
        exactos = np.asarray(out[depth_dim].values, dtype=float)
        out = out.assign_coords(
            {depth_dim: np.round(exactos, round_decimals).astype(np.float64)}
        )
        out[depth_dim].attrs = {
            **obj[depth_dim].attrs,
            "depth_original": ", ".join(f"{v:.8f}" for v in exactos),
            "note": (
                "coordenada redondeada para que .sel(depth=...) funcione con "
                "los valores de config.DEPTHS_WORK"
            ),
        }

    return out


def water_mask(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Mascara booleana de los pixeles con dato en todo el periodo.

    En GLORYS12 la mascara de tierra es constante en el tiempo y depende de la
    profundidad: un pixel costero puede tener dato en superficie y ser tierra a
    155 m. Esta funcion no asume esa constancia: exige dato en **todos** los
    tiempos, asi que un pixel con cobertura parcial queda fuera.

    Parameters
    ----------
    da : xarray.DataArray
        Campo con dimension temporal.
    time_dim : str, optional
        Nombre de la dimension temporal.

    Returns
    -------
    xarray.DataArray
        Mascara sin la dimension temporal: ``True`` donde hay agua con serie
        completa.
    """
    return da.notnull().all(dim=time_dim)


def area_weights(
    obj: xr.Dataset | xr.DataArray, lat_dim: str = "latitude"
) -> xr.DataArray:
    """Pesos ``sqrt(cos(lat))`` para el analisis de modos.

    En una malla regular en latitud y longitud, el area de la celda decrece con
    ``cos(lat)``. Como los EOF se calculan sobre la **matriz de datos** y la
    varianza es cuadratica, el peso que hay que aplicar a los datos es la raiz
    de ese factor: asi la varianza resultante queda ponderada por area. Ver
    North et al. (1982) y la practica habitual del paquete ``eofs``.

    A las latitudes de este estudio (18 a 32.5 grados sur) el peso varia poco,
    pero se aplica de forma consistente en todos los EOF, EEOF y CEOF para que
    las varianzas explicadas sean comparables entre cajas.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Objeto del que se toma la coordenada de latitud.
    lat_dim : str, optional
        Nombre de la coordenada de latitud, en grados.

    Returns
    -------
    xarray.DataArray
        Pesos con dimension de latitud, en ``[0, 1]``.
    """
    latitude = obj[lat_dim]
    cos_lat = np.cos(np.deg2rad(latitude))
    weights = np.sqrt(cos_lat.clip(min=0.0))
    weights.name = "area_weight"
    weights.attrs = {
        "long_name": "sqrt(cos(latitud))",
        "description": (
            "peso por area para analisis de modos; la raiz compensa que la "
            "varianza sea cuadratica en los datos"
        ),
    }
    return weights


def apply_area_weights(
    da: xr.DataArray, lat_dim: str = "latitude"
) -> xr.DataArray:
    """Multiplicar un campo por los pesos ``sqrt(cos(lat))``.

    Parameters
    ----------
    da : xarray.DataArray
        Campo a ponderar.
    lat_dim : str, optional
        Nombre de la coordenada de latitud.

    Returns
    -------
    xarray.DataArray
        Campo ponderado, con ``area_weighting`` anotado en ``attrs``.
    """
    out = da * area_weights(da, lat_dim)
    out.name = da.name
    out.attrs = {**da.attrs, "area_weighting": "sqrt_coslat"}
    return out


def flatten_space(
    da: xr.DataArray, time_dim: str = "time", drop_land: bool = True
) -> tuple[np.ndarray, xr.DataArray]:
    """Aplanar las dimensiones espaciales a una matriz ``(tiempo, espacio)``.

    Es la forma que esperan el SVD y los metodos de descomposicion: cada fila
    es un instante y cada columna un pixel.

    Parameters
    ----------
    da : xarray.DataArray
        Campo con dimension temporal y una o mas dimensiones espaciales.
    time_dim : str, optional
        Nombre de la dimension temporal.
    drop_land : bool, optional
        Descartar las columnas sin serie completa (tierra). Recomendado: un
        SVD sobre columnas con NaN no esta definido.

    Returns
    -------
    matrix : numpy.ndarray
        Matriz de forma ``(n_tiempos, n_pixeles)``.
    index : xarray.DataArray
        El objeto apilado con su ``MultiIndex`` espacial, necesario para
        reconstruir los mapas con :func:`unflatten_space`.

    Notes
    -----
    No se submuestrean pixeles: los scripts viejos bajaban a 1200-1500 pixeles
    al azar solo por velocidad, lo que esta prohibido en resultados
    reportables. Norte Grande tiene unos 4800 pixeles de agua en superficie,
    perfectamente manejable.
    """
    space_dims = [d for d in da.dims if d != time_dim]
    stacked = da.stack(pixel=space_dims).transpose(time_dim, "pixel")

    if drop_land:
        keep = np.isfinite(stacked.values).all(axis=0)
        stacked = stacked.isel(pixel=np.flatnonzero(keep))

    return stacked.values, stacked


def unflatten_space(
    values: np.ndarray, index: xr.DataArray, name: str | None = None
) -> xr.DataArray:
    """Devolver una matriz aplanada a su forma espacial original.

    Parameters
    ----------
    values : numpy.ndarray
        Matriz ``(tiempo, espacio)`` o vector ``(espacio,)``, p. ej. un patron
        espacial de EOF.
    index : xarray.DataArray
        El objeto apilado que devolvio :func:`flatten_space`.
    name : str or None, optional
        Nombre del campo resultante.

    Returns
    -------
    xarray.DataArray
        Campo con las dimensiones espaciales restauradas; los pixeles
        descartados como tierra reaparecen como NaN.
    """
    if values.ndim == 1:
        template = xr.DataArray(
            values, coords={"pixel": index["pixel"]}, dims=["pixel"], name=name
        )
    else:
        template = xr.DataArray(
            values,
            coords={index.dims[0]: index[index.dims[0]], "pixel": index["pixel"]},
            dims=[index.dims[0], "pixel"],
            name=name,
        )
    return template.unstack("pixel")
