"""Descarga y parseo de indices climaticos, y consolidacion en una tabla mensual.

Fuentes (todas de acceso publico, sin credenciales):

===========  =========================================================
Indice       Fuente
===========  =========================================================
Nino 1+2,
3, 4, 3.4    CPC/NOAA ``sstoi.indices``
ONI          CPC/NOAA ``oni.ascii.txt`` (medias trimestrales moviles)
PDO          NCEI/NOAA ERSST v6
SOI          CPC/NOAA ``soi`` (bloque estandarizado)
OLR          CPC/NOAA ``olr``
MEI.v2       PSL/NOAA ``meiv2.data`` (bimestral, anclado al mes final)
SAM / AAO    CPC/NOAA indice AAO mensual
===========  =========================================================

Notes
-----
Dos correcciones frente a los scripts originales:

1. **Eje temporal.** Los scripts construian las fechas con
   ``pd.date_range(start=..., periods=len(serie), freq="ME")`` *despues* de
   ``DataFrame.stack()``. Como ``stack`` descarta los NaN, cualquier mes
   faltante en el interior de la serie desplazaba todas las fechas
   posteriores. Aqui la fecha se construye a partir del par (anio, mes) de
   cada celda, asi que un hueco queda como hueco.
2. **Convencion de fecha.** Se usa el **inicio** de mes (``YYYY-MM-01``), igual
   que el eje ``time`` de GLORYS12, en lugar del fin de mes del CSV heredado
   (``1993-01-31 23:59:59.999999999``), que obligaba a reindexar para cruzar
   indices con el cubo.

Los `skiprows` fijos de los scripts viejos se reemplazan por deteccion de la
primera fila con un anio de cuatro digitos, de modo que el parseo no se rompe
si la NOAA agrega o quita lineas de encabezado.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "MONTHS",
    "MISSING_CODES",
    "fetch_text",
    "load_nino_regions",
    "load_oni",
    "load_pdo",
    "load_soi",
    "load_olr",
    "load_mei",
    "load_sam",
    "build_master_table",
]

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

MONTHS: Final[tuple[str, ...]] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

#: Codigos de dato faltante que usan los archivos de la NOAA.
MISSING_CODES: Final[tuple[float, ...]] = (
    -99.0, -99.9, -99.99, -999.0, -999.9, -999.00, -9999.0, 99.99,
)

URLS: Final[dict[str, str]] = {
    "nino": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "pdo": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/v6/index/ersst.v6.pdo.dat",
    "soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
    "olr": "https://www.cpc.ncep.noaa.gov/data/indices/olr",
    "mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "sam": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/"
        "aao/monthly.aao.index.b79.current.ascii"
    ),
}

_USER_AGENT: Final[str] = "enso-chile-norte/0.1 (research; python-urllib)"

# --------------------------------------------------------------------------- #
# Utilidades de descarga y parseo
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Contexto TLS que verifica contra el paquete de certificados de ``certifi``.

    En esta maquina el almacen de certificados que OpenSSL toma por defecto
    (``C:/Program Files/Common Files/ssl/cert.pem``) esta corrupto y cualquier
    descarga HTTPS falla con ``SSLError: [ASN1: NOT_ENOUGH_DATA]``. Con el
    paquete de ``certifi`` la verificacion funciona con normalidad. Si
    ``certifi`` no estuviera instalado se usa el contexto por defecto, que es
    el comportamiento habitual en otros equipos.

    Returns
    -------
    ssl.SSLContext
        Contexto con verificacion de certificado activa. **No** se desactiva
        la verificacion en ningun caso.
    """
    try:
        import certifi
    except ImportError:  # pragma: no cover - depende del entorno
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_text(url: str, cache_path: Path | None = None, refresh: bool = False) -> str:
    """Descargar un archivo de texto, con cache en disco opcional.

    Parameters
    ----------
    url : str
        Direccion del archivo de texto plano.
    cache_path : Path or None, optional
        Si se indica, el contenido se guarda ahi y en llamadas posteriores se
        lee del disco en lugar de la red (descarga idempotente).
    refresh : bool, optional
        Forzar la descarga aunque exista la cache.

    Returns
    -------
    str
        Contenido del archivo.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8", errors="replace")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(
        request, timeout=120, context=_ssl_context()
    ) as response:
        text = response.read().decode("utf-8", errors="replace")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")

    return text


def _clean_missing(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reemplazar por NaN los codigos de dato faltante de la NOAA."""
    return values.replace(list(MISSING_CODES), np.nan)


def _first_data_line(lines: list[str], n_fields_min: int) -> int:
    """Indice de la primera linea cuyo primer campo es un anio de 4 digitos.

    Parameters
    ----------
    lines : list of str
        Lineas del archivo.
    n_fields_min : int
        Numero minimo de campos que debe tener la linea para aceptarla. Evita
        confundir una nota al pie que empiece con un anio con una fila de datos.

    Returns
    -------
    int
        Indice de la linea, o ``len(lines)`` si no se encontro ninguna.
    """
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) < n_fields_min:
            continue
        head = fields[0]
        if len(head) == 4 and head.isdigit() and 1800 <= int(head) <= 2200:
            return i
    return len(lines)


def _parse_year_month_matrix(text: str, name: str) -> pd.Series:
    """Parsear un archivo con formato ``anio`` + 12 columnas mensuales.

    La fecha de cada celda se construye desde el par (anio, mes) de su
    posicion, nunca por ``date_range``, de modo que los meses ausentes quedan
    como huecos en lugar de desplazar la serie.

    Parameters
    ----------
    text : str
        Contenido del archivo.
    name : str
        Nombre de la serie resultante.

    Returns
    -------
    pandas.Series
        Serie mensual indexada al inicio de cada mes, ordenada y sin duplicados.
    """
    lines = text.splitlines()
    start = _first_data_line(lines, n_fields_min=13)
    rows: list[tuple[int, int, float]] = []

    for line in lines[start:]:
        fields = line.split()
        if len(fields) < 13:
            continue
        head = fields[0]
        if not (len(head) == 4 and head.isdigit()):
            continue
        year = int(head)
        for month, raw in enumerate(fields[1:13], start=1):
            try:
                value = float(raw)
            except ValueError:
                value = np.nan
            rows.append((year, month, value))

    if not rows:
        raise ValueError(f"No se pudo parsear ninguna fila de datos para {name!r}.")

    frame = pd.DataFrame(rows, columns=["year", "month", name])
    index = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=1)
    )
    series = pd.Series(frame[name].to_numpy(), index=index, name=name)
    series = _clean_missing(series)
    return series[~series.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- #
# Parsers por indice
# --------------------------------------------------------------------------- #


def load_nino_regions(
    cache_dir: Path | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Anomalias de SST de las regiones Nino 1+2, 3, 4 y 3.4.

    El archivo ``sstoi.indices`` trae, por mes, la SST absoluta y su anomalia
    para cada region, en ese orden; ``pandas`` desambigua los nombres repetidos
    como ``ANOM``, ``ANOM.1``, ``ANOM.2``, ``ANOM.3``.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.DataFrame
        Columnas ``nino12``, ``nino3``, ``nino4``, ``nino34`` (anomalias, en
        grados Celsius), indexadas al inicio de cada mes.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.126-146.
    """
    cache = None if cache_dir is None else cache_dir / "sstoi.indices"
    text = fetch_text(URLS["nino"], cache, refresh)
    raw = pd.read_csv(io.StringIO(text), sep=r"\s+")

    index = pd.to_datetime(dict(year=raw["YR"], month=raw["MON"], day=1))
    frame = pd.DataFrame(
        {
            "nino12": raw["ANOM"].to_numpy(),
            "nino3": raw["ANOM.1"].to_numpy(),
            "nino4": raw["ANOM.2"].to_numpy(),
            "nino34": raw["ANOM.3"].to_numpy(),
        },
        index=index,
    )
    return _clean_missing(frame).sort_index()


def load_oni(cache_dir: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Oceanic Nino Index del CPC, con el eje temporal en meses.

    El ONI se publica como medias moviles de tres meses etiquetadas por
    temporada (``DJF``, ``JFM``, ...). Cada valor se asigna al **mes central**
    de su trimestre, que es la convencion del CPC: ``DJF`` del anio Y
    corresponde a enero de Y.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.DataFrame
        Indexado al inicio del mes central, con columnas ``season`` (etiqueta
        trimestral), ``sst`` (media absoluta) y ``oni`` (anomalia).

    Notes
    -----
    Ningun script original leia este archivo: los eventos se listaban a mano.
    """
    cache = None if cache_dir is None else cache_dir / "oni.ascii.txt"
    text = fetch_text(URLS["oni"], cache, refresh)
    raw = pd.read_csv(io.StringIO(text), sep=r"\s+")
    raw.columns = [str(c).strip().upper() for c in raw.columns]

    seasons = [
        "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
        "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
    ]
    #: Mes central de cada temporada: DJF -> enero (1), ..., NDJ -> diciembre (12).
    center_month = {season: i + 1 for i, season in enumerate(seasons)}

    month = raw["SEAS"].str.strip().str.upper().map(center_month)
    if month.isna().any():
        unknown = sorted(set(raw.loc[month.isna(), "SEAS"]))
        raise ValueError(f"Temporadas no reconocidas en oni.ascii.txt: {unknown}")

    index = pd.to_datetime(dict(year=raw["YR"], month=month.astype(int), day=1))
    frame = pd.DataFrame(
        {
            "season": raw["SEAS"].to_numpy(),
            "sst": pd.to_numeric(raw["TOTAL"], errors="coerce").to_numpy(),
            "oni": pd.to_numeric(raw["ANOM"], errors="coerce").to_numpy(),
        },
        index=index,
    )
    frame["oni"] = _clean_missing(frame["oni"])
    return frame.sort_index()


def load_pdo(cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """Pacific Decadal Oscillation (ERSST v6, NCEI).

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.Series
        Serie mensual ``pdo``.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.148-190.
    """
    cache = None if cache_dir is None else cache_dir / "ersst.v6.pdo.dat"
    return _parse_year_month_matrix(fetch_text(URLS["pdo"], cache, refresh), "pdo")


def load_soi(cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """Southern Oscillation Index estandarizado (CPC).

    El archivo trae dos bloques: anomalias de presion y valores
    estandarizados. Se usa el segundo, localizado por la etiqueta ``STAND`` en
    lugar del ``skiprows=87`` de respaldo del script original.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.Series
        Serie mensual ``soi``.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.191-256.
    """
    cache = None if cache_dir is None else cache_dir / "soi.txt"
    text = fetch_text(URLS["soi"], cache, refresh)
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if "STAND" in line.upper():
            start = i
            break
    if start is None:
        raise ValueError(
            "No se encontro el bloque estandarizado ('STAND') en el archivo SOI."
        )

    return _parse_year_month_matrix("\n".join(lines[start:]), "soi")


def load_olr(cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """Anomalia de radiacion de onda larga saliente, en W/m2 (CPC).

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.Series
        Serie mensual ``olr``.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.257-306. El script saltaba 20 lineas
    fijas para caer en un bloque de anios completos; aqui se detecta la primera
    fila con 13 campos, asi que los anios incompletos de los setenta se
    conservan con NaN en vez de descartarse en bloque.
    """
    cache = None if cache_dir is None else cache_dir / "olr.txt"
    return _parse_year_month_matrix(fetch_text(URLS["olr"], cache, refresh), "olr")


def load_mei(cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """MEI.v2, indice ENSO multivariado (PSL/NOAA).

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.Series
        Serie mensual ``mei``.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.308-326.

    El MEI.v2 es **bimestral**: las 12 columnas son los pares deslizantes
    DJ, JF, FM, ..., ND. Al asignar la columna *k* al mes *k* se ancla cada
    valor al **segundo** mes del par (DJ -> enero), igual que hacia el script
    original. Es una serie suavizada en dos meses, no una medicion mensual
    independiente: no mezclar su suavizado con el de las anomalias del cubo.
    """
    cache = None if cache_dir is None else cache_dir / "meiv2.data"
    return _parse_year_month_matrix(fetch_text(URLS["mei"], cache, refresh), "mei")


def load_sam(cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """Southern Annular Mode / Antarctic Oscillation mensual (CPC).

    El archivo viene en formato largo: anio, mes, valor.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear el archivo descargado.
    refresh : bool, optional
        Forzar la descarga.

    Returns
    -------
    pandas.Series
        Serie mensual ``sam``.

    Notes
    -----
    LEGACY: ``Glorys12_and_Index.py`` l.328-335.
    """
    cache = None if cache_dir is None else cache_dir / "monthly.aao.index.ascii"
    text = fetch_text(URLS["sam"], cache, refresh)
    raw = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    raw = raw.iloc[:, :3]
    raw.columns = ["year", "month", "sam"]
    raw = raw[pd.to_numeric(raw["year"], errors="coerce").notna()]

    index = pd.to_datetime(
        dict(year=raw["year"].astype(int), month=raw["month"].astype(int), day=1)
    )
    series = pd.Series(
        pd.to_numeric(raw["sam"], errors="coerce").to_numpy(), index=index, name="sam"
    )
    return _clean_missing(series).sort_index()


# --------------------------------------------------------------------------- #
# Consolidacion
# --------------------------------------------------------------------------- #


def build_master_table(
    cache_dir: Path | None = None,
    refresh: bool = False,
    period: tuple[str, str] | None = None,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Consolidar todos los indices en una tabla mensual unica.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Carpeta donde cachear los archivos descargados.
    refresh : bool, optional
        Forzar la descarga de todos los archivos.
    period : tuple of str or None, optional
        Par ``(inicio, fin)`` para recortar la tabla, p. ej.
        ``config.PERIOD_FULL``. Si es ``None``, se devuelve el rango completo.
    out_csv : Path or None, optional
        Si se indica, la tabla se escribe ahi en CSV.

    Returns
    -------
    pandas.DataFrame
        Indexado por ``time`` al inicio de cada mes, con columnas ``nino12``,
        ``nino3``, ``nino4``, ``nino34``, ``oni``, ``pdo``, ``soi``, ``olr``,
        ``mei``, ``sam``. El indice es un rango mensual continuo: los indices
        con menos cobertura quedan como NaN en vez de desalinearse.
    """
    nino = load_nino_regions(cache_dir, refresh)
    oni = load_oni(cache_dir, refresh)[["oni"]]
    others = [
        load_pdo(cache_dir, refresh),
        load_soi(cache_dir, refresh),
        load_olr(cache_dir, refresh),
        load_mei(cache_dir, refresh),
        load_sam(cache_dir, refresh),
    ]

    table = pd.concat([nino, oni, *others], axis=1)
    full_index = pd.date_range(table.index.min(), table.index.max(), freq="MS")
    table = table.reindex(full_index)
    table.index.name = "time"

    if period is not None:
        table = table.loc[period[0] : period[1]]

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, float_format="%.4f")

    return table


def ec_indices(nino12: pd.Series, nino4: pd.Series) -> pd.DataFrame:
    """Indices E y C de Takahashi et al. (2011), por su aproximacion lineal.

    ``C = 1.7 Nino4 - 0.1 Nino1+2`` (ec. 3, R2 = 98 %) y
    ``E = Nino1+2 - 0.5 Nino4`` (ec. 4, R2 = 95 %), verificadas en el articulo.
    E resume la variabilidad del Pacifico oriental (Ninos extraordinarios) y C la
    del central; son casi incorrelacionados por construccion.

    Parameters
    ----------
    nino12, nino4 : pandas.Series
        Anomalias de TSM (°C) en Nino 1+2 y Nino 4, con el mismo tratamiento
        (por ejemplo, ambas con la media movil de 3 meses del ICEN).

    Returns
    -------
    pandas.DataFrame
        Columnas ``E`` y ``C`` sobre los meses comunes a ambas series.

    Notes
    -----
    Los indices exactos son una rotacion de 45° de las dos primeras PC de la TSM
    de HadISST 1.1 en 10 S - 10 N (base 1979-2009, PC normalizadas y suavizadas
    1-2-1): ``C = (PC1 + PC2) / sqrt(2)``, ``E = (PC1 - PC2) / sqrt(2)``. Frente a
    la serie oficial del IGP (ERSST v3b, 1993-2020) la aproximacion da r = 0.94 (E)
    y 0.95 (C) con los indices Nino de este proyecto.
    """
    df = pd.concat([nino12.rename("n12"), nino4.rename("n4")], axis=1).dropna()
    return pd.DataFrame({"E": df["n12"] - 0.5 * df["n4"], "C": 1.7 * df["n4"] - 0.1 * df["n12"]})
