"""Piezas del poster: nombres de componentes, anomalias congeladas y proyeccion.

Reune lo que comparten los scripts 04 (EOF combinado y rotaciones), 06 (figuras)
y 07 (pagina web): nombrar cada componente rotado por su perfil vertical,
orientar y estandarizar sus series, y proyectar meses nuevos sobre los ejes ya
estimados sin reajustar nada ("congelados").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

from .anomalies import _DAYS_PER_YEAR, design_matrix
from .cube import apply_area_weights, unflatten_space

#: Perfiles de R2 por capa (0.5, 25, 92, 156 m) que definen el nombre de cada
#: componente localizado de una rotacion.
PERFILES: dict[str, list[float]] = {
    "Superficie": [1.0, 0.5, 0.0, 0.0],
    "Fondo": [0.0, 0.0, 0.3, 1.0],
    "92 m": [0.0, 0.3, 1.0, 0.0],
    "Columna": [1.0, 1.0, 1.0, 1.0],
}


def name_by_profile(r2: np.ndarray, profiles: Mapping[str, Sequence[float]] = PERFILES) -> list[str]:
    """Nombrar componentes por el perfil de referencia mas parecido a su R2 por capa.

    Parameters
    ----------
    r2 : numpy.ndarray
        R2 por capa, ``(n_componentes, n_capas)``, con ``n_componentes <= len(profiles)``.
    profiles : mapping
        Nombre -> perfil de referencia (misma longitud que las capas).

    Returns
    -------
    list of str
        Un nombre distinto por componente: asignacion uno a uno que maximiza la
        suma de cosenos entre perfiles (algoritmo hungaro).
    """
    r2 = np.atleast_2d(np.asarray(r2, dtype=float))
    ref = np.asarray(list(profiles.values()), dtype=float)
    if r2.shape[0] > ref.shape[0]:
        raise ValueError("Hay mas componentes que perfiles de referencia.")
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    obs = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
    filas, cols = linear_sum_assignment(-(obs @ ref.T))
    nombres = list(profiles)
    out = [""] * r2.shape[0]
    for i, j in zip(filas, cols):
        out[i] = nombres[j]
    return out


def frozen_anomalies(hist: xr.DataArray, new: xr.DataArray, detrend: bool = True) -> xr.DataArray:
    """Anomalias de meses nuevos con el ajuste por pixel estimado en el periodo historico.

    Se ajusta, por pixel, constante + tendencia + 12 efectos mensuales sobre
    ``hist`` (igual que :func:`enso_chile.anomalies.fit_anomalies`) y se resta a
    ``new`` el valor que ese ajuste predice para sus fechas. La tendencia se
    extrapola; nada se reestima con los meses nuevos.

    Parameters
    ----------
    hist, new : xarray.DataArray
        Campos ``(time, ...)`` con la misma grilla espacial.
    detrend : bool, optional
        Incluir la tendencia lineal en el ajuste.

    Returns
    -------
    xarray.DataArray
        Anomalias de ``new``, con sus coordenadas.
    """
    Xh, _ = design_matrix(hist.time, detrend=detrend, seasonal="monthly")
    t0 = pd.Timestamp(hist.time.values[0])

    def anios(t) -> np.ndarray:
        return (pd.DatetimeIndex(t) - t0).days.to_numpy(float) / _DAYS_PER_YEAR

    tn = pd.DatetimeIndex(new.time.values)
    columnas = [np.ones(len(tn))]
    if detrend:
        columnas.append(anios(tn) - anios(hist.time.values).mean())
    Xn = np.column_stack(columnas + [(tn.month == m).astype(float) for m in range(2, 13)])

    hv = hist.values.reshape(len(hist.time), -1)
    agua = np.isfinite(hv).all(0)
    beta = np.linalg.lstsq(Xh, hv[:, agua], rcond=None)[0]
    out = np.full((len(tn), hv.shape[1]), np.nan)
    out[:, agua] = new.values.reshape(len(tn), -1)[:, agua] - Xn @ beta
    return xr.DataArray(out.reshape(new.shape), coords=new.coords, dims=new.dims)


def prefilter_blocks(field: xr.DataArray, depths: Sequence[float], results, block_scales: np.ndarray,
                     n_pcs: int) -> np.ndarray:
    """Llevar un campo de anomalias al espacio del EOF combinado (bloques de PC por capa).

    Aplica a ``field`` exactamente lo mismo que al periodo de ajuste: ponderacion
    ``sqrt(cos(lat))``, resta de la media de cada pixel, proyeccion sobre las
    ``n_pcs`` EOF de cada capa y division por la escala de cada bloque.

    Parameters
    ----------
    field : xarray.DataArray
        Anomalias ``(time, depth, latitude, longitude)``.
    depths : sequence of float
        Capas, en el orden de ``results``.
    results : sequence of EOFResult
        EOF por capa del ajuste.
    block_scales : numpy.ndarray
        Escala de cada bloque (``CombinedEOFResult.block_scales``).
    n_pcs : int
        PC por capa.

    Returns
    -------
    numpy.ndarray
        Matriz ``(T, n_capas * n_pcs)``.
    """
    bloques = []
    for d, (z, r) in enumerate(zip(depths, results)):
        x = apply_area_weights(field.sel(depth=z), "latitude")
        x = x.stack(pixel=["latitude", "longitude"]).transpose("time", "pixel")
        x = x.sel(pixel=r.index.pixel).values
        bloques.append(((x - r.mean) @ r.eofs[:n_pcs].T) / block_scales[d])
    return np.hstack(bloques)


def project_scores(Z: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Puntajes por minimos cuadrados ``Z A (A'A)^-1`` sobre cargas fijas.

    Con las cargas del EOF sin rotar reproduce sus PC; con cargas rotadas
    ``A T`` da las series rotadas. Sirve para meses que no entraron al ajuste.
    """
    A = np.asarray(loadings, dtype=float)
    return np.asarray(Z, dtype=float) @ A @ np.linalg.inv(A.T @ A)


def loading_maps(loadings: np.ndarray, results, block_scales: np.ndarray, n_pcs: int
                 ) -> list[list[xr.DataArray]]:
    """Mapas de carga de cada componente en cada capa, en °C por sd de su serie.

    Parameters
    ----------
    loadings : numpy.ndarray
        Cargas por canal ``(n_capas * n_pcs, k)`` (sin rotar o ``A T``).
    results : sequence of EOFResult
        EOF por capa del ajuste.
    block_scales : numpy.ndarray
        Escala de cada bloque.
    n_pcs : int
        PC por capa.

    Returns
    -------
    list of list of xarray.DataArray
        ``maps[j][d]``: componente ``j`` en la capa ``d``, sin la ponderacion por area.
    """
    out = []
    for j in range(loadings.shape[1]):
        capas = []
        for d, r in enumerate(results):
            a = loadings[d * n_pcs:(d + 1) * n_pcs, j]
            m = unflatten_space(block_scales[d] * r.eofs[:n_pcs].T @ a, r.index) / r.weights
            capas.append(m.sortby("latitude").sortby("longitude"))
        out.append(capas)
    return out


def orient(maps: Sequence[xr.DataArray], r2: np.ndarray) -> float:
    """Signo que deja positiva la carga media en la capa de mayor R2 (convencion del poster)."""
    zmax = int(np.argmax(r2))
    return -1.0 if float(maps[zmax].mean()) < 0 else 1.0
