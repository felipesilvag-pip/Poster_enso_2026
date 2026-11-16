"""Tests de :mod:`enso_chile.poster`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import config
from enso_chile.anomalies import _DAYS_PER_YEAR, fit_anomalies
from enso_chile.eof import combined_eof_from_results, eof_from_dataarray
from enso_chile.poster import (PERFILES, frozen_anomalies, loading_maps, name_by_profile, orient,
                               prefilter_blocks, project_scores)

DEPTHS = (0.5, 25.0, 92.0, 156.0)


def _cubo(n_t: int = 80, seed: int = config.RANDOM_SEED) -> xr.DataArray:
    """Cubo sintetico (time, depth, latitude, longitude) con un pixel de tierra."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2000-01-01", periods=n_t, freq="MS")
    lat, lon = np.linspace(-27, -18, 6), np.linspace(-74, -69, 5)
    senal = np.sin(2 * np.pi * np.arange(n_t) / 40)
    datos = rng.normal(size=(n_t, len(DEPTHS), lat.size, lon.size))
    datos += 3 * senal[:, None, None, None] * np.linspace(1, 0.3, len(DEPTHS))[None, :, None, None]
    datos[:, :, 0, -1] = np.nan
    return xr.DataArray(datos, dims=("time", "depth", "latitude", "longitude"),
                        coords={"time": t, "depth": list(DEPTHS), "latitude": lat, "longitude": lon})


def test_name_by_profile_recupera_perfiles_permutados() -> None:
    nombres = ["Fondo", "Columna", "Superficie", "92 m"]
    r2 = np.array([PERFILES[n] for n in nombres]) * 37.0 + 0.01
    assert name_by_profile(r2) == nombres


def test_name_by_profile_tolera_ruido_y_subconjunto() -> None:
    r2 = np.array([[50.5, 26.1, 2.0, 1.5], [2.0, 2.5, 3.0, 16.8], [1.7, 2.2, 15.7, 1.5]])
    assert name_by_profile(r2) == ["Superficie", "Fondo", "92 m"]


def test_name_by_profile_rechaza_mas_componentes_que_perfiles() -> None:
    with pytest.raises(ValueError):
        name_by_profile(np.ones((5, 4)))


@pytest.mark.parametrize("detrend", [True, False])
def test_frozen_anomalies_sobre_el_mismo_periodo_es_el_ajuste(detrend: bool) -> None:
    campo = _cubo().isel(depth=0) + 10.0
    esperado = fit_anomalies(campo, detrend, "monthly", "time")
    obtenido = frozen_anomalies(campo, campo, detrend=detrend)
    np.testing.assert_allclose(obtenido.values, esperado.values, atol=1e-10, equal_nan=True)


def test_frozen_anomalies_extrapola_ciclo_y_tendencia() -> None:
    t = pd.date_range("2000-01-01", periods=126, freq="MS")
    anios = (t - t[0]).days.to_numpy(float) / _DAYS_PER_YEAR      # el mismo tiempo del ajuste
    valores = 2.0 + 0.5 * anios + (t.month.to_numpy() - 1.0)       # tendencia + ciclo exactos
    campo = xr.DataArray(valores[:, None], dims=("time", "x"), coords={"time": t})
    serie, nuevo = campo.isel(time=slice(0, 120)), campo.isel(time=slice(120, None))
    np.testing.assert_allclose(frozen_anomalies(serie, nuevo).values, 0.0, atol=1e-6)


def test_project_scores_recupera_puntajes() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    A, F = rng.normal(size=(24, 4)), rng.normal(size=(50, 4))
    np.testing.assert_allclose(project_scores(F @ A.T, A), F, atol=1e-10)


def test_proyeccion_del_periodo_de_ajuste_reproduce_las_pc() -> None:
    cubo, k_pcs, k = _cubo(), 3, 2
    res = [eof_from_dataarray(cubo.sel(depth=z), n_modes=k_pcs, scaling="orthonormal") for z in DEPTHS]
    cmb = combined_eof_from_results(res, k_pcs)
    Z = prefilter_blocks(cubo, DEPTHS, res, cmb.block_scales, k_pcs)
    np.testing.assert_allclose(project_scores(Z, cmb.loadings(k)), cmb.scores(k), atol=1e-8)


def test_loading_maps_y_orient() -> None:
    cubo, k_pcs = _cubo(), 3
    res = [eof_from_dataarray(cubo.sel(depth=z), n_modes=k_pcs, scaling="orthonormal") for z in DEPTHS]
    cmb = combined_eof_from_results(res, k_pcs)
    mapas = loading_maps(cmb.loadings(2), res, cmb.block_scales, k_pcs)
    assert len(mapas) == 2 and len(mapas[0]) == len(DEPTHS)
    assert mapas[0][0].shape == (6, 5) and np.isnan(mapas[0][0].values[0, -1])
    r2 = cmb.r2_by_block(cmb.loadings(2))[0]
    s = orient(mapas[0], r2)
    assert float((s * mapas[0][int(np.argmax(r2))]).mean()) > 0
