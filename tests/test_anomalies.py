"""Tests del ajuste de anomalias.

La estrategia es construir series sinteticas con constante, tendencia, ciclo
anual y una senal conocida, y verificar que el ajuste recupera los parametros y
deja la senal intacta en el residuo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from enso_chile.anomalies import (
    build_anomaly_cube,
    design_matrix,
    fit_anomalies,
    smooth_3m,
)

SEED = 42


def _time(n: int = 360) -> pd.DatetimeIndex:
    return pd.date_range("1993-01-01", periods=n, freq="MS")


def _synthetic_cube(
    n_time: int = 360,
    mean: float = 15.0,
    trend_per_year: float = 0.02,
    amplitude: float = 3.0,
    noise: float = 0.0,
    n_lat: int = 4,
    n_lon: int = 3,
) -> xr.DataArray:
    """Cubo sintetico con constante, tendencia y ciclo anual conocidos."""
    time = _time(n_time)
    years = (time - time[0]).days.to_numpy(dtype=float) / 365.2425
    season = amplitude * np.cos(2 * np.pi * (time.month.to_numpy() - 1) / 12)
    series = mean + trend_per_year * years + season

    rng = np.random.default_rng(SEED)
    values = np.broadcast_to(
        series[:, None, None], (n_time, n_lat, n_lon)
    ).astype(float).copy()
    if noise:
        values += rng.normal(0.0, noise, values.shape)

    return xr.DataArray(
        values,
        coords={
            "time": time,
            "latitude": np.linspace(-27.0, -18.0, n_lat),
            "longitude": np.linspace(-74.0, -69.0, n_lon),
        },
        dims=["time", "latitude", "longitude"],
        name="thetao",
    )


# --------------------------------------------------------------------------- #
# Matriz de diseno
# --------------------------------------------------------------------------- #


def test_matriz_mensual_tiene_13_columnas() -> None:
    """Constante + tendencia + 11 indicadoras (enero es la referencia)."""
    design, names = design_matrix(_time(), detrend=True, seasonal="monthly")

    assert design.shape == (360, 13)
    assert names[:2] == ["const", "trend"]
    assert "month_01" not in names
    assert names[-1] == "month_12"


def test_matriz_sin_detrend_omite_la_tendencia() -> None:
    """La version de sensibilidad no incluye la columna de tendencia."""
    _, names = design_matrix(_time(), detrend=False, seasonal="monthly")
    assert "trend" not in names


def test_matriz_armonica_tiene_6_columnas() -> None:
    """Constante + tendencia + armonicos anual y semianual."""
    design, names = design_matrix(_time(), detrend=True, seasonal="harmonic")

    assert design.shape == (360, 6)
    assert names == ["const", "trend", "cos_1", "sin_1", "cos_2", "sin_2"]


def test_matriz_mensual_tiene_rango_completo() -> None:
    """Omitir enero evita la colinealidad exacta con la constante."""
    design, _ = design_matrix(_time(), detrend=True, seasonal="monthly")
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_tiempo_centrado_descorrelaciona_constante_y_tendencia() -> None:
    """La columna de tendencia tiene media cero."""
    design, names = design_matrix(_time(), detrend=True, seasonal="monthly")
    assert design[:, names.index("trend")].mean() == pytest.approx(0.0, abs=1e-12)


def test_seasonal_invalido_falla() -> None:
    with pytest.raises(ValueError, match="seasonal"):
        design_matrix(_time(), seasonal="semanal")  # type: ignore[arg-type]


def test_serie_mas_corta_que_el_modelo_falla() -> None:
    """Menos observaciones que parametros es un error, no un ajuste exacto."""
    with pytest.raises(ValueError, match="parametros"):
        design_matrix(_time(10), detrend=True, seasonal="monthly")


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #


def test_residuo_nulo_sin_senal_adicional() -> None:
    """Si los datos son exactamente el modelo, la anomalia es cero."""
    anom = fit_anomalies(_synthetic_cube())
    assert np.abs(anom.values).max() < 1e-9


def test_recupera_la_tendencia_impuesta() -> None:
    """La pendiente ajustada coincide con la impuesta, en unidades por decada."""
    _, coefs = fit_anomalies(_synthetic_cube(trend_per_year=0.02), return_coefs=True)

    assert float(coefs["trend"].mean()) == pytest.approx(0.02, abs=1e-6)
    assert float(coefs["trend_per_decade"].mean()) == pytest.approx(0.2, abs=1e-5)


def test_constante_es_el_nivel_en_el_centro_de_la_serie() -> None:
    """La constante se interpreta en el centro de la serie, no en su inicio.

    El cubo sintetico tiene su tendencia referida al primer mes, mientras que
    la matriz de diseno usa el tiempo centrado. La constante ajustada es, por
    tanto, el nivel de enero a mitad de la serie: el valor inicial mas la
    deriva acumulada en esos anios. Importa al leer los coeficientes, no al
    usar los residuos.
    """
    da = _synthetic_cube(mean=15.0, amplitude=3.0, trend_per_year=0.02)
    _, coefs = fit_anomalies(da, return_coefs=True)

    time = da.time.to_index()
    anios_al_centro = (time[-1] - time[0]).days / 365.2425 / 2
    esperado = 15.0 + 3.0 + 0.02 * anios_al_centro

    assert float(coefs["const"].mean()) == pytest.approx(esperado, abs=1e-3)


def test_la_senal_sobrevive_en_el_residuo() -> None:
    """Un pulso tipo evento no es absorbido por el modelo: queda en la anomalia."""
    da = _synthetic_cube()
    pulse = np.zeros(da.sizes["time"])
    pulse[60:72] = 2.5  # doce meses de calentamiento, tipo 1997-98
    da = da + xr.DataArray(pulse, coords={"time": da.time}, dims=["time"])

    anom = fit_anomalies(da)
    ventana = anom.isel(time=slice(60, 72)).mean().item()

    assert ventana > 2.0
    assert ventana == pytest.approx(2.5, abs=0.25)
    assert anom.mean().item() == pytest.approx(0.0, abs=1e-9)


def test_un_evento_descentrado_sesga_la_pendiente() -> None:
    """Un evento lejos del centro de la serie arrastra la tendencia estimada.

    Es la razon de ser de dos decisiones del proyecto: conservar siempre la
    version ``detrend=False`` como sensibilidad, y no leer la pendiente
    ajustada como una tendencia climatica. Aqui un pulso de doce meses situado
    en el quinto anio de una serie de treinta **reduce a la mitad** la
    pendiente estimada, aunque la deriva real no haya cambiado.

    La pendiente de minimos cuadrados es un promedio ponderado de los datos con
    pesos proporcionales a ``t - t_medio``, asi que un pulso aporta
    ``A * sum(t_i - t_medio) / sum((t_i - t_medio)^2)`` sobre los meses que
    cubre: nulo si cae centrado, maximo y de signo opuesto en cada extremo. El
    mismo pulso puesto en el ultimo anio en vez del quinto lleva la pendiente a
    181 por ciento de la real en lugar de 42 por ciento.

    El ajuste conjunto ordena la estimacion frente al esquema viejo
    (climatologia, resta, detrend), pero no vuelve robusta a la regresion
    lineal frente al leverage: eso exige comparar las dos versiones.
    """
    da = _synthetic_cube(trend_per_year=0.02)
    pulse = np.zeros(da.sizes["time"])
    pulse[60:72] = 2.5
    sesgado = da + xr.DataArray(pulse, coords={"time": da.time}, dims=["time"])

    _, limpio = fit_anomalies(da, return_coefs=True)
    _, con_evento = fit_anomalies(sesgado, return_coefs=True)

    pendiente_limpia = float(limpio["trend"].mean())
    pendiente_sesgada = float(con_evento["trend"].mean())

    assert pendiente_limpia == pytest.approx(0.02, abs=1e-6)
    assert pendiente_sesgada < 0.6 * pendiente_limpia


def test_anomalia_tiene_media_temporal_nula() -> None:
    """Un residuo de minimos cuadrados con constante tiene media cero por pixel."""
    anom = fit_anomalies(_synthetic_cube(noise=0.5))
    assert np.abs(anom.mean(dim="time").values).max() < 1e-9


def test_anomalia_no_tiene_ciclo_anual_residual() -> None:
    """Las medias por mes calendario quedan en cero: el ciclo fue removido."""
    anom = fit_anomalies(_synthetic_cube(noise=0.3))
    por_mes = anom.groupby("time.month").mean()
    assert np.abs(por_mes.values).max() < 1e-9


def test_detrend_falso_conserva_la_tendencia_en_el_residuo() -> None:
    """Sin la columna de tendencia, la deriva queda dentro de la anomalia."""
    da = _synthetic_cube(trend_per_year=0.05)
    anom = fit_anomalies(da, detrend=False)

    primeros = anom.isel(time=slice(0, 24)).mean().item()
    ultimos = anom.isel(time=slice(-24, None)).mean().item()

    assert ultimos - primeros > 1.0
    assert anom.attrs["detrended"] == "False"


def test_mascara_de_tierra_se_conserva() -> None:
    """Los pixeles de tierra siguen siendo NaN y no entran al ajuste."""
    da = _synthetic_cube(noise=0.2)
    da[:, 0, 0] = np.nan

    anom = fit_anomalies(da)

    assert bool(anom[:, 0, 0].isnull().all())
    assert bool(anom[:, 1, 1].notnull().all())
    assert anom.attrs["n_pixels_fitted"] == da.sizes["latitude"] * da.sizes["longitude"] - 1
    assert anom.attrs["n_pixels_partial_nan"] == 0


def test_nan_parcial_se_marca_y_no_se_rellena() -> None:
    """Un pixel con cobertura parcial queda NaN y queda contabilizado."""
    da = _synthetic_cube(noise=0.2)
    da[5, 0, 0] = np.nan

    anom = fit_anomalies(da)

    assert bool(anom[:, 0, 0].isnull().all())
    assert anom.attrs["n_pixels_partial_nan"] == 1


def test_dimensiones_y_coordenadas_intactas() -> None:
    """El ajuste no reordena ni renombra dimensiones."""
    da = _synthetic_cube()
    anom = fit_anomalies(da)

    assert anom.dims == da.dims
    assert anom.name == "thetao_anom"
    xr.testing.assert_identical(anom.latitude, da.latitude)


def test_r2_alto_sin_ruido_y_menor_con_ruido() -> None:
    """El R2 del ajuste refleja cuanta varianza explica el modelo."""
    _, limpio = fit_anomalies(_synthetic_cube(noise=0.0), return_coefs=True)
    _, ruidoso = fit_anomalies(_synthetic_cube(noise=1.0), return_coefs=True)

    assert float(limpio["r2"].mean()) > 0.999
    assert float(ruidoso["r2"].mean()) < float(limpio["r2"].mean())


def test_armonico_tambien_remueve_el_ciclo_coseno() -> None:
    """Con un ciclo puramente cosenoidal, el modelo armonico basta."""
    anom = fit_anomalies(_synthetic_cube(), seasonal="harmonic")
    assert np.abs(anom.values).max() < 1e-9


def test_falla_sin_dimension_temporal() -> None:
    da = _synthetic_cube().isel(time=0)
    with pytest.raises(ValueError, match="time"):
        fit_anomalies(da)


# --------------------------------------------------------------------------- #
# Suavizado
# --------------------------------------------------------------------------- #


def test_suavizado_pierde_un_mes_por_borde() -> None:
    """La media movil centrada de 3 meses deja NaN en el primer y ultimo mes."""
    anom = smooth_3m(fit_anomalies(_synthetic_cube(noise=0.5)))

    assert bool(anom.isel(time=0).isnull().all())
    assert bool(anom.isel(time=-1).isnull().all())
    assert bool(anom.isel(time=1).notnull().all())


def test_suavizado_reduce_la_varianza() -> None:
    """Promediar tres meses atenua la varianza de alta frecuencia."""
    anom = fit_anomalies(_synthetic_cube(noise=1.0))
    suave = smooth_3m(anom)

    assert float(suave.var()) < float(anom.var())


# --------------------------------------------------------------------------- #
# Cubo
# --------------------------------------------------------------------------- #


def test_cubo_nombra_las_variables_y_registra_el_metodo() -> None:
    """Cada variable produce '<nombre>_anom' y el metodo queda en los attrs."""
    ds = xr.Dataset({"thetao": _synthetic_cube(), "so": _synthetic_cube(mean=34.5)})
    cube = build_anomaly_cube(ds, ["thetao", "so"], detrend=True)

    assert set(cube.data_vars) == {"thetao_anom", "so_anom"}
    assert cube.attrs["detrended"] == "True"
    assert cube.attrs["smoothed_3m"] == "False"
    assert "1993-01-01" in cube.attrs["baseline_period"]


def test_cubo_falla_con_variable_ausente() -> None:
    ds = xr.Dataset({"thetao": _synthetic_cube()})
    with pytest.raises(KeyError, match="zos"):
        build_anomaly_cube(ds, ["thetao", "zos"])
