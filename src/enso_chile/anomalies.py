"""Anomalias por ajuste conjunto de constante, tendencia y ciclo anual.

Para cada pixel se resuelve por minimos cuadrados un solo modelo

.. math::

    y_t = \\beta_0 + \\beta_1 t + s(t) + \\varepsilon_t,

donde :math:`s(t)` es el ciclo estacional, parametrizado como once variables
indicadoras mensuales (enero es la categoria de referencia) o como armonicos
anual y semianual. La **anomalia es el residuo** :math:`\\varepsilon_t`.

Por que el ajuste conjunto y no el esquema de los scripts viejos
----------------------------------------------------------------
El esquema anterior era secuencial: calcular la climatologia mensual, restarla
y despues quitar la tendencia. Resolver constante, tendencia y efectos
mensuales en una sola matriz de diseno elimina ese orden de estimacion: cada
termino queda estimado condicionado a los demas.

**Verificado sobre este cubo, los dos esquemas son equivalentes.** Con una
serie de 399 meses (casi 33 anios completos), cada mes calendario aparece
aproximadamente el mismo numero de veces y de forma regular, asi que las
indicadoras mensuales son casi ortogonales a la columna de tendencia. Cuando el
diseno es ortogonal, estimar en dos pasos o en uno da el mismo resultado: la
pendiente por pixel de ambos esquemas coincide con correlacion 1.0000 y
diferencia media nula (``thetao``, superficie, Norte Grande).

El ajuste conjunto se adopta porque **no depende** de ese balance, no porque
corrija un sesgo medible aqui. Deja de ser equivalente cuando el diseno se
desbalancea: series que no cubren anios enteros, meses faltantes, o una linea
base climatologica distinta del periodo de la anomalia. El propio cubo heredado
ilustra el ultimo caso (ver abajo).

Lo que ningun esquema de minimos cuadrados lineales evita es el **leverage**:
la pendiente es un promedio ponderado con pesos proporcionales a la distancia
al centro de la serie, de modo que un evento cercano a un extremo la arrastra.
Con los datos reales, excluir 1997-98 lleva la tendencia superficial de
Norte Grande de +0.014 a +0.074 grados por decada, y excluir 2023-24 la lleva a
-0.030: **el signo depende de que evento se incluya**. Por eso la tendencia
lineal se declara como simplificacion, se conserva siempre la version
``detrend=False`` como sensibilidad, y la pendiente ajustada no se interpreta
como tendencia climatica.

Diferencia con el cubo de anomalias heredado
--------------------------------------------
El cubo heredado (``chile_norte_processed_anomaly_cube*.nc``) se reproduce aqui
bit a bit (correlacion 1.000000, diferencia maxima 0.0000) al replicar su
esquema. Su discrepancia con los cubos nuevos, de 0.0139 grados de desviacion,
proviene **por completo** de la linea base: ``COAST.py`` l.50 fija
``base_period=('1993-01-01', '2025-12-31')`` mientras calcula la anomalia sobre
toda la serie, que llega a 2026-03. Los tres primeros meses de 2026 quedan con
una climatologia estimada sobre un periodo que los excluye, y el resto de la
serie arrastra un desplazamiento constante de unos +0.0095 grados. No tiene
relacion con el cambio de esquema de ajuste.

Notes
-----
LEGACY: el motor de anomalias mas limpio de los scripts originales esta en
``COAST.py`` l.50-75; ``get_noaa_anomalies_spatial`` se repite unas once veces
en ``Glorys12_and_Index.py``, tres en ``COAST.py`` y dos en
``EarlySignal_and_Impact.py``. Las anomalias de ``TS_NOAA.py`` **no** se
reutilizan: ahi el detrend se aplica sobre el vector aplanado, lo que mezcla
pixeles distintos en una sola regresion.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "SeasonalModel",
    "design_matrix",
    "fit_anomalies",
    "smooth_3m",
    "build_anomaly_cube",
]

SeasonalModel = Literal["monthly", "harmonic"]

#: Dias por anio usados para escalar el tiempo a anios decimales.
_DAYS_PER_YEAR: Final[float] = 365.2425


def _time_in_years(time: xr.DataArray | pd.DatetimeIndex) -> np.ndarray:
    """Tiempo en anios decimales, centrado en la mitad de la serie.

    Centrar deja la constante igual a la media ajustada y descorrelaciona la
    constante de la pendiente, lo que mejora el condicionamiento de la matriz.
    """
    index = pd.DatetimeIndex(np.asarray(time))
    days = (index - index[0]).days.to_numpy(dtype=float)
    years = days / _DAYS_PER_YEAR
    return years - years.mean()


def design_matrix(
    time: xr.DataArray | pd.DatetimeIndex,
    detrend: bool = True,
    seasonal: SeasonalModel = "monthly",
) -> tuple[np.ndarray, list[str]]:
    """Construir la matriz de diseno del ajuste de anomalias.

    Parameters
    ----------
    time : xarray.DataArray or pandas.DatetimeIndex
        Eje temporal mensual de la serie.
    detrend : bool, optional
        Incluir la columna de tendencia lineal. Si es ``False``, el modelo solo
        tiene constante y ciclo estacional (version de sensibilidad).
    seasonal : {'monthly', 'harmonic'}, optional
        Parametrizacion del ciclo anual. ``'monthly'`` usa once indicadoras
        mensuales (13 parametros con la tendencia); ``'harmonic'`` usa los
        armonicos anual y semianual (6 parametros), mas parsimonioso pero
        incapaz de representar un ciclo estacional de forma arbitraria.

    Returns
    -------
    X : numpy.ndarray
        Matriz de diseno de forma ``(n_tiempos, n_parametros)``.
    names : list of str
        Nombre de cada columna, en orden.

    Raises
    ------
    ValueError
        Si ``seasonal`` no es un valor valido, o si la matriz resultante tiene
        mas parametros que observaciones.

    Notes
    -----
    Con ``'monthly'`` se omite enero para evitar la colinealidad exacta entre
    las doce indicadoras y la constante. Los efectos mensuales se leen entonces
    como contrastes respecto a enero; esto no altera los residuos, que son lo
    unico que se usa como anomalia.
    """
    index = pd.DatetimeIndex(np.asarray(time))
    n = len(index)

    columns: list[np.ndarray] = [np.ones(n)]
    names: list[str] = ["const"]

    if detrend:
        columns.append(_time_in_years(index))
        names.append("trend")

    if seasonal == "monthly":
        month = index.month.to_numpy()
        for m in range(2, 13):  # enero (1) es la categoria de referencia
            columns.append((month == m).astype(float))
            names.append(f"month_{m:02d}")
    elif seasonal == "harmonic":
        phase = 2.0 * np.pi * (index.month.to_numpy() - 1) / 12.0
        for k in (1, 2):
            columns.append(np.cos(k * phase))
            names.append(f"cos_{k}")
            columns.append(np.sin(k * phase))
            names.append(f"sin_{k}")
    else:
        raise ValueError(
            f"seasonal debe ser 'monthly' o 'harmonic', no {seasonal!r}."
        )

    design = np.column_stack(columns)
    if design.shape[1] >= n:
        raise ValueError(
            f"La matriz de diseno tiene {design.shape[1]} parametros para "
            f"{n} observaciones."
        )
    return design, names


def fit_anomalies(
    da: xr.DataArray,
    detrend: bool = True,
    seasonal: SeasonalModel = "monthly",
    time_dim: str = "time",
    return_coefs: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.Dataset]:
    """Ajustar el modelo pixel a pixel y devolver los residuos como anomalia.

    El ajuste esta vectorizado: la matriz de diseno es comun a todos los
    pixeles, asi que se resuelve un unico sistema de minimos cuadrados con
    todas las series como columnas del lado derecho.

    Parameters
    ----------
    da : xarray.DataArray
        Campo con dimension temporal. Las demas dimensiones (profundidad,
        latitud, longitud) se tratan como pixeles independientes.
    detrend : bool, optional
        Incluir la tendencia lineal en el modelo.
    seasonal : {'monthly', 'harmonic'}, optional
        Parametrizacion del ciclo anual.
    time_dim : str, optional
        Nombre de la dimension temporal.
    return_coefs : bool, optional
        Devolver tambien los coeficientes ajustados.

    Returns
    -------
    anom : xarray.DataArray
        Residuos, con las mismas dimensiones, coordenadas y mascara de tierra
        que la entrada. Lleva en ``attrs`` el modelo usado.
    coefs : xarray.Dataset, optional
        Solo si ``return_coefs``. Una variable por parametro del modelo, mas
        ``trend_per_decade`` (unidades de ``da`` por decada) cuando
        ``detrend=True``, y ``r2`` por pixel.

    Notes
    -----
    Los pixeles enteramente ausentes (tierra) se devuelven como NaN sin pasar
    por el ajuste. Un pixel con NaN **parciales** no se puede resolver con el
    sistema comun: esos pixeles tambien se devuelven como NaN y se cuentan en
    el atributo ``n_pixels_partial_nan``, en vez de rellenarse por
    interpolacion. En GLORYS12 la mascara es constante en el tiempo, asi que
    ese contador deberia ser cero; si no lo es, conviene revisar el cubo antes
    de seguir.
    """
    if time_dim not in da.dims:
        raise ValueError(f"{da.name!r} no tiene la dimension {time_dim!r}.")

    design, names = design_matrix(da[time_dim], detrend, seasonal)

    stacked = da.stack(pixel=[d for d in da.dims if d != time_dim])
    stacked = stacked.transpose(time_dim, "pixel")
    values = stacked.values.astype(np.float64)  # (n_tiempos, n_pixeles)

    finite = np.isfinite(values)
    complete = finite.all(axis=0)
    empty = ~finite.any(axis=0)
    partial = ~complete & ~empty

    residuals = np.full_like(values, np.nan)
    coefficients = np.full((design.shape[1], values.shape[1]), np.nan)

    if complete.any():
        observed = values[:, complete]
        beta, *_ = np.linalg.lstsq(design, observed, rcond=None)
        fitted = design @ beta
        residuals[:, complete] = observed - fitted
        coefficients[:, complete] = beta

    anom = xr.DataArray(
        residuals, coords=stacked.coords, dims=stacked.dims
    ).unstack("pixel")
    anom = anom.transpose(*da.dims)
    anom.name = None if da.name is None else f"{da.name}_anom"
    anom.attrs = {
        **da.attrs,
        "long_name": f"anomalia de {da.name}",
        "anomaly_method": (
            "residuo de minimos cuadrados: constante"
            + (" + tendencia lineal" if detrend else "")
            + (
                " + 11 indicadoras mensuales"
                if seasonal == "monthly"
                else " + armonicos anual y semianual"
            )
        ),
        "detrended": str(bool(detrend)),
        "seasonal_model": seasonal,
        "n_pixels_fitted": int(complete.sum()),
        "n_pixels_partial_nan": int(partial.sum()),
    }

    if not return_coefs:
        return anom

    template = xr.DataArray(
        coefficients[0], coords={"pixel": stacked["pixel"]}, dims=["pixel"]
    )
    coef_vars: dict[str, xr.DataArray] = {}
    for i, name in enumerate(names):
        coef_vars[name] = template.copy(data=coefficients[i]).unstack("pixel")

    if detrend:
        decade = coef_vars["trend"] * 10.0
        decade.attrs["long_name"] = "tendencia lineal por decada"
        coef_vars["trend_per_decade"] = decade

    ss_res = np.nansum(residuals**2, axis=0)
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    ss_tot = np.nansum(centered**2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    r2[~complete] = np.nan
    r2_da = template.copy(data=r2).unstack("pixel")
    r2_da.attrs["long_name"] = "R2 del ajuste (constante + tendencia + estacional)"
    coef_vars["r2"] = r2_da

    coefs = xr.Dataset(coef_vars)
    coefs.attrs = dict(anom.attrs)
    return anom, coefs


def smooth_3m(
    da: xr.DataArray, time_dim: str = "time", center: bool = True
) -> xr.DataArray:
    """Media movil de tres meses sobre la dimension temporal.

    Replica la convencion del ONI, que se publica como media movil trimestral.
    Pierde un mes en cada borde de la serie: por eso esta **desactivada por
    defecto** en el cubo base (``config.ANOMALY_SMOOTH_3M``). El cubo de
    anomalias heredado la traia aplicada, de ahi sus 397 pasos en lugar de 399.

    La razon principal para no suavizar antes de un analisis de modos no es el
    par de meses perdidos, sino los grados de libertad. Medido sobre
    ``thetao`` a 0 m en Norte Grande: la autocorrelacion de lag 1 sube de 0.62
    a 0.90 y el tamanio de muestra efectivo bajo la aproximacion AR(1),
    ``N (1 - r) / (1 + r)``, cae de 94 a 21 sobre 399 meses nominales. Como los
    intervalos de North et al. (1982) escalan con ``sqrt(2 / N_efectivo)``, ese
    factor pasa de 0.15 a 0.31: suavizar **duplica** el ancho de los intervalos
    de los autovalores y vuelve mucho mas facil declarar modos degenerados que
    en realidad son separables. Para el ONI el suavizado es correcto, porque
    ahi el objetivo es definir persistencia de eventos, no descomponer.

    Parameters
    ----------
    da : xarray.DataArray
        Campo a suavizar.
    time_dim : str, optional
        Nombre de la dimension temporal.
    center : bool, optional
        Ventana centrada. Con ``False`` la media es retrospectiva y pierde dos
        meses al inicio.

    Returns
    -------
    xarray.DataArray
        Campo suavizado, con los bordes como NaN.
    """
    out = da.rolling({time_dim: 3}, center=center, min_periods=3).mean()
    out.attrs = {**da.attrs, "smoothing": "media movil de 3 meses"}
    return out


def build_anomaly_cube(
    ds: xr.Dataset,
    variables: tuple[str, ...] | list[str],
    detrend: bool = True,
    seasonal: SeasonalModel = "monthly",
    smooth: bool = False,
    time_dim: str = "time",
) -> xr.Dataset:
    """Calcular las anomalias de varias variables y reunirlas en un cubo.

    Parameters
    ----------
    ds : xarray.Dataset
        Cubo de entrada, ya recortado a la caja y a las profundidades de
        trabajo (ver :mod:`enso_chile.cube`).
    variables : sequence of str
        Variables a procesar. Cada una produce ``<nombre>_anom``.
    detrend : bool, optional
        Incluir la tendencia lineal.
    seasonal : {'monthly', 'harmonic'}, optional
        Parametrizacion del ciclo anual.
    smooth : bool, optional
        Aplicar la media movil de tres meses despues del ajuste.
    time_dim : str, optional
        Nombre de la dimension temporal.

    Returns
    -------
    xarray.Dataset
        Cubo de anomalias. Los atributos globales registran el modelo, si hubo
        suavizado y el periodo de la linea base, para que ningun resultado
        quede sin trazabilidad de como se calculo.

    Raises
    ------
    KeyError
        Si alguna variable solicitada no esta en el cubo.
    """
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables ausentes en el cubo: {missing}")

    out: dict[str, xr.DataArray] = {}
    for var in variables:
        anom = fit_anomalies(ds[var], detrend, seasonal, time_dim)
        if smooth:
            anom = smooth_3m(anom, time_dim)
        out[f"{var}_anom"] = anom

    cube = xr.Dataset(out)
    index = pd.DatetimeIndex(np.asarray(ds[time_dim]))
    cube.attrs = {
        "anomaly_method": (
            "residuo de minimos cuadrados conjunto por pixel: constante"
            + (" + tendencia lineal" if detrend else "")
            + (
                " + 11 indicadoras mensuales"
                if seasonal == "monthly"
                else " + armonicos anual y semianual"
            )
        ),
        "detrended": str(bool(detrend)),
        "seasonal_model": seasonal,
        "smoothed_3m": str(bool(smooth)),
        "baseline_period": f"{index[0].date()}/{index[-1].date()}",
        "baseline_note": (
            "la linea base es el periodo completo e incluye los eventos de "
            "estudio; ver la version con linea base alternativa como "
            "analisis de sensibilidad"
        ),
        "source": "GLORYS12 cmems_mod_glo_phy_my_0.083deg_P1M-m",
    }
    return cube
