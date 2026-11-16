"""Deteccion de eventos El Nino / La Nina a partir del ONI.

La lista de eventos se genera siempre por codigo desde el ONI del CPC, nunca a
mano. Criterio operativo del CPC: un evento es una racha de **al menos cinco
temporadas trimestrales consecutivas** con el ONI en o por sobre +0.5 grados
(El Nino) o en o por debajo de -0.5 grados (La Nina).

Como ``indices.load_oni`` ancla cada temporada a su mes central, cinco
temporadas consecutivas equivalen a cinco meses consecutivos en la serie
mensual, y las fechas devueltas aqui son meses centrales de temporada.

Notes
-----
La clasificacion de intensidad (debil / moderada / fuerte / muy fuerte) es una
**convencion de uso habitual**, no una definicion oficial de la NOAA: el CPC
publica el ONI y el criterio de persistencia, pero no estos cortes.

El ONI no marca 2016-17 como evento. El "Nino costero" de 2017 fue un
calentamiento local frente a Peru y Chile, visible en el Nino 1+2 pero no en el
Pacifico central: requiere un criterio propio sobre el indice Nino 1+2 y se
trata como caso aparte, nunca mezclado con esta lista.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd

__all__ = [
    "INTENSITY_CUTS",
    "monthly_oni",
    "classify_intensity",
    "detect_events",
    "phase_series",
    "icen_index",
    "detect_events_icen",
]

EventType = Literal["nino", "nina"]

#: Cortes de intensidad sobre el maximo de |ONI| del evento, en grados Celsius.
#: Cada tramo es cerrado por abajo y abierto por arriba: debil [0.5, 1.0),
#: moderada [1.0, 1.5), fuerte [1.5, 2.0), muy fuerte [2.0, inf). Equivalen a
#: los tramos 0.5-0.9 / 1.0-1.4 / 1.5-1.9 / >=2.0 de la convencion habitual,
#: porque el CPC publica el ONI con un decimal.
INTENSITY_CUTS: Final[tuple[tuple[float, str], ...]] = (
    (2.0, "muy_fuerte"),
    (1.5, "fuerte"),
    (1.0, "moderada"),
    (0.5, "debil"),
)


def monthly_oni(oni: pd.DataFrame | pd.Series) -> pd.Series:
    """Extraer la serie mensual de ONI de la tabla de ``indices.load_oni``.

    Parameters
    ----------
    oni : pandas.DataFrame or pandas.Series
        Salida de :func:`enso_chile.indices.load_oni`, la tabla maestra de
        :func:`enso_chile.indices.build_master_table`, o ya una serie de ONI.

    Returns
    -------
    pandas.Series
        Serie ``oni`` indexada al inicio de cada mes, ordenada y sin duplicados.

    Raises
    ------
    KeyError
        Si el ``DataFrame`` no tiene una columna ``oni``.
    """
    series = oni if isinstance(oni, pd.Series) else oni["oni"]
    series = series.astype(float).sort_index()
    return series[~series.index.duplicated(keep="last")].rename("oni")


def classify_intensity(peak: float) -> str:
    """Clasificar la intensidad de un evento por el maximo de |ONI|.

    Parameters
    ----------
    peak : float
        Maximo de |ONI| alcanzado durante el evento, en grados Celsius.

    Returns
    -------
    str
        ``'debil'``, ``'moderada'``, ``'fuerte'`` o ``'muy_fuerte'``; y
        ``'subumbral'`` si el valor no alcanza 0.5, caso que no deberia
        aparecer en un evento detectado.

    Notes
    -----
    Convencion de uso, no una definicion oficial de la NOAA. Ver
    :data:`INTENSITY_CUTS`.
    """
    magnitude = abs(float(peak))
    for cut, label in INTENSITY_CUTS:
        if magnitude >= cut:
            return label
    return "subumbral"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Posiciones ``(inicio, fin)`` inclusivas de cada racha de ``True``."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def _event_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Etiqueta del evento al estilo ``'1997-98'`` o ``'2020-23'``.

    Un evento contenido en un solo anio calendario se etiqueta con ese anio.
    """
    if start.year == end.year:
        return str(start.year)
    return f"{start.year}-{end.year % 100:02d}"


def detect_events(
    oni: pd.DataFrame | pd.Series,
    threshold: float = 0.5,
    min_months: int = 5,
) -> pd.DataFrame:
    """Detectar eventos El Nino y La Nina por persistencia del ONI.

    Parameters
    ----------
    oni : pandas.DataFrame or pandas.Series
        ONI mensual; ver :func:`monthly_oni`.
    threshold : float, optional
        Umbral absoluto en grados Celsius. Por defecto 0.5.
    min_months : int, optional
        Numero minimo de meses (temporadas) consecutivos sobre el umbral. Por
        defecto 5, el criterio del CPC.

    Returns
    -------
    pandas.DataFrame
        Una fila por evento, ordenada en el tiempo, con columnas:

        ``type``
            ``'nino'`` o ``'nina'``.
        ``label``
            Etiqueta del tipo ``'1997-98'``.
        ``start``, ``end``
            Primer y ultimo mes central de la racha.
        ``n_months``
            Largo de la racha, en meses.
        ``peak_oni``
            Valor del ONI con mayor magnitud dentro de la racha, con su signo.
        ``peak_time``
            Mes en que ocurre ese maximo.
        ``intensity``
            Etiqueta de :func:`classify_intensity`.

    Notes
    -----
    Los meses con ONI ausente (NaN) **cortan** la racha: no se interpolan ni se
    tratan como si cumplieran el umbral. Esto importa al final de la serie,
    donde el ultimo trimestre puede estar aun sin publicar, y significa que un
    evento en curso aparece truncado al ultimo mes disponible.
    """
    if min_months < 1:
        raise ValueError("min_months debe ser >= 1.")
    if threshold <= 0:
        raise ValueError("threshold debe ser > 0.")

    series = monthly_oni(oni)
    values = series.to_numpy(dtype=float)
    valid = ~np.isnan(values)

    records: list[dict[str, object]] = []
    conditions: tuple[tuple[EventType, np.ndarray], ...] = (
        ("nino", valid & (values >= threshold)),
        ("nina", valid & (values <= -threshold)),
    )

    for event_type, mask in conditions:
        for first, last in _runs(mask):
            if last - first + 1 < min_months:
                continue
            window = values[first : last + 1]
            offset = int(np.nanargmax(np.abs(window)))
            start = series.index[first]
            end = series.index[last]
            records.append(
                {
                    "type": event_type,
                    "label": _event_label(start, end),
                    "start": start,
                    "end": end,
                    "n_months": last - first + 1,
                    "peak_oni": float(window[offset]),
                    "peak_time": series.index[first + offset],
                    "intensity": classify_intensity(window[offset]),
                }
            )

    columns = [
        "type", "label", "start", "end",
        "n_months", "peak_oni", "peak_time", "intensity",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    events = pd.DataFrame.from_records(records, columns=columns)
    return events.sort_values("start", ignore_index=True)


def phase_series(
    oni: pd.DataFrame | pd.Series, events: pd.DataFrame | None = None
) -> pd.Series:
    """Etiquetar cada mes con la fase ENSO del evento que lo contiene.

    Util para estratificar el cubo de anomalias o colorear series de tiempo:
    marca solo los meses que pertenecen a un evento que **supero** el criterio
    de persistencia, no todos los meses que cruzan el umbral.

    Parameters
    ----------
    oni : pandas.DataFrame or pandas.Series
        ONI mensual; ver :func:`monthly_oni`.
    events : pandas.DataFrame or None, optional
        Tabla de :func:`detect_events`. Si es ``None``, se calcula con los
        parametros por defecto.

    Returns
    -------
    pandas.Series
        Serie de texto con valores ``'nino'``, ``'nina'`` o ``'neutral'``,
        sobre el mismo indice mensual del ONI.
    """
    series = monthly_oni(oni)
    table = detect_events(series) if events is None else events

    phase = pd.Series("neutral", index=series.index, name="phase", dtype=object)
    for row in table.itertuples(index=False):
        phase.loc[row.start : row.end] = row.type
    return phase


def icen_index(nino12: pd.Series) -> pd.Series:
    """Indice costero estilo ICEN: media movil centrada de 3 meses del Nino 1+2.

    Parameters
    ----------
    nino12 : pandas.Series
        Anomalia mensual de TSM en la region Nino 1+2 (0-10 S, 90-80 W),
        indexada por mes.

    Returns
    -------
    pandas.Series
        Media movil centrada de 3 meses, sin los extremos que no tienen
        ventana completa.

    Notes
    -----
    El ICEN oficial (ENFEN 2012; Takahashi et al. 2014) usa su propia fuente y
    linea base de anomalias; aqui se aplica la misma construccion a la serie
    Nino 1+2 del proyecto, asi que los valores no son identicos a los publicados.
    """
    series = nino12.astype(float).sort_index()
    return series.rolling(3, center=True, min_periods=3).mean().dropna().rename("icen")


def detect_events_icen(
    index: pd.Series,
    warm: float = 0.4,
    cold: float = -1.0,
    min_months: int = 3,
) -> pd.DataFrame:
    """Detectar eventos costeros con el criterio asimetrico del ICEN.

    Parameters
    ----------
    index : pandas.Series
        Indice ya suavizado (por ejemplo :func:`icen_index`), mensual.
    warm : float, optional
        Umbral calido: Nino si el indice **supera** ``warm`` (estricto).
    cold : float, optional
        Umbral frio: Nina si el indice esta **en o por debajo** de ``cold``.
    min_months : int, optional
        Meses consecutivos minimos. Por defecto 3, el criterio del ICEN.

    Returns
    -------
    pandas.DataFrame
        Una fila por evento, ordenada en el tiempo, con columnas ``type``,
        ``label``, ``start``, ``end``, ``n_months``, ``peak`` (valor del indice
        con mayor magnitud, con signo) y ``peak_time``.

    Notes
    -----
    Umbrales de ENFEN (2012) y Takahashi et al. (2014): +0.4 y -1.0 grados. Que
    el umbral calido sea estricto y el frio inclusivo es una convencion de este
    proyecto; la fuente consultada no la precisa. Los meses ausentes (NaN) cortan
    la racha, igual que en :func:`detect_events`. Sirve para cualquier indice
    costero suavizado, no solo para el Nino 1+2.
    """
    if min_months < 1:
        raise ValueError("min_months debe ser >= 1.")
    if not cold < warm:
        raise ValueError("cold debe ser menor que warm.")

    series = index.astype(float).sort_index()
    values = series.to_numpy()
    valid = ~np.isnan(values)
    records: list[dict[str, object]] = []
    conditions: tuple[tuple[EventType, np.ndarray], ...] = (
        ("nino", valid & (values > warm)),
        ("nina", valid & (values <= cold)),
    )
    for event_type, mask in conditions:
        for first, last in _runs(mask):
            if last - first + 1 < min_months:
                continue
            window = values[first : last + 1]
            offset = int(np.nanargmax(np.abs(window)))
            start, end = series.index[first], series.index[last]
            records.append(
                {
                    "type": event_type,
                    "label": _event_label(start, end),
                    "start": start,
                    "end": end,
                    "n_months": last - first + 1,
                    "peak": float(window[offset]),
                    "peak_time": series.index[first + offset],
                }
            )

    columns = ["type", "label", "start", "end", "n_months", "peak", "peak_time"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        "start", ignore_index=True
    )
