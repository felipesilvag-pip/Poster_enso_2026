"""Tests de la deteccion de eventos ENSO desde el ONI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_chile.events import (
    classify_intensity,
    detect_events,
    detect_events_icen,
    icen_index,
    monthly_oni,
    phase_series,
)


def _oni_from_values(values: list[float], start: str = "2000-01-01") -> pd.Series:
    """Serie de ONI sintetica, mensual y continua."""
    index = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=index, name="oni")


# --------------------------------------------------------------------------- #
# Intensidad
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (0.5, "debil"),
        (0.9, "debil"),
        (1.0, "moderada"),
        (1.4, "moderada"),
        (1.5, "fuerte"),
        (1.9, "fuerte"),
        (2.0, "muy_fuerte"),
        (2.6, "muy_fuerte"),
        (0.4, "subumbral"),
    ],
)
def test_classify_intensity_cortes(peak: float, expected: str) -> None:
    """Los cortes reproducen la convencion 0.5-0.9 / 1.0-1.4 / 1.5-1.9 / >=2.0."""
    assert classify_intensity(peak) == expected


def test_classify_intensity_es_simetrica() -> None:
    """La intensidad depende de la magnitud, no del signo (Nino o Nina)."""
    assert classify_intensity(-1.7) == classify_intensity(1.7) == "fuerte"


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #


def test_racha_corta_no_es_evento() -> None:
    """Cuatro meses sobre el umbral no alcanzan el criterio de cinco."""
    oni = _oni_from_values([0.0, 0.6, 0.8, 0.9, 0.7, 0.0, 0.0])
    assert detect_events(oni).empty


def test_racha_de_cinco_es_evento() -> None:
    """Cinco meses consecutivos sobre el umbral constituyen un evento."""
    oni = _oni_from_values([0.0, 0.6, 0.8, 1.2, 0.9, 0.7, 0.0])
    events = detect_events(oni)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["type"] == "nino"
    assert event["n_months"] == 5
    assert event["start"] == pd.Timestamp("2000-02-01")
    assert event["end"] == pd.Timestamp("2000-06-01")
    assert event["peak_oni"] == pytest.approx(1.2)
    assert event["peak_time"] == pd.Timestamp("2000-04-01")
    assert event["intensity"] == "moderada"


def test_umbral_es_inclusivo() -> None:
    """Un ONI de exactamente +0.5 cumple el umbral ('en o por sobre')."""
    oni = _oni_from_values([0.0] + [0.5] * 5 + [0.0])
    assert len(detect_events(oni)) == 1


def test_nina_se_detecta_con_signo_negativo() -> None:
    """La rama negativa produce eventos de tipo 'nina' con pico negativo."""
    oni = _oni_from_values([0.0] + [-0.6, -0.9, -1.6, -1.1, -0.5] + [0.0])
    events = detect_events(oni)

    assert len(events) == 1
    assert events.iloc[0]["type"] == "nina"
    assert events.iloc[0]["peak_oni"] == pytest.approx(-1.6)
    assert events.iloc[0]["intensity"] == "fuerte"


def test_nan_corta_la_racha() -> None:
    """Un mes sin dato parte la racha en dos tramos cortos, sin evento."""
    oni = _oni_from_values([0.6, 0.8, np.nan, 0.9, 0.7, 0.8, 0.6])
    assert detect_events(oni).empty


def test_eventos_consecutivos_no_se_fusionan() -> None:
    """Un Nino y una Nina separados por meses neutros son dos eventos."""
    oni = _oni_from_values(
        [0.0] + [0.8] * 5 + [0.1, 0.0] + [-0.8] * 5 + [0.0]
    )
    events = detect_events(oni)

    assert list(events["type"]) == ["nino", "nina"]
    assert events["start"].is_monotonic_increasing


def test_etiqueta_cruza_el_anio() -> None:
    """Un evento que cruza diciembre se etiqueta al estilo '2000-01'."""
    oni = _oni_from_values([0.0] * 9 + [0.8] * 6, start="2000-01-01")
    assert detect_events(oni).iloc[0]["label"] == "2000-01"


def test_min_months_configurable() -> None:
    """Bajar la persistencia minima admite rachas mas cortas."""
    oni = _oni_from_values([0.0, 0.6, 0.8, 0.0])
    assert detect_events(oni, min_months=2).shape[0] == 1


@pytest.mark.parametrize(("threshold", "min_months"), [(0.0, 5), (-0.5, 5), (0.5, 0)])
def test_parametros_invalidos(threshold: float, min_months: int) -> None:
    """Un umbral no positivo o una persistencia nula son errores."""
    with pytest.raises(ValueError):
        detect_events(_oni_from_values([0.0] * 6), threshold, min_months)


# --------------------------------------------------------------------------- #
# Series derivadas
# --------------------------------------------------------------------------- #


def test_monthly_oni_acepta_dataframe_y_ordena() -> None:
    """Se extrae la columna 'oni', se ordena y se quitan indices duplicados."""
    index = pd.to_datetime(["2000-03-01", "2000-01-01", "2000-01-01"])
    frame = pd.DataFrame({"season": ["FMA", "DJF", "DJF"], "oni": [0.3, 0.1, 0.2]}, index=index)
    series = monthly_oni(frame)

    assert series.index.is_monotonic_increasing
    assert len(series) == 2
    assert series.loc["2000-01-01"] == pytest.approx(0.2)


def test_phase_series_solo_marca_eventos_validos() -> None:
    """Un mes sobre el umbral fuera de un evento persistente queda neutral."""
    oni = _oni_from_values([0.9, 0.0] + [0.8] * 5 + [0.0])
    phase = phase_series(oni)

    assert phase.iloc[0] == "neutral"
    assert set(phase.iloc[2:7]) == {"nino"}
    assert phase.iloc[-1] == "neutral"
    assert len(phase) == len(oni)


# --------------------------------------------------------------------------- #
# Criterio ICEN (Nino 1+2)
# --------------------------------------------------------------------------- #


def test_icen_index_es_media_movil_centrada() -> None:
    """Media de 3 meses centrada; se pierden el primer y el ultimo mes."""
    s = _oni_from_values([0.0, 3.0, 6.0, 9.0])
    icen = icen_index(s)

    assert len(icen) == 2
    assert icen.iloc[0] == pytest.approx(3.0)
    assert icen.index[0] == s.index[1]


def test_icen_umbrales_asimetricos() -> None:
    """-0.5 no es Nina (umbral frio -1.0); +0.5 si es Nino (umbral +0.4)."""
    idx = _oni_from_values([0.5] * 3 + [0.0] + [-0.5] * 3)
    events = detect_events_icen(idx)

    assert list(events["type"]) == ["nino"]


def test_icen_calido_estricto_frio_inclusivo() -> None:
    """+0.4 exacto no cuenta como Nino; -1.0 exacto si cuenta como Nina."""
    assert detect_events_icen(_oni_from_values([0.4] * 4)).empty
    events = detect_events_icen(_oni_from_values([-1.0] * 3))
    assert list(events["type"]) == ["nina"]


def test_icen_minimo_tres_meses_y_pico_con_signo() -> None:
    """Dos meses no bastan; el pico conserva el signo y su fecha."""
    assert detect_events_icen(_oni_from_values([1.0, 1.0, 0.0])).empty
    events = detect_events_icen(_oni_from_values([0.0, -1.2, -2.5, -1.1, 0.0]))

    assert events.loc[0, "peak"] == pytest.approx(-2.5)
    assert events.loc[0, "peak_time"] == pd.Timestamp("2000-03-01")
    assert events.loc[0, "n_months"] == 3


def test_icen_nan_corta_la_racha() -> None:
    idx = _oni_from_values([1.0, 1.0, np.nan, 1.0, 1.0])
    assert detect_events_icen(idx).empty


def test_icen_umbrales_invalidos() -> None:
    with pytest.raises(ValueError):
        detect_events_icen(_oni_from_values([0.0] * 3), warm=-1.0, cold=0.4)
