"""Tests del parseo de indices climaticos, sin tocar la red.

Los parsers se prueban contra fragmentos de texto con el mismo formato que los
archivos de la NOAA. El unico test que descarga esta marcado ``network`` y se
excluye por defecto (ver ``pyproject.toml``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_chile import indices
from enso_chile.indices import (
    _parse_year_month_matrix,
    build_master_table,
    fetch_text,
    load_oni,
    load_soi,
)

# --------------------------------------------------------------------------- #
# Fixtures de texto
# --------------------------------------------------------------------------- #

ONI_TEXT = """SEAS YR TOTAL ANOM
DJF 1997 26.52 -0.48
JFM 1997 26.98 -0.36
FMA 1997 27.35 0.14
MAM 1997 27.95 0.76
AMJ 1997 28.21 1.18
MJJ 1997 28.08 1.61
JJA 1997 27.95 1.92
"""

#: Matriz anio + 12 meses, con un mes faltante (-99.99) en el interior.
MATRIX_TEXT = """  Linea de encabezado que no es dato
  Otra nota cualquiera
1993  0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90  1.00  1.10  1.20
1994  1.30 -99.99  1.50  1.60  1.70  1.80  1.90  2.00  2.10  2.20  2.30  2.40
Nota al pie del archivo, se ignora
"""

SOI_TEXT = """  ANOMALY
1993  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
  STANDARDIZED DATA
  YEAR JAN FEB
1993 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0 -1.1 -1.2 -1.3 -1.4 -1.5 -1.6
"""


# --------------------------------------------------------------------------- #
# Matriz anio x 12 meses
# --------------------------------------------------------------------------- #


def test_matriz_construye_fechas_al_inicio_de_mes() -> None:
    """Cada celda se fecha por su par (anio, mes), al dia 1."""
    series = _parse_year_month_matrix(MATRIX_TEXT, "x")

    assert len(series) == 24
    assert series.index[0] == pd.Timestamp("1993-01-01")
    assert series.index[-1] == pd.Timestamp("1994-12-01")
    assert (series.index.day == 1).all()


def test_matriz_no_desplaza_la_serie_por_un_hueco() -> None:
    """Un mes faltante queda como NaN en su lugar y no corre las fechas.

    Es la correccion al bug de los scripts originales, que fechaban con
    ``pd.date_range(..., periods=len(serie))`` despues de ``stack()``: al
    descartar ``stack`` los NaN, todo lo posterior al hueco se adelantaba un mes.
    """
    series = _parse_year_month_matrix(MATRIX_TEXT, "x")

    assert np.isnan(series.loc["1994-02-01"])
    assert series.loc["1994-03-01"] == pytest.approx(1.50)
    assert series.loc["1993-12-01"] == pytest.approx(1.20)


def test_matriz_ignora_encabezados_y_notas() -> None:
    """Se parsean solo las filas que empiezan con un anio de cuatro digitos."""
    series = _parse_year_month_matrix(MATRIX_TEXT, "x")
    assert set(series.index.year) == {1993, 1994}


def test_matriz_sin_datos_falla_explicitamente() -> None:
    """Un archivo sin filas validas levanta ValueError, no devuelve vacio."""
    with pytest.raises(ValueError, match="No se pudo parsear"):
        _parse_year_month_matrix("solo texto\nsin datos\n", "x")


# --------------------------------------------------------------------------- #
# ONI
# --------------------------------------------------------------------------- #


def test_oni_ancla_cada_temporada_a_su_mes_central(monkeypatch) -> None:
    """DJF va a enero, JFM a febrero, ..., NDJ a diciembre."""
    monkeypatch.setattr(indices, "fetch_text", lambda *a, **k: ONI_TEXT)
    frame = load_oni()

    assert frame.index[0] == pd.Timestamp("1997-01-01")
    assert frame.loc["1997-01-01", "season"] == "DJF"
    assert frame.loc["1997-07-01", "season"] == "JJA"
    assert frame.loc["1997-07-01", "oni"] == pytest.approx(1.92)
    assert frame.index.is_monotonic_increasing


def test_oni_temporada_desconocida_falla(monkeypatch) -> None:
    """Una etiqueta de temporada no reconocida levanta ValueError."""
    bad = ONI_TEXT.replace("JJA", "XYZ")
    monkeypatch.setattr(indices, "fetch_text", lambda *a, **k: bad)

    with pytest.raises(ValueError, match="Temporadas no reconocidas"):
        load_oni()


# --------------------------------------------------------------------------- #
# SOI
# --------------------------------------------------------------------------- #


def test_soi_usa_el_bloque_estandarizado(monkeypatch) -> None:
    """Se localiza 'STAND' y se descarta el bloque de anomalias previo."""
    monkeypatch.setattr(indices, "fetch_text", lambda *a, **k: SOI_TEXT)
    series = load_soi()

    assert series.loc["1993-01-01"] == pytest.approx(-0.5)
    assert len(series) == 12


def test_soi_sin_bloque_estandarizado_falla(monkeypatch) -> None:
    """Si el archivo cambia y ya no trae 'STAND', se falla en vez de adivinar.

    Los scripts originales caian a un ``skiprows=87`` fijo, que leeria datos
    equivocados en silencio.
    """
    monkeypatch.setattr(indices, "fetch_text", lambda *a, **k: "1993 1.0\n")

    with pytest.raises(ValueError, match="STAND"):
        load_soi()


# --------------------------------------------------------------------------- #
# Cache y consolidacion
# --------------------------------------------------------------------------- #


def test_fetch_text_usa_la_cache_sin_red(tmp_path) -> None:
    """Con la cache presente no se toca la red (una URL invalida no falla)."""
    cache = tmp_path / "cacheado.txt"
    cache.write_text("contenido\n", encoding="utf-8")

    assert fetch_text("http://invalido.invalido/x", cache) == "contenido\n"


def test_master_table_tiene_indice_mensual_continuo(monkeypatch) -> None:
    """La tabla maestra se reindexa a un rango mensual sin huecos."""
    monkeypatch.setattr(
        indices,
        "load_nino_regions",
        lambda *a, **k: pd.DataFrame(
            {"nino12": [0.1, 0.2], "nino3": [0.1, 0.2], "nino4": [0.1, 0.2], "nino34": [0.1, 0.2]},
            index=pd.to_datetime(["1993-01-01", "1993-04-01"]),
        ),
    )
    monkeypatch.setattr(
        indices,
        "load_oni",
        lambda *a, **k: pd.DataFrame(
            {"season": ["DJF"], "sst": [26.5], "oni": [0.16]},
            index=pd.to_datetime(["1993-01-01"]),
        ),
    )
    for name in ("load_pdo", "load_soi", "load_olr", "load_mei", "load_sam"):
        key = name.removeprefix("load_")
        monkeypatch.setattr(
            indices,
            name,
            lambda *a, _key=key, **k: pd.Series(
                [0.0], index=pd.to_datetime(["1993-01-01"]), name=_key
            ),
        )

    table = build_master_table()

    assert list(table.index) == list(pd.date_range("1993-01-01", "1993-04-01", freq="MS"))
    assert table.index.name == "time"
    assert np.isnan(table.loc["1993-02-01", "nino34"])
    assert table.loc["1993-01-01", "oni"] == pytest.approx(0.16)


# --------------------------------------------------------------------------- #
# Red (excluido por defecto)
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_oni_real_detecta_el_evento_de_1997(tmp_path) -> None:
    """Contra el archivo real del CPC, 1997-98 aparece como evento muy fuerte."""
    from enso_chile.events import detect_events

    events = detect_events(load_oni(cache_dir=tmp_path))
    ninos = events[events["type"] == "nino"]
    evento = ninos[ninos["start"].dt.year == 1997]

    assert len(evento) == 1
    assert evento.iloc[0]["intensity"] == "muy_fuerte"


# --------------------------------------------------------------------------- #
# Indices E y C (Takahashi et al. 2011)
# --------------------------------------------------------------------------- #


def test_ec_indices_aplica_las_ecuaciones_3_y_4() -> None:
    t = pd.date_range("2000-01-01", periods=3, freq="MS")
    n12 = pd.Series([1.0, 0.0, 2.0], index=t)
    n4 = pd.Series([0.0, 1.0, -1.0], index=t)
    ec = indices.ec_indices(n12, n4)
    np.testing.assert_allclose(ec["E"], [1.0, -0.5, 2.5])
    np.testing.assert_allclose(ec["C"], [-0.1, 1.7, -1.9])


def test_ec_indices_usa_solo_meses_comunes() -> None:
    n12 = pd.Series([1.0, np.nan, 2.0], index=pd.date_range("2000-01-01", periods=3, freq="MS"))
    n4 = pd.Series([0.5, 0.5], index=pd.date_range("2000-02-01", periods=2, freq="MS"))
    ec = indices.ec_indices(n12, n4)
    assert list(ec.index) == [pd.Timestamp("2000-03-01")]
