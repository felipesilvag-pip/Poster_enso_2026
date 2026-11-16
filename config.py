"""Configuracion unica del repositorio del poster ENSO 2026.

Unico lugar donde se declaran rutas, caja geografica, profundidades, periodos y
parametros de los metodos. Los scripts y el paquete ``enso_chile`` importan desde
aqui (``import config``); ningun otro archivo repite estos valores.

Las credenciales de Copernicus Marine NO viven aqui: cada usuario se autentica
con ``copernicusmarine login``, que las guarda en su propio equipo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Rutas (todas relativas a la raiz del repositorio)
# --------------------------------------------------------------------------- #

ROOT: Final[Path] = Path(__file__).resolve().parent
SRC: Final[Path] = ROOT / "src"
DATA_RAW: Final[Path] = ROOT / "data" / "raw"
DATA_PROCESSED: Final[Path] = ROOT / "data" / "processed"
FIGURES: Final[Path] = ROOT / "figures"
DOCS: Final[Path] = ROOT / "docs"

# --------------------------------------------------------------------------- #
# Producto Copernicus Marine
# --------------------------------------------------------------------------- #

#: Reanalisis GLORYS12, medias mensuales, 1/12 de grado.
DATASET_ID: Final[str] = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

#: Variable usada en el poster: temperatura potencial.
VARIABLES: Final[tuple[str, ...]] = ("thetao",)

# --------------------------------------------------------------------------- #
# Caja de estudio y profundidades
# --------------------------------------------------------------------------- #

#: Norte Grande de Chile: 18-27 S, desde la costa hasta 74 W (~200 km). Es la caja
#: del poster. Norte Chico (27-32.5 S) queda declarada para la extension futura y
#: para los tests del limite entre cajas; nunca se descomponen juntas.
BOXES: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "norte_grande": {"lat": (-27.0, -18.0), "lon": (-74.0, -69.0)},
    "norte_chico": {"lat": (-32.5, -27.0), "lon": (-74.0, -69.0)},
}
BOX: Final[str] = "norte_grande"

#: Niveles reales del modelo usados como capas (m): 0.5, 25, 92 y 156 m.
DEPTHS_WORK: Final[tuple[float, ...]] = (0.494, 25.211, 92.326, 155.851)

#: Rango vertical de la descarga (m); incluye las cuatro capas.
DEPTH_RANGE: Final[tuple[float, float]] = (0.0, 160.0)

# --------------------------------------------------------------------------- #
# Periodos
# --------------------------------------------------------------------------- #

#: Periodo de ajuste: anomalias, EOF, rotaciones y EEOF (399 meses).
PERIOD_FULL: Final[tuple[str, str]] = ("1993-01-01", "2026-03-01")

#: Periodo descargado: el de ajuste mas los meses que se proyectan (El Nino 2026).
PERIOD_DOWNLOAD: Final[tuple[str, str]] = ("1993-01-01", "2026-08-01")

#: Cubo crudo descargado por ``scripts/01_descargar_datos.py``.
CUBE_RAW: Final[Path] = DATA_RAW / "glorys12_thetao_norte_grande_1993-01_2026-08.nc"

#: Solo para el mapa de la zona de estudio (fig01): temperatura a 0.5 m en un
#: dominio mas amplio (18-33.5 S, 78-69 W) sobre ``PERIOD_FULL``.
BOX_MAP: Final[dict[str, tuple[float, float]]] = {"lat": (-33.5, -18.0), "lon": (-78.0, -69.0)}
CUBE_MAP: Final[Path] = DATA_RAW / "glorys12_thetao_0.5m_mapa_1993-01_2026-03.nc"

#: Indices de TSM Nino 1+2, 3, 3.4 y 4 (CPC/NOAA), en cache local.
INDICES_CACHE: Final[Path] = DATA_RAW / "indices"

#: Cubo de anomalias de las cuatro capas sobre ``PERIOD_FULL`` (script 02).
CUBE_ANOM: Final[Path] = DATA_PROCESSED / "anom_thetao_norte_grande.nc"

#: Tabla de eventos ICEN (script 03).
EVENTS_CSV: Final[Path] = DATA_PROCESSED / "eventos_icen.csv"

# --------------------------------------------------------------------------- #
# Anomalias y ponderacion
# --------------------------------------------------------------------------- #

#: Ajuste por pixel por minimos cuadrados: constante + tendencia lineal + 12
#: efectos mensuales, sobre ``PERIOD_FULL``. La anomalia es el residuo.
ANOMALY_DETREND: Final[bool] = True

#: Ponderacion por area en todos los EOF/EEOF: sqrt(cos(lat)).
AREA_WEIGHTING: Final[str] = "sqrt_coslat"

# --------------------------------------------------------------------------- #
# Eventos (criterio ICEN: ENFEN 2012; Takahashi et al. 2014)
# --------------------------------------------------------------------------- #

#: Umbral |pico| del ICEN para los eventos de estudio.
EVENT_MIN_PEAK: Final[float] = 1.5

# --------------------------------------------------------------------------- #
# Parametros de los metodos
# --------------------------------------------------------------------------- #

#: PC por capa en el prefiltro del EOF combinado (ultimo modo separado por North).
K_PER_LAYER: Final[int] = 6

#: Modos rotados del EOF combinado (saltos significativos de North).
K_ROTATED: Final[int] = 4

#: Ventana del EEOF (meses).
EEOF_WINDOW: Final[int] = 72

#: Meses de extension AR(p) a cada lado antes del EEOF (= ventana).
EEOF_EXTENSION: Final[int] = 72

#: Orden maximo del AR(p) por canal (elegido por AIC).
AR_MAX_ORDER: Final[int] = 24

#: Modos del EEOF rotados con el varimax estructurado.
EEOF_ROTATED: Final[int] = 10

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #

RANDOM_SEED: Final[int] = 42

#: Prohibido submuestrear pixeles al azar en resultados reportables.
ALLOW_PIXEL_SUBSAMPLING: Final[bool] = False
