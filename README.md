# Póster ENSO 2026: anomalías térmicas en la costa norte de Chile

Código para reproducir los resultados del póster *Caracterización espacio-temporal
de anomalías térmicas asociadas a eventos El Niño intensos mediante funciones
ortogonales empíricas (EOF) y su versión extendida (EEOF) en la costa norte de
Chile* (Felipe Silva Guevara, Ingeniería Estadística, Universidad de Santiago de
Chile, 2026).

**Página interactiva:** <https://felipesilvag-pip.github.io/Poster_enso_2026/>
(los 12 componentes con sus mapas por capa y El Niño 2026 proyectado).

## Objetivo

El Niño y La Niña alteran la temperatura de toda la columna de agua frente al norte
de Chile, pero su señal se mezcla con variabilidad más lenta y cambia con la
profundidad. Se descomponen en conjunto cuatro capas (0.5, 25, 92 y 156 m) de
Norte Grande (18–27 °S, costa a 74 °W) para:

1. separar la señal ENSO de la variabilidad lenta,
2. localizar en qué profundidad domina cada una (EOF combinado y sus rotaciones),
3. distinguir sus ritmos de corto y largo plazo (EEOF), y
4. caracterizar la firma de cada evento intenso.

## Métodos (resumen)

| Paso | Método |
|---|---|
| Anomalías | Por píxel: residuo de constante + tendencia + 12 meses (mínimos cuadrados, 1993-01 a 2026-03); ponderación √cos φ |
| Eventos | Criterio ICEN sobre Niño 1+2 (ENFEN 2012; Takahashi et al. 2014); eventos de estudio con \|pico\| ≥ 1.5 °C |
| EOF combinado | 6 PC por capa, cada capa normalizada a varianza 1, concatenación y SVD |
| Rotaciones | 4 modos (North et al. 1982), píxeles como filas: varimax y bifactor bi-quartimin (Jennrich & Bentler 2011) |
| EEOF | M-SSA con ventana L = 72 meses, extensión AR(p) de 72 meses a cada lado, varimax estructurado de los 10 primeros modos (Groth & Ghil 2011, 2015) |
| Índices | E y C de Takahashi et al. (2011): Pacífico oriental y central |
| El Niño 2026 | Meses nuevos proyectados sobre los ejes congelados (sin reajustar anomalías, EOF ni rotaciones) |

## Instalación

Requiere Python 3.10 y una cuenta gratuita de
[Copernicus Marine](https://data.marine.copernicus.eu/register).

```bash
git clone https://github.com/felipesilvag-pip/Poster_enso_2026.git
cd Poster_enso_2026
conda create -n poster_enso python=3.10
conda activate poster_enso
conda install -c conda-forge cartopy
pip install -r requirements.txt
copernicusmarine login        # credenciales propias; quedan en tu equipo, nunca en el código
```

## Reproducir los resultados

Ejecutar en orden desde la raíz del repositorio:

| Script | Qué hace | Salida |
|---|---|---|
| `scripts/01_descargar_datos.py` | Descarga GLORYS12 (~150 MB en total) y los índices Niño del CPC/NOAA | `data/raw/` |
| `scripts/02_anomalias.py` | Anomalías de las 4 capas | `data/processed/anom_thetao_norte_grande.nc` |
| `scripts/03_eventos_icen.py` | Eventos ICEN y su figura | `eventos_icen.csv`, `figures/fig03_icen.png` |
| `scripts/04_eof_combinado.py` | EOF combinado, varimax, bifactor y proyección 2026 | `data/processed/componentes.nc` |
| `scripts/05_eeof.py` | EEOF con extensión AR y varimax estructurado | `data/processed/componentes_eeof.nc` |
| `scripts/06_figuras_poster.py` | Figuras del póster y puntajes por evento | `figures/` |
| `scripts/07_pagina_web.py` | Página interactiva | `docs/index.html` |

Los tests de las funciones (222) se corren con `pytest`.

## Resultados principales

- **EOF combinado:** el modo 1 explica el 50.1 % de la varianza combinada
  (45 / 42 / 27 / 29 % de cada capa) y mezcla el ENSO con la variabilidad lenta.
- **Varimax** separa por profundidad: un componente de superficie (51 % de la
  varianza a 0.5 m, r = 0.53 con E) y uno de fondo (35 % a 156 m).
- **Bifactor:** un factor general, el ENSO de cuenca en toda la columna (46 % del
  total; r = 0.63 con E y 0.51 con C), más componentes propios de superficie,
  92 m y fondo.
- **EEOF con varimax estructurado:** una señal rápida (ENSO, ~3–4 años;
  21 / 21 / 10 / 11 % de cada capa) y una lenta, más fuerte a 156 m (14 %).
- **Eventos** (puntaje medio del factor general, en sd): 1997–98 +2.5 (toda la
  columna), 2023–24 +1.6 (intenso pero somero, fondo −1.1), 2015–16 +1.0.

Los resultados son descriptivos.

### Extensión exploratoria de El Niño 2026

GLORYS12 llega hasta 2026-08. Para mostrar septiembre y octubre de 2026, la página
usa el producto de análisis y pronóstico de Copernicus (*anfc*), reajustado por su
sesgo medio por píxel y capa frente a GLORYS12 en los meses en que ambos coinciden
(`scripts/04_eof_combinado.py`). Esos meses son **exploratorios**: no entran en las
figuras del póster y se reemplazarán cuando GLORYS12 los publique.

## Estructura

```
├── config.py            rutas, caja, capas, periodos y parámetros (único lugar)
├── scripts/             pasos 01–07
├── src/enso_chile/      funciones (anomalías, EOF, EEOF, rotaciones, eventos, índices)
├── tests/               tests de esas funciones (pytest)
├── docs/index.html      página interactiva (GitHub Pages)
├── data/                datos descargados y productos (no versionados)
└── figures/             figuras generadas (no versionadas)
```

## Datos

- **GLORYS12**, reanálisis global de Copernicus Marine Service
  (`cmems_mod_glo_phy_my_0.083deg_P1M-m`), y su producto de análisis y pronóstico
  (`cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m`). Los datos no se redistribuyen
  aquí: cada usuario los descarga con su cuenta.
- **Índices de TSM Niño** (1+2, 3, 3.4, 4) del Climate Prediction Center (CPC/NOAA).

## Referencias

- ENFEN (2012). Definición operacional del ICEN. Comité Multisectorial ENFEN, Perú.
- Groth, A. & Ghil, M. (2011). *Phys. Rev. E* 84, 036206.
- Groth, A. & Ghil, M. (2015). *J. Climate* 28, 7873–7893.
- Jennrich, R. I. & Bentler, P. M. (2011). *Psychometrika* 76, 537–549.
- North, G. R. et al. (1982). *Mon. Wea. Rev.* 110, 699–706.
- Takahashi, K. et al. (2011). *Geophys. Res. Lett.* 38, L10704.
- Takahashi, K., Mosquera, K. & Reupo, J. (2014). *Bol. Téc. IGP* 1(2), 8–9.

## Licencia

Código bajo licencia MIT (ver [LICENSE](LICENSE)).
