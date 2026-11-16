# Póster ENSO 2026: anomalías térmicas en la costa norte de Chile

Código para reproducir los resultados del póster *Caracterización espacio-temporal
de anomalías térmicas asociadas a eventos El Niño intensos mediante funciones
ortogonales empíricas (EOF) y su versión extendida (EEOF) en la costa norte de
Chile* (Ingeniería Estadística, Universidad de Santiago de Chile).

> En construcción: los scripts de reproducción se agregan por etapas.

## Página interactiva

<https://felipesilvag-pip.github.io/Poster_enso_2026/>

## Datos

GLORYS12 (Copernicus Marine Service) e índices de TSM Niño del CPC/NOAA. Los datos
**no** se incluyen en el repositorio: cada usuario los descarga con su propia cuenta
gratuita de Copernicus Marine (ver *Instalación*).

## Instalación

```bash
conda create -n poster_enso python=3.10
conda activate poster_enso
conda install -c conda-forge cartopy
pip install -r requirements.txt
copernicusmarine login        # credenciales propias; nunca van en el código
```

## Licencia

MIT (ver [LICENSE](LICENSE)).
