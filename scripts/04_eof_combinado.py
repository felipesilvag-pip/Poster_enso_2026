"""Paso 4: EOF combinado por profundidad, rotaciones varimax y bifactor, y proyeccion 2026.

1. EOF por capa (``config.K_PER_LAYER`` PC, escalado ortonormal), cada capa
   normalizada a varianza 1, concatenacion y SVD: el EOF combinado.
2. Rotacion de sus ``config.K_ROTATED`` primeros modos con pixeles como filas:
   varimax (Kaiser) y bifactor bi-quartimin ortogonal (Jennrich & Bentler 2011).
3. Proyeccion de los meses posteriores a ``config.PERIOD_FULL`` (2026-04..08)
   sobre los ejes congelados: misma anomalia, mismas EOF, mismas escalas.

Guarda ``data/processed/componentes.nc`` (series, R2 por capa, % del total y
mapas de carga de los 12 componentes) e imprime la tabla del poster. Uso::

    python scripts/04_eof_combinado.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import config  # noqa: E402
from enso_chile.cube import open_cube, save_cube, select_box, select_depths  # noqa: E402
from enso_chile.eof import combined_eof_from_results, eof_from_dataarray, pixel_loadings  # noqa: E402
from enso_chile.indices import ec_indices, load_nino_regions  # noqa: E402
from enso_chile.poster import (frozen_anomalies, loading_maps, name_by_profile, orient,  # noqa: E402
                               prefilter_blocks, project_scores)
from enso_chile.rotations import bifactor, varimax  # noqa: E402

COMPONENTES_NC = config.DATA_PROCESSED / "componentes.nc"


def meses_nuevos(t_fin: pd.Timestamp) -> xr.DataArray:
    """Anomalias congeladas de los meses descargados despues del periodo de ajuste."""
    crudo = open_cube(config.CUBE_RAW)["thetao"]
    capas = select_depths(select_box(crudo, config.BOXES[config.BOX]), config.DEPTHS_WORK)
    hist = capas.sel(time=slice(*config.PERIOD_FULL)).load()
    nuevo = capas.sel(time=capas.time > np.datetime64(t_fin)).load()
    return xr.concat([frozen_anomalies(hist.sel(depth=z), nuevo.sel(depth=z), config.ANOMALY_DETREND)
                      for z in config.DEPTHS_WORK], dim="depth").transpose("time", "depth", ...)


def indices(t: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """E y C de Takahashi et al. (2011) con la media de 3 meses del ICEN."""
    s3 = load_nino_regions(cache_dir=config.INDICES_CACHE).rolling(3, center=True).mean()
    ec = ec_indices(s3["nino12"], s3["nino4"])
    return {"E": ec["E"].reindex(t).to_numpy(), "C": ec["C"].reindex(t).to_numpy()}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main() -> None:
    """Calcular, guardar e imprimir los componentes del poster."""
    K, k = config.K_PER_LAYER, config.K_ROTATED
    cubo = open_cube(config.CUBE_ANOM)["thetao_anom"].load()
    t_h = pd.DatetimeIndex(cubo.time.values)
    res = [eof_from_dataarray(cubo.sel(depth=z), n_modes=K, scaling="orthonormal") for z in config.DEPTHS_WORK]
    cmb = combined_eof_from_results(res, K)
    A, F = cmb.loadings(k), cmb.scores(k)
    Lp = pixel_loadings(A, res, K)
    total = cmb.eigenvalues.sum()

    metodos = {"EOF": np.eye(k), "Varimax": varimax(Lp, normalize=True)[1],
               "Bifactor": bifactor(Lp, n_starts=20, random_state=0)[1]}

    nuevos = meses_nuevos(t_h[-1])
    t_all = t_h.append(pd.DatetimeIndex(nuevos.time.values))
    Z = np.vstack([prefilter_blocks(c, config.DEPTHS_WORK, res, cmb.block_scales, K) for c in (cubo, nuevos)])
    P0 = project_scores(Z, A)
    print(f"Control: la proyeccion del periodo de ajuste reproduce las PC (max |dif| "
          f"{np.abs(P0[:len(t_h)] - F).max():.1e})")

    idx = indices(t_h)
    filas = []
    for met, T in metodos.items():
        Ar = A @ T
        r2 = 100 * cmb.r2_by_block(Ar)
        if met == "EOF":
            nombres = [f"Modo {j + 1}" for j in range(k)]
        elif met == "Varimax":
            nombres = name_by_profile(r2)
        else:
            nombres = ["General"] + name_by_profile(r2[1:])
        mapas = loading_maps(Ar, res, cmb.block_scales, K)
        P = P0 @ T
        for j, nombre in enumerate(nombres):
            s = orient(mapas[j], r2[j])
            ph = P[:len(t_h), j]
            serie = s * (P[:, j] - ph.mean()) / ph.std(ddof=1)
            filas.append(dict(metodo=met, nombre=nombre, serie=serie, r2=r2[j],
                              tot=100 * float((Ar[:, j] ** 2).sum() / total),
                              mapa=np.stack([s * m.values for m in mapas[j]]),
                              rE=corr(serie[:len(t_h)], idx["E"]), rC=corr(serie[:len(t_h)], idx["C"])))

    m0 = loading_maps(A[:, :1], res, cmb.block_scales, K)[0][0]
    ds = xr.Dataset(
        {"serie": (("comp", "time"), np.stack([f["serie"] for f in filas])),
         "r2": (("comp", "depth"), np.stack([f["r2"] for f in filas])),
         "tot": ("comp", [f["tot"] for f in filas]),
         "rE": ("comp", [f["rE"] for f in filas]),
         "rC": ("comp", [f["rC"] for f in filas]),
         "mapa": (("comp", "depth", "latitude", "longitude"), np.stack([f["mapa"] for f in filas]))},
        coords={"comp": np.arange(len(filas)), "metodo": ("comp", [f["metodo"] for f in filas]),
                "nombre": ("comp", [f["nombre"] for f in filas]), "time": t_all,
                "depth": list(config.DEPTHS_WORK), "latitude": m0.latitude.values,
                "longitude": m0.longitude.values},
        attrs={"periodo_ajuste": f"{t_h[0]:%Y-%m}/{t_h[-1]:%Y-%m}",
               "serie": "puntaje estandarizado con la media y sd del periodo de ajuste; meses "
                        "posteriores proyectados sobre los ejes congelados",
               "mapa": "carga en grados C por sd de la serie",
               "r2": "% de la varianza de cada capa", "tot": "% de la varianza combinada"})
    save_cube(ds, COMPONENTES_NC, overwrite=True)

    tabla = pd.DataFrame({"% total": ds.tot.values,
                          **{f"R2 {z:.0f} m": ds.r2.values[:, d] for d, z in enumerate(config.DEPTHS_WORK)},
                          "r E": ds.rE.values, "r C": ds.rC.values},
                         index=[f"{m} {n}" for m, n in zip(ds.metodo.values, ds.nombre.values)])
    print(tabla.round(2).to_string())
    print(f"Guardado: {COMPONENTES_NC} ({t_all[0]:%Y-%m} a {t_all[-1]:%Y-%m})")


if __name__ == "__main__":
    main()
