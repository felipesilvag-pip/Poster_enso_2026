"""Paso 5: EEOF combinado (M-SSA) con extension AR(p) y varimax estructurado.

1. Canales: las ``config.K_PER_LAYER`` PC de cada capa, centradas y normalizadas por
   capa (24 canales), igual que el EOF combinado.
2. Extension AR(p) de cada canal, ``config.EEOF_EXTENSION`` meses a cada lado,
   para reducir el error de borde (se recorta despues).
3. EEOF con ventana ``L = config.EEOF_WINDOW`` (formulacion climatologica,
   estimador de Broomhead & King).
4. Varimax estructurado (Groth & Ghil 2011, 2015 ap. A) de los ``S`` primeros modos:
   rota sumando la participacion de cada canal sobre los rezagos y conserva los pares.
5. Cada modo rotado va a su banda por fraccion de potencia: senal rapida (ENSO,
   24-84 meses) y lenta (> 84 meses). Sin rotar se agrupan los pares 3-4 y 1-2.

Guarda ``data/processed/componentes_eeof.nc`` e imprime las cifras. Uso::

    python scripts/05_eeof.py
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
from enso_chile.cube import open_cube, save_cube  # noqa: E402
from enso_chile.eeof import eeof, extend_ar, reconstruct_components, spectral_purity, wcorr_matrix  # noqa: E402
from enso_chile.eof import eof_from_dataarray  # noqa: E402
from enso_chile.events import icen_index  # noqa: E402
from enso_chile.indices import ec_indices, load_nino_regions  # noqa: E402
from enso_chile.rotations import structured_varimax  # noqa: E402

COMPONENTES_EEOF_NC = config.DATA_PROCESSED / "componentes_eeof.nc"
BANDAS = {"decadal": (84.0, np.inf), "ENSO": (24.0, 84.0), "cuasi-bienal": (15.0, 24.0), "resto": (0.0, 15.0)}
GRUPOS_SIN_ROTAR = {"rapida": [2, 3], "lenta": [0, 1]}     # pares ENSO (~44 m) y lento del EEOF sin rotar


def corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def resumen(rc: np.ndarray, icen: np.ndarray) -> np.ndarray:
    """Primera direccion principal de un componente (T, canales), en sd, orientada con el ICEN."""
    x = rc - rc.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    f = u[:, 0] * s[0]
    f = f / f.std(ddof=1)
    return f * (np.sign(corr(f, icen)) or 1.0)


def main() -> None:
    """Calcular, guardar e imprimir los componentes EEOF."""
    K, L, M, S = config.K_PER_LAYER, config.EEOF_WINDOW, config.EEOF_EXTENSION, config.EEOF_ROTATED
    cubo = open_cube(config.CUBE_ANOM)["thetao_anom"].load()
    t = pd.DatetimeIndex(cubo.time.values)
    T = len(t)
    res = [eof_from_dataarray(cubo.sel(depth=z), n_modes=K, scaling="orthonormal") for z in config.DEPTHS_WORK]
    retenida = np.array([r.variance_fraction[:K].sum() for r in res])
    bloques = []
    for r in res:
        B = r.pcs[:, :K] - r.pcs[:, :K].mean(0)
        bloques.append(B / np.sqrt((B ** 2).sum() / (T - 1)))
    Z = np.hstack(bloques)

    def pct_capa(rc: np.ndarray) -> np.ndarray:
        """% de la varianza de cada capa: canales normalizados por la fraccion retenida."""
        v = rc.var(0, ddof=1)
        return np.array([100 * v[d * K:(d + 1) * K].sum() * retenida[d] for d in range(len(res))])

    ZE = extend_ar(Z, M, p_max=config.AR_MAX_ORDER)
    NE, C = ZE.shape
    rec = slice(M, M + T)
    lam, EE, PC = eeof(ZE, L, n_modes=S)
    E_flat = EE.transpose(0, 2, 1).reshape(S, L * C).T
    _, R = structured_varimax(E_flat * np.sqrt(lam * PC.shape[0]), n_channels=C, n_lags=L)
    EE_R, PC_R = (E_flat @ R).T.reshape(S, L, C).transpose(0, 2, 1), PC @ R

    def banda(k: int) -> str:
        p = spectral_purity(reconstruct_components(EE_R, PC_R, [k], NE)[rec], BANDAS)
        return max(p, key=p.get)

    bandas = [banda(k) for k in range(S)]
    grupos = {"Sin rotar": (EE, PC, GRUPOS_SIN_ROTAR),
              "Varimax estructurado": (EE_R, PC_R, {"rapida": [k for k in range(S) if bandas[k] == "ENSO"],
                                                    "lenta": [k for k in range(S) if bandas[k] == "decadal"]})}
    print("Banda de cada modo rotado:", bandas)

    nino = load_nino_regions(cache_dir=config.INDICES_CACHE)
    s3 = nino.rolling(3, center=True).mean()
    ec = ec_indices(s3["nino12"], s3["nino4"])
    icen = icen_index(nino["nino12"]).reindex(t).to_numpy()
    E, Cc = ec["E"].reindex(t).to_numpy(), ec["C"].reindex(t).to_numpy()

    filas = []
    for met, (ee, pc, g) in grupos.items():
        rcs = {}
        for comp, modos in g.items():
            rc = reconstruct_components(ee, pc, modos, NE)
            rcs[comp] = rc
            rc = rc[rec]
            f = resumen(rc, icen)
            pur = spectral_purity(rc, BANDAS)
            filas.append(dict(metodo=met, comp=comp, modos=" ".join(str(m + 1) for m in modos), serie=f,
                              capa=pct_capa(rc), pureza_enso=pur["ENSO"], pureza_decadal=pur["decadal"],
                              rICEN=corr(f, icen), rE=corr(f, E), rC=corr(f, Cc)))
        w = wcorr_matrix(np.stack([rcs["rapida"], rcs["lenta"]]), L)[0, 1]
        print(f"{met}: W(rapida, lenta) = {w:.2f}")

    ds = xr.Dataset(
        {"serie": (("comp", "time"), np.stack([f["serie"] for f in filas])),
         "capa": (("comp", "depth"), np.stack([f["capa"] for f in filas])),
         **{k: ("comp", [f[k] for f in filas]) for k in ("pureza_enso", "pureza_decadal", "rICEN", "rE", "rC")}},
        coords={"comp": np.arange(len(filas)), "metodo": ("comp", [f["metodo"] for f in filas]),
                "nombre": ("comp", [f["comp"] for f in filas]), "modos": ("comp", [f["modos"] for f in filas]),
                "time": t, "depth": list(config.DEPTHS_WORK)},
        attrs={"L": L, "extension_AR": M, "S": S, "capa": "% de la varianza de cada capa",
               "serie": "primera direccion principal del componente, en sd, orientada con el ICEN"})
    save_cube(ds, COMPONENTES_EEOF_NC, overwrite=True)

    tabla = pd.DataFrame({"modos": ds.modos.values,
                          **{f"% {z:.0f} m": ds.capa.values[:, d] for d, z in enumerate(config.DEPTHS_WORK)},
                          "pureza ENSO": ds.pureza_enso.values, "r ICEN": ds.rICEN.values,
                          "r E": ds.rE.values, "r C": ds.rC.values},
                         index=[f"{m} {n}" for m, n in zip(ds.metodo.values, ds.nombre.values)])
    print(tabla.round(2).to_string())
    print(f"Guardado: {COMPONENTES_EEOF_NC}")


if __name__ == "__main__":
    main()
