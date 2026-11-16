"""Paso 6: figuras y tabla de puntajes del poster.

Lee lo calculado por los scripts 03-05 y escribe en ``figures/``:

- ``fig01_zona_estudio.png``: temperatura media a 0.5 m y cajas de estudio.
- ``fig_fichas_8componentes.png``: EOF modos 1-2, varimax (superficie, fondo,
  columna) y bifactor (general, superficie, fondo): mapa de carga en la capa mas
  fuerte, serie mensual frente al ICEN y R2 por capa.
- ``fig_eeof_sv.png``: senal rapida y lenta del EEOF, sin rotar y con varimax
  estructurado.
- ``puntajes_eventos.csv``: puntaje medio de cada componente en cada evento de
  estudio (duracion ICEN), en sd.

(La figura del ICEN, ``fig03_icen.png``, la genera el script 03.) Uso::

    python scripts/06_figuras_poster.py
"""

from __future__ import annotations

import ssl
import sys
from pathlib import Path

import certifi

# cartopy baja las costas de Natural Earth por HTTPS la primera vez; certifi evita
# fallar en equipos con el almacen de certificados del sistema danado.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import config  # noqa: E402
from enso_chile.events import icen_index  # noqa: E402
from enso_chile.indices import load_nino_regions  # noqa: E402

ETIQ = ["0.5 m", "25 m", "92 m", "156 m"]
NEGRO, ROJO, AZUL, MORADO, VERDE, GRIS = "#222222", "#d62728", "#1f77b4", "#7b3294", "#2ca02c", "#8c6d31"
FICHAS = [("EOF", "Modo 1", "EOF modo 1", NEGRO), ("EOF", "Modo 2", "EOF modo 2", GRIS),
          ("Varimax", "Superficie", "Varimax superficie", ROJO), ("Varimax", "Fondo", "Varimax fondo", AZUL),
          ("Varimax", "Columna", "Varimax columna", VERDE), ("Bifactor", "General", "Bifactor general", MORADO),
          ("Bifactor", "Superficie", "Bifactor superficie", ROJO), ("Bifactor", "Fondo", "Bifactor fondo", AZUL)]
LON_FMT = mticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}°O")
LAT_FMT = mticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}°S")
CIUDADES = {"Arica": (-70.32, -18.48), "Iquique": (-70.15, -20.21), "Antofagasta": (-70.40, -23.65),
            "Caldera": (-70.83, -27.07), "Coquimbo": (-71.34, -29.95)}


# ----------------------------------------------------------------------------- datos comunes
def icen_estandarizado(t: pd.DatetimeIndex) -> np.ndarray:
    icen = icen_index(load_nino_regions(cache_dir=config.INDICES_CACHE)["nino12"]).reindex(t).to_numpy()
    return (icen - np.nanmean(icen)) / np.nanstd(icen)


def eventos_estudio() -> pd.DataFrame:
    ev = pd.read_csv(config.EVENTS_CSV, parse_dates=["start", "end", "peak_time"])
    return ev[ev["estudio"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- fig01
def figura_zona(destino: Path) -> None:
    """Temperatura media a 0.5 m (1993-2026) con las cajas Norte Grande y Norte Chico."""
    with xr.open_dataset(config.CUBE_MAP) as ds:
        sst = ds["thetao"].isel(depth=0).mean("time").load()
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(5.2, 7.0), subplot_kw={"projection": crs})
    ax.set_extent([-79.0, -68.0, -34.0, -16.5], crs=crs)
    pc = ax.pcolormesh(sst.longitude, sst.latitude, sst, cmap="RdYlBu_r", shading="auto", transform=crs, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.88", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, edgecolor="0.5", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    for clave, nombre, ls, col in [("norte_grande", "Norte Grande", "-", "k"),
                                   ("norte_chico", "Norte Chico\n(extensión)", "--", "0.3")]:
        b = config.BOXES[clave]
        ax.add_patch(Rectangle((b["lon"][0], b["lat"][0]), b["lon"][1] - b["lon"][0], b["lat"][1] - b["lat"][0],
                               transform=crs, fill=False, lw=1.8, edgecolor=col, linestyle=ls, zorder=3))
        ax.text(b["lon"][0] - 0.3, sum(b["lat"]) / 2, nombre, transform=crs, color=col, fontsize=9,
                fontweight="bold", ha="right", va="center", rotation=90, zorder=4)
    for nombre, (lon, lat) in CIUDADES.items():
        ax.plot(lon, lat, "o", ms=3, color="k", transform=crs, zorder=5)
        ax.text(lon + 0.25, lat, nombre, transform=crs, fontsize=7.5, va="center", zorder=5)
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85)
    cb.set_label("Temperatura potencial media a 0.5 m (°C)\n1993-2026")
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- fichas
def figura_fichas(comp: xr.Dataset, ev: pd.DataFrame, icen_z: np.ndarray, destino: Path,
                  hm: float = 2.8, ncol: int = 2, ws: float = 7.2, gf: float = 0.55, gc: float = 0.7) -> None:
    """Una ficha por componente: mapa en su capa mas fuerte | serie + ICEN | R2 por capa."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    t = pd.DatetimeIndex(comp.time.values)
    lon, lat = comp.longitude.values, comp.latitude.values
    lon0, lon1, lat0, lat1 = lon.min(), lon.max(), lat.min(), lat.max()
    wm = hm / ((lat1 - lat0) / ((lon1 - lon0) * np.cos(np.deg2rad(22.5))))
    izq, g1, g2, wp, der = 0.75, 0.55, 0.55, 1.9, 0.2
    top, abajo = 0.75 + 0.25 * (ncol > 1), 1.0
    sel = [int(np.flatnonzero((comp.metodo.values == m) & (comp.nombre.values == n))[0]) for m, n, _, _ in FICHAS]
    n = len(sel)
    nf = int(np.ceil(n / ncol))
    Wc = izq + wm + g1 + ws + g2 + wp + der
    W, H = ncol * Wc + (ncol - 1) * gc, top + nf * hm + (nf - 1) * gf + abajo
    fig = plt.figure(figsize=(W, H))

    def eje(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    zmax = {c: int(np.argmax(comp.r2.values[c])) for c in sel}
    vmax = max(float(np.nanmax(np.abs(comp.mapa.values[c, zmax[c]]))) for c in sel)
    YM = 1.05 * max([float(np.nanmax(np.abs(comp.serie.values[c]))) for c in sel] + [float(np.nanmax(np.abs(icen_z)))])
    for i_, (c, (_, _, titulo, col)) in enumerate(zip(sel, FICHAS)):
        cc, i = divmod(i_, nf)
        ultima = i == min(nf, n - cc * nf) - 1
        xo, y0 = cc * (Wc + gc), H - top - (i + 1) * hm - i * gf
        z = zmax[c]
        ax = eje(xo + izq, y0, wm, hm)
        pc = ax.pcolormesh(lon, lat, comp.mapa.values[c, z], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_facecolor("0.8")
        ax.set(xlim=(lon0, lon1), ylim=(lat0, lat1))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.xaxis.set_major_formatter(LON_FMT)
        ax.yaxis.set_major_formatter(LAT_FMT)
        ax.tick_params(labelsize=7, pad=1)
        ax.set_title(ETIQ[z], fontsize=9, pad=2)
        if not ultima:
            ax.set_xticklabels([])
        axs = eje(xo + izq + wm + g1, y0, ws, hm)
        for e in ev.itertuples():
            axs.axvspan(e.start, e.end, color="0.9" if e.type == "nino" else "#e3ecf7", lw=0, zorder=0)
            if i == 0:
                nino = e.type == "nino"
                axs.text(e.peak_time, 1.27 if nino else 1.15, e.label, transform=axs.get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=9, color="0.25" if nino else "#2b5d9e")
        axs.plot(t, icen_z, color="0.7", lw=1.0)
        axs.plot(t, comp.serie.values[c], color=col, lw=0.8)
        axs.axhline(0, color="0.6", lw=0.4)
        axs.set(xlim=(t[0], t[-1]), ylim=(-YM, YM))
        axs.yaxis.set_major_locator(mticker.MultipleLocator(2))
        axs.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        axs.tick_params(axis="y", labelsize=9)
        axs.set_title(f"{titulo}:  {float(comp.tot[c]):.1f} % del total", loc="left", fontsize=11,
                      fontweight="bold", color=col, pad=3)
        axs.set_title(f"r E = {float(comp.rE[c]):.2f}  ·  r C = {float(comp.rC[c]):.2f}", loc="right",
                      fontsize=9.5, color=col, pad=3)
        if not ultima:
            axs.set_xticklabels([])
        axp = eje(xo + izq + wm + g1 + ws + g2, y0, wp, hm)
        r2 = comp.r2.values[c]
        axp.barh(np.arange(4), r2, color=col, height=0.65)
        for k_, x in enumerate(r2):
            axp.text(x + 1, k_, f"{x:.0f}", va="center", fontsize=8.5)
        axp.set(yticks=range(4), yticklabels=ETIQ, xlim=(0, 60))
        axp.invert_yaxis()
        axp.tick_params(labelsize=8.5)
        if i == 0:
            axp.set_title("$R^2$ por capa (%)", fontsize=9.5, pad=3)
        if not ultima:
            axp.set_xticklabels([])
    cb = fig.colorbar(pc, cax=eje(izq - 0.3, 0.35, wm + 0.9, 0.12), orientation="horizontal")
    cb.set_label("°C por sd de la PC", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)
    fig.legend(handles=[Line2D([], [], color="0.3", lw=1.0, label="PC estandarizada (sd)"),
                        Line2D([], [], color="0.7", lw=1.2, label="ICEN estandarizado"),
                        Patch(color="0.9", label="El Niño"), Patch(color="#e3ecf7", label="La Niña")],
               loc="lower left", bbox_to_anchor=((izq + wm + g1 + 0.5) / W, 0.25 / H), ncol=4, frameon=False,
               fontsize=9.5)
    fig.savefig(destino, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- EEOF
def figura_eeof(eeof: xr.Dataset, ev: pd.DataFrame, icen_z: np.ndarray, destino: Path) -> None:
    """Senal rapida y lenta: series frente al ICEN y % de la varianza de cada capa."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    t = pd.DatetimeIndex(eeof.time.values)
    col, nombre = {"rapida": "C3", "lenta": "C0"}, {"rapida": "Rápida (ENSO)", "lenta": "Lenta"}
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.35, wspace=0.18)
    for i, met in enumerate(("Sin rotar", "Varimax estructurado")):
        ax = fig.add_subplot(gs[i, 0])
        for e in ev.itertuples():
            ax.axvspan(e.start, e.end, color="0.88", lw=0, zorder=0)
            if i == 0:
                ax.text(e.peak_time, 1.0, e.label, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=9)
        ax.plot(t, icen_z, color="0.6", lw=0.8, label="ICEN")
        axb = fig.add_subplot(gs[i, 1])
        for k_, comp in enumerate(("rapida", "lenta")):
            c = int(np.flatnonzero((eeof.metodo.values == met) & (eeof.nombre.values == comp))[0])
            ax.plot(t, eeof.serie.values[c], color=col[comp], lw=1.5, label=nombre[comp])
            axb.barh(np.arange(4) + (k_ - 0.5) * 0.38, eeof.capa.values[c], height=0.38, color=col[comp])
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set(xlim=(t[0], t[-1]), ylabel="Estandarizado")
        ax.set_title(met, loc="left", fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=9)
        axb.set(yticks=np.arange(4), yticklabels=ETIQ, xlabel="% de la varianza de la capa")
        axb.invert_yaxis()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- puntajes
def puntajes(comp: xr.Dataset, ev: pd.DataFrame) -> pd.DataFrame:
    """Media de cada serie durante cada evento de estudio (de inicio a fin de la racha ICEN)."""
    t = pd.DatetimeIndex(comp.time.values)
    filas = {}
    for c in range(comp.sizes["comp"]):
        s = pd.Series(comp.serie.values[c], index=t)
        filas[f"{comp.metodo.values[c]} {comp.nombre.values[c]}"] = {
            e.label: s.loc[e.start:e.end].mean() for e in ev.itertuples()}
    return pd.DataFrame(filas).T


def main() -> None:
    """Generar las figuras y la tabla de puntajes."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    ev = eventos_estudio()
    comp = xr.open_dataset(config.DATA_PROCESSED / "componentes.nc").load()
    comp_h = comp.sel(time=slice(*config.PERIOD_FULL))
    eeof = xr.open_dataset(config.DATA_PROCESSED / "componentes_eeof.nc").load()
    icen_z = icen_estandarizado(pd.DatetimeIndex(comp_h.time.values))

    figura_zona(config.FIGURES / "fig01_zona_estudio.png")
    figura_fichas(comp_h, ev, icen_z, config.FIGURES / "fig_fichas_8componentes.png")
    figura_eeof(eeof, ev, icen_z, config.FIGURES / "fig_eeof_sv.png")
    tabla = puntajes(comp_h, ev)
    tabla.to_csv(config.FIGURES / "puntajes_eventos.csv", float_format="%.2f")
    print("Puntaje medio por evento (duracion ICEN, sd):")
    print(tabla.round(2).to_string())
    print(f"Figuras en {config.FIGURES}")


if __name__ == "__main__":
    main()
