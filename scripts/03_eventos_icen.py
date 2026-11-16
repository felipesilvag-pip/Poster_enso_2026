"""Paso 3: eventos El Nino y La Nina por el criterio ICEN, y su figura.

ICEN (ENFEN 2012; Takahashi et al. 2014): media movil centrada de 3 meses de la
anomalia de Nino 1+2; El Nino si > +0.4 y La Nina si <= -1.0 durante >= 3 meses.
Eventos de estudio: |pico| >= ``config.EVENT_MIN_PEAK``. Se usa el periodo
``config.PERIOD_FULL``. Uso::

    python scripts/03_eventos_icen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config  # noqa: E402
from enso_chile.events import detect_events_icen, icen_index  # noqa: E402
from enso_chile.indices import load_nino_regions  # noqa: E402

COLORES = {("nino", False): "#f4c7c3", ("nina", False): "#c6dbef",
           ("nino", True): "#d62728", ("nina", True): "#1f5fbf"}


def eventos() -> tuple[pd.Series, pd.DataFrame]:
    """ICEN sobre ``PERIOD_FULL`` y la tabla de eventos (con la columna ``estudio``)."""
    nino12 = load_nino_regions(cache_dir=config.INDICES_CACHE)["nino12"]
    icen = icen_index(nino12.loc[slice(*config.PERIOD_FULL)])
    ev = detect_events_icen(icen)
    ev["estudio"] = ev["peak"].abs() >= config.EVENT_MIN_PEAK
    return icen, ev


def figura(icen: pd.Series, ev: pd.DataFrame, destino: Path) -> None:
    """Serie del ICEN con todos los eventos y los de estudio destacados."""
    medio_mes = pd.Timedelta(days=15)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for e in ev.itertuples():
        ax.axvspan(e.start - medio_mes, e.end + medio_mes, color=COLORES[(e.type, e.estudio)],
                   alpha=0.45 if e.estudio else 0.7, lw=0, zorder=0)
        if e.estudio:
            nombre = "El Niño" if e.type == "nino" else "La Niña"
            ax.annotate(f"{nombre} {e.label}\n{e.n_months} meses", xy=(e.peak_time, e.peak),
                        xytext=(0, 8 if e.peak > 0 else -10), textcoords="offset points",
                        ha="center", va="bottom" if e.peak > 0 else "top", fontsize=8,
                        color=COLORES[(e.type, True)], fontweight="bold")
    ax.plot(icen.index, icen.to_numpy(), color="k", lw=1.0, zorder=3)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.axhline(0.4, color="#d62728", lw=0.8, ls="--")
    ax.axhline(-1.0, color="#1f5fbf", lw=0.8, ls="--")
    for y in (config.EVENT_MIN_PEAK, -config.EVENT_MIN_PEAK):
        ax.axhline(y, color="0.3", lw=0.8, ls=":")
    ax.set_ylabel("ICEN (°C)")
    ax.set_xlim(icen.index[0], icen.index[-1])
    ax.set_ylim(-2.6, 4.6)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    leyenda = [
        plt.Line2D([], [], color="k", lw=1.0, label="ICEN"),
        plt.Line2D([], [], color="#d62728", lw=0.8, ls="--", label="Umbral cálido (+0.4 °C)"),
        plt.Line2D([], [], color="#1f5fbf", lw=0.8, ls="--", label="Umbral frío (−1.0 °C)"),
        plt.Line2D([], [], color="0.3", lw=0.8, ls=":", label=f"Selección (±{config.EVENT_MIN_PEAK} °C)"),
        Patch(color=COLORES[("nino", False)], label="El Niño (ICEN)"),
        Patch(color=COLORES[("nina", False)], label="La Niña (ICEN)"),
        Patch(color=COLORES[("nino", True)], alpha=0.45, label="El Niño de estudio"),
        Patch(color=COLORES[("nina", True)], alpha=0.45, label="La Niña de estudio"),
    ]
    ax.legend(handles=leyenda, ncol=4, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Detectar eventos, guardar la tabla y la figura."""
    icen, ev = eventos()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ev.to_csv(config.EVENTS_CSV, index=False)
    print(ev.loc[ev["estudio"], ["label", "type", "start", "end", "n_months", "peak", "peak_time"]]
          .to_string(index=False))
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    figura(icen, ev, config.FIGURES / "fig03_icen.png")
    print(f"Figura: {config.FIGURES / 'fig03_icen.png'}")


if __name__ == "__main__":
    main()
