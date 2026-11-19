"""Paso 7: pagina web del poster (``docs/index.html``, publicada con GitHub Pages).

Lee ``data/processed/componentes.nc`` (script 04) y los eventos ICEN del CPC, y
escribe una pagina autonoma con:

1. Explorador de los 12 componentes (EOF modos 1-4, varimax y bifactor): mapas de
   carga por capa, serie mensual frente al ICEN, R2 por capa y r con E y C.
2. El Nino 2026: puntaje medio por evento y trayectorias mes a mes desde el inicio
   de cada El Nino, con los meses posteriores a 2026-03 proyectados sobre los ejes
   congelados: GLORYS12 hasta su ultimo mes y, marcados como exploratorios, los
   meses del producto anfc reajustado (ver script 04).

Plotly se carga desde un CDN. Uso::

    python scripts/07_pagina_web.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import config  # noqa: E402
from enso_chile.events import detect_events_icen, icen_index  # noqa: E402
from enso_chile.indices import load_nino_regions  # noqa: E402

SALIDA = config.DOCS / "index.html"
REPO = "https://github.com/felipesilvag-pip/Poster_enso_2026"
ETIQ = ["0.5 m", "25 m", "92 m", "156 m"]
DESCRIPCION = {
    ("EOF", "Modo 1"): "Monopolo en toda la columna: la señal de tipo ENSO, pero mezclada con "
                       "variabilidad lenta y con todas las profundidades a la vez.",
    ("EOF", "Modo 2"): "Dipolo superficie/subsuperficie, casi sin relación con los índices ENSO.",
    ("EOF", "Modo 3"): "Forma un par degenerado con el modo 4 (North et al. 1982): el reparto entre "
                       "ambos es arbitrario y no se interpretan por separado.",
    ("EOF", "Modo 4"): "Forma un par degenerado con el modo 3 (North et al. 1982): el reparto entre "
                       "ambos es arbitrario y no se interpretan por separado.",
    ("Varimax", "Superficie"): "Concentrado en 0–25 m: sigue al ENSO costero.",
    ("Varimax", "Fondo"): "Concentrado a 156 m, bajo la termoclina; poco ligado al ICEN.",
    ("Varimax", "Columna"): "Repartido en toda la columna. Cambia con el número de modos rotados: "
                            "es el resto de la señal común, no un modo físico estable.",
    ("Varimax", "92 m"): "Concentrado a 92 m, en la base de la termoclina.",
    ("Bifactor", "General"): "La señal común a toda la columna: el ENSO de cuenca. Coincide con el "
                             "modo 1 del EOF, pero más pareja entre capas.",
    ("Bifactor", "Superficie"): "Lo propio de 0–25 m, una vez quitada la señal común.",
    ("Bifactor", "Fondo"): "Lo propio de 156 m, una vez quitada la señal común.",
    ("Bifactor", "92 m"): "Lo propio de 92 m, una vez quitada la señal común.",
}


def calcular() -> dict:
    """Datos de la pagina."""
    comp = xr.open_dataset(config.DATA_PROCESSED / "componentes.nc").load()
    t_all = pd.DatetimeIndex(comp.time.values)
    t_h = t_all[t_all <= pd.Timestamp(config.PERIOD_FULL[1])]
    t_n = t_all[len(t_h):]
    fuente = comp.fuente.values
    t_glorys, t_anfc = t_all[fuente != "anfc"], t_all[fuente == "anfc"]   # anfc: exploratorio

    icen = icen_index(load_nino_regions(cache_dir=config.INDICES_CACHE)["nino12"])
    icen_h = icen.reindex(t_h).to_numpy()
    mu, sd = np.nanmean(icen_h), np.nanstd(icen_h, ddof=1)
    icen_t = icen.reindex(t_all).to_numpy()

    lat, lon = comp.latitude.values, comp.longitude.values
    comps = []
    for c in range(comp.sizes["comp"]):
        met, nom = str(comp.metodo.values[c]), str(comp.nombre.values[c])
        comps.append(dict(
            met=met, nombre=nom, desc=DESCRIPCION[(met, nom)], tot=round(float(comp.tot[c]), 1),
            r2=[round(float(v), 1) for v in comp.r2.values[c]],
            r={"E": round(float(comp.rE[c]), 2), "C": round(float(comp.rC[c]), 2)},
            mapas=[[[None if not np.isfinite(v) else int(round(1000 * v)) for v in fila] for fila in capa]
                   for capa in comp.mapa.values[c]],
            serie=[round(float(v), 3) for v in comp.serie.values[c]]))

    # eventos de estudio con el ICEN completo del CPC: el de 2026 queda abierto
    ev = detect_events_icen(icen)
    ev = ev[(ev["peak"].abs() >= config.EVENT_MIN_PEAK) & (ev["start"] >= t_all[0])].reset_index(drop=True)
    fin = t_all[-1]
    corte = min(fin, icen.dropna().index[-1])        # un evento que dura hasta aqui sigue en curso
    ult = ev.iloc[-1]
    n_2026 = len(pd.date_range(ult["start"], fin if ult["end"] >= corte else min(ult["end"], fin), freq="MS"))
    series = np.array([c_["serie"] for c_ in comps])
    eventos = []
    for e in ev.itertuples():
        abierto = bool(e.end >= corte)
        hasta = fin if abierto else min(e.end, fin)
        sel = (t_all >= e.start) & (t_all <= hasta)
        sel_n = (t_all >= e.start) & (t_all < e.start + pd.DateOffset(months=n_2026))
        corto = e.n_months < n_2026 and not abierto      # las Ninas (4 m): ventana no comparable
        eventos.append(dict(
            label=e.label, inicio=f"{e.start:%Y-%m}", fin=f"{hasta:%Y-%m}", abierto=abierto,
            pico=round(float(e.peak), 2), pico_mes=f"{e.peak_time:%Y-%m}", tipo=e.type,
            pico_ventana=round(float(icen.loc[e.start:hasta].max()), 2),
            completo=[round(float(v), 2) for v in series[:, sel].mean(1)], duracion=int(e.n_months),
            primeros=[None if corto else round(float(v), 2) for v in series[:, sel_n].mean(1)]))

    return dict(
        t=[f"{x:%Y-%m}" for x in t_all], t_nuevo=f"{t_n[0]:%Y-%m}", t_fin=f"{fin:%Y-%m}",
        t_hist_fin=f"{t_h[-1]:%Y-%m}",
        icen=[None if not np.isfinite(v) else round(float((v - mu) / sd), 3) for v in icen_t],
        lat=[round(float(v), 4) for v in lat], lon=[round(float(v), 4) for v in lon],
        etiq=ETIQ, comps=comps, eventos=eventos, n_2026=n_2026, fuente_icen="CPC/NOAA",
        icen_ult=f"{icen.dropna().index[-1]:%Y-%m}", t_glorys_fin=f"{t_glorys[-1]:%Y-%m}",
        t_anfc=f"{t_anfc[0]:%Y-%m}" if len(t_anfc) else None)


HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ENSO en el norte de Chile</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f2f4f5; --surface:#ffffff; --ink:#15212b; --muted:#56656f; --rule:#d3dadf; --grid:#e3e8eb;
  --accent:#0a6a86; --accent-soft:#dcecf1; --warm:#b2182b; --cold:#2166ac; --new:#e08214;
  --f-disp:"Source Serif 4",Georgia,serif; --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f161c; --surface:#16212a; --ink:#e4ebef; --muted:#9aa9b3; --rule:#2c3a45; --grid:#22303a;
  --accent:#5fb7d1; --accent-soft:#183544; --warm:#e0646f; --cold:#6aa3db; --new:#f0a54a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f161c; --surface:#16212a; --ink:#e4ebef; --muted:#9aa9b3; --rule:#2c3a45; --grid:#22303a;
  --accent:#5fb7d1; --accent-soft:#183544; --warm:#e0646f; --cold:#6aa3db; --new:#f0a54a; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.6 var(--f-body);margin:0}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
header{border-bottom:1px solid var(--rule);padding-bottom:18px}
.eyebrow{font:600 11.5px/1 var(--f-body);letter-spacing:.09em;text-transform:uppercase;color:var(--accent)}
h1{font:700 clamp(26px,4vw,38px)/1.15 var(--f-disp);margin:10px 0;text-wrap:balance}
h2{font:700 24px/1.25 var(--f-disp);margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--ink);text-wrap:balance}
h4{font:600 12px/1.3 var(--f-body);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
p,li{max-width:74ch} p{margin:8px 0}
.lede{font-size:16.5px;color:var(--muted)}
.meta{font:12.5px/1.5 var(--f-mono);color:var(--muted)}
a{color:var(--accent)}
.widget{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px 10px;margin:16px 0}
.ctrls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-bottom:8px}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.seg-l{font:600 11px/1 var(--f-body);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:6px}
.seg button{font:500 13px/1 var(--f-body);background:transparent;color:var(--ink);border:0;border-right:1px solid var(--rule);padding:8px 12px;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--surface)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}
.two>div{min-width:0}
.maps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
.plot{width:100%;min-height:220px}
.map{height:330px}
.cap{font-size:12.5px;color:var(--muted);margin:4px 0 2px}
.desc{background:var(--accent-soft);border-radius:4px;padding:10px 14px;margin:10px 0;font-size:14.5px}
.kpi{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:14px;margin:6px 0}
.kpi b{font-family:var(--f-mono);font-weight:500}
.note{border-left:3px solid var(--rule);padding:4px 0 4px 14px;color:var(--muted);font-size:14px;margin:12px 0}
.tbl{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font:600 11.5px/1.3 var(--f-body);letter-spacing:.04em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule)}
tr.nuevo td{font-weight:600;color:var(--new)}
dl.idx{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px;margin:8px 0}
dl.idx dt{font-family:var(--f-mono);color:var(--accent)} dl.idx dd{margin:0}
footer{margin-top:48px;border-top:1px solid var(--rule);padding-top:14px;font-size:13px;color:var(--muted)}
@media (max-width:760px){.two{grid-template-columns:minmax(0,1fr)}.maps{grid-template-columns:repeat(2,minmax(0,1fr))}.map{height:280px}}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@2.35.2/plotly-cartesian.min.js"></script>
</head>
<body>
<main>
<div id="banner"></div>
<header>
  <div class="eyebrow">Póster · Ingeniería Estadística, USACH</div>
  <h1>El Niño en la columna de agua del norte de Chile: EOF combinado y sus rotaciones</h1>
  <p class="lede">Anomalías de temperatura de GLORYS12 en Norte Grande (18–27 °S, costa a 74 °W) en cuatro
  capas: 0.5, 25, 92 y 156 m. Aquí puedes revisar los 12 componentes (también los que no caben en el póster)
  y seguir El Niño 2026 hasta el último mes disponible de GLORYS12.</p>
  <p class="meta">Felipe Silva Guevara · datos 1993-01 a <span id="m-fin"></span></p>
</header>

<h2>1 · Explorador de componentes</h2>
<p>El <b>EOF</b> resume la variabilidad en pocos modos, pero el primero se lleva todo lo que varía junto y
mezcla profundidades. <b>Varimax</b> rota esos modos para que cada componente se concentre en una capa. El
<b>bifactor</b> reserva un componente <b>general</b>, común a toda la columna, y deja el resto como
<b>grupos</b> propios de cada profundidad.</p>
<p>Elige un método y un componente. Los <b>mapas</b> son las cargas en cada capa: cuántos °C cambia cada punto
cuando la serie sube una desviación estándar (rojo, calentamiento). La <b>serie</b> es el puntaje
estandarizado, con el ICEN de fondo. Los últimos meses, en naranjo, son la <b>proyección</b> de datos nuevos (sección 2).
Las barras dan el <b>R²</b>: el % de la varianza de cada capa que explica el componente.</p>
<dl class="idx">
  <dt>ICEN</dt><dd>Niño 1+2 con media móvil de 3 meses (ENFEN 2012): el ENSO costero.</dd>
  <dt>E</dt><dd>Niño 1+2 − 0.5 · Niño 4 (Takahashi et al. 2011): El Niño del Pacífico oriental.</dd>
  <dt>C</dt><dd>1.7 · Niño 4 − 0.1 · Niño 1+2 (Takahashi et al. 2011): El Niño del Pacífico central. E y C son casi independientes entre sí (r = 0.07).</dd>
</dl>
<div class="widget">
  <div class="ctrls">
    <span><span class="seg-l">Método</span><span class="seg" id="s-met"></span></span>
    <span><span class="seg-l">Componente</span><span class="seg" id="s-comp"></span></span>
  </div>
  <div class="desc" id="t-desc"></div>
  <div class="kpi" id="t-kpi"></div>
  <div class="maps"><div id="p-m0" class="plot map"></div><div id="p-m1" class="plot map"></div><div id="p-m2" class="plot map"></div><div id="p-m3" class="plot map"></div></div>
  <div class="two" style="grid-template-columns:minmax(0,3fr) minmax(0,1fr)">
    <div><div id="p-serie" class="plot" style="height:270px"></div></div>
    <div><div id="p-r2" class="plot" style="height:270px"></div></div>
  </div>
  <p class="cap">Varimax y bifactor son rotaciones ortogonales de los 4 primeros modos: sus % del total se suman
  igual que en el EOF (76 %). El signo de cada componente se fija para que la carga media de su capa más fuerte
  sea positiva.</p>
</div>

<h2>2 · El Niño 2026, hasta <span id="m-fin2"></span></h2>
<p>Todo se estimó con 1993-01 a <span id="m-hist"></span> y queda <b>congelado</b>: el ajuste de anomalías de cada
píxel (constante, tendencia y 12 meses), los EOF por capa, la normalización y las rotaciones. Los meses nuevos de
GLORYS12 solo se <b>proyectan</b> sobre esos componentes.</p>
<div class="desc" id="t-cortes"><b>Dos fechas de corte distintas.</b> Los puntajes de los componentes (tabla y trayectorias)
llegan solo hasta <b><span id="m-fin3"></span></b>, el último mes publicado de GLORYS12: el reanálisis sale con unos
meses de retraso. El ICEN, en cambio, llega hasta <b><span id="m-icen-ult"></span></b>. Por eso el pico del ICEN de
2026 (<span id="m-pico"></span>) es posterior a los datos oceánicos y no entra en las medias. Dentro de la ventana con
datos, el ICEN máximo de 2026 es <span id="m-pico-v"></span>.</div>
<div class="widget">
  <div class="ctrls">
    <span><span class="seg-l">Método</span><span class="seg" id="s-met2"></span></span>
    <span><span class="seg-l">Ventana</span><span class="seg" id="s-ven"></span></span>
  </div>
  <div class="tbl" id="t-ev"></div>
  <p class="cap" id="c-ev"></p>
</div>
<h4>Trayectorias de los El Niño, mes a mes desde su inicio</h4>
<p>Cada curva es el puntaje mensual de un componente desde el primer mes del evento según el ICEN. Así se compara
etapa con etapa y se ve cómo crece 2026, cosa que una media de varios meses esconde.</p>
<div class="widget">
  <div class="ctrls">
    <span><span class="seg-l">Método</span><span class="seg" id="s-met3"></span></span>
    <span><span class="seg-l">Componente</span><span class="seg" id="s-comp3"></span></span>
  </div>
  <div id="p-tray" class="plot" style="height:340px"></div>
  <p class="cap" id="c-tray"></p>
</div>
<div class="note">Exploratorio: las proyecciones no tienen intervalos de confianza, y el evento de 2026 sigue en
curso al cierre de los datos. Comparar 2026 con los primeros meses de los otros eventos (la misma etapa) es más
justo que compararlo con eventos completos.</div>

<footer>
  <p>Código: <a href="__REPO__" id="a-repo">repositorio en GitHub</a>. Datos: Copernicus Marine Service, GLORYS12
  (<code>cmems_mod_glo_phy_my_0.083deg_P1M-m</code>); índices Niño del CPC/NOAA (ICEN mostrado: <span id="m-icen"></span>).
  Rotaciones con píxeles como filas sobre el EOF combinado (K = 6 PC por capa, cada capa normalizada).
  Varimax con normalización de Kaiser; bifactor bi-quartimin ortogonal (Jennrich &amp; Bentler 2011).</p>
</footer>
</main>

<script>
const D = __DATOS__;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const hasP = typeof Plotly !== "undefined";
const CFG = {displayModeBar:false, responsive:true, scrollZoom:false};
const renders = [];
function reg(f){ renders.push(f); try{ f(); }catch(e){ console.error(e); } }
/* pantallas angostas (celular): graficos mas altos, menos marcas y lineas mas gruesas */
const movil = () => window.matchMedia("(max-width:760px)").matches;
/* sin arrastre ni zoom: en pantallas tactiles el dedo hace scroll en vez de mover el grafico */
function lay(l){ return Object.assign({paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
  font:{family:css("--f-body"), color:css("--ink"), size:12}, margin:{l:44,r:10,t:10,b:36}, showlegend:false,
  dragmode:false}, l,
  {xaxis:Object.assign({gridcolor:css("--grid"), zerolinecolor:css("--rule"), fixedrange:true}, l.xaxis||{}),
   yaxis:Object.assign({gridcolor:css("--grid"), zerolinecolor:css("--rule"), fixedrange:true}, l.yaxis||{})}); }
function plot(id, data, layout){ if(!hasP){ $(id).textContent = "No se pudo cargar Plotly desde el CDN."; return; } Plotly.react(id, data, lay(layout), CFG); }
function seg(id, labels, ini, cb){
  const el = $(id); el.innerHTML = "";
  labels.forEach((t,i)=>{ const b = document.createElement("button"); b.type = "button"; b.textContent = t;
    b.setAttribute("aria-pressed", i===ini); b.onclick = ()=>{ [...el.children].forEach(x=>x.setAttribute("aria-pressed", x===b)); cb(i); }; el.appendChild(b); });
}
const f2 = v => (v===null||v===undefined) ? "–" : (v>0?"+":"")+v.toFixed(2);
["m-fin","m-fin2"].forEach(i=>$(i).textContent = D.t_fin); $("m-fin3").textContent = D.t_glorys_fin;
if (D.t_anfc){
  $("banner").innerHTML = "<div class='desc' style='border:2px solid var(--new);margin-bottom:16px'><b>Versión exploratoria.</b> GLORYS12 llega hasta "
    + D.t_glorys_fin+". Los meses "+D.t_anfc+" a "+D.t_fin+" vienen del análisis operativo de Copernicus (análisis y pronóstico, «anfc»),"
    + " reajustado por su sesgo medio frente a GLORYS12 para aproximarlo. Se reemplazarán cuando GLORYS12 los publique.</div>";
  setTimeout(()=>{ $("t-cortes").innerHTML = "<b>Fechas de corte.</b> GLORYS12 llega hasta <b>"+D.t_glorys_fin+"</b>; de "+D.t_anfc+" a <b>"+D.t_fin
    + "</b> los puntajes vienen de anfc corregido (sombreado más oscuro, línea punteada). El ICEN llega hasta <b>"+D.icen_ult+"</b>.";}, 0);
}
$("m-hist").textContent = D.t_hist_fin; $("m-icen").textContent = D.fuente_icen; $("m-icen-ult").textContent = D.icen_ult;
(function(){ const e = D.eventos.find(x=>x.abierto);
  $("m-pico").textContent = e ? f2(e.pico)+" °C en "+e.pico_mes : "–";
  $("m-pico-v").textContent = e ? f2(e.pico_ventana)+" °C" : "–"; })();

/* ---------- 2. explorador ---------- */
const METS = ["EOF","Varimax","Bifactor"];
const porMet = m => D.comps.map((c,i)=>Object.assign({i}, c)).filter(c=>c.met===m).sort((a,b)=>b.tot-a.tot);  // por % del total
let met = 0, cmp = 0;
function sombras(){
  const sh = D.eventos.map(e=>({type:"rect", xref:"x", yref:"paper", x0:e.inicio+"-01", x1:e.fin+"-28", y0:0, y1:1,
    fillcolor:css("--grid"), line:{width:0}, layer:"below"}));
  sh.push({type:"rect", xref:"x", yref:"paper", x0:D.t_nuevo+"-01", x1:D.t_fin+"-28", y0:0, y1:1, fillcolor:css("--new"), opacity:0.15, line:{width:0}, layer:"below"});
  if (D.t_anfc) sh.push({type:"rect", xref:"x", yref:"paper", x0:D.t_anfc+"-01", x1:D.t_fin+"-28", y0:0, y1:1, fillcolor:css("--new"), opacity:0.25, line:{width:0}, layer:"below"});
  return sh;
}
function dibujaComp(){
  const c = porMet(METS[met])[cmp], zmax = Math.max(...c.mapas.flat(2).filter(v=>v!==null).map(Math.abs))/1000;
  $("t-desc").textContent = METS[met]+" · "+c.nombre+". "+c.desc;
  $("t-kpi").innerHTML = "<span>% del total: <b>"+c.tot.toFixed(1)+"</b></span><span>r E <b>"+f2(c.r["E"])
    +"</b></span><span>r C <b>"+f2(c.r["C"])+"</b></span>";
  c.mapas.forEach((m,z)=>{
    plot("p-m"+z, [{type:"heatmap", x:D.lon, y:D.lat, z:m.map(f=>f.map(v=>v===null?null:v/1000)), zmin:-zmax, zmax:zmax,
      colorscale:[[0,"#2166ac"],[0.25,"#92c5de"],[0.5,"#f7f7f7"],[0.75,"#f4a582"],[1,"#b2182b"]], showscale:z===3,
      colorbar:{thickness:10, len:0.9, title:{text:"°C/sd", side:"right", font:{size:10}}, tickfont:{size:10}},
      hovertemplate:"%{y:.2f}°, %{x:.2f}°: %{z:.2f} °C/sd<extra></extra>"}],
      {margin:{l:z===0?40:20,r:z===3?44:6,t:26,b:30}, title:{text:D.etiq[z]+" · R² "+c.r2[z].toFixed(0)+" %", font:{size:12}, x:0.02, y:0.98},
       xaxis:{tickfont:{size:10}, dtick:2}, yaxis:{tickfont:{size:10}, scaleanchor:"x", scaleratio:1/Math.cos(22.5*Math.PI/180), showticklabels:z===0}});
  });
  $("p-serie").style.height = $("p-r2").style.height = movil() ? "300px" : "270px";
  plot("p-serie", [
    {x:D.t, y:D.icen, mode:"lines", line:{color:css("--muted"), width:1}, name:"ICEN", hovertemplate:"%{y:.2f}"},
    {x:D.t, y:c.serie, mode:"lines", line:{color:css("--accent"), width:movil()?2:1.6}, name:c.nombre, hovertemplate:"%{y:.2f} sd"}],
    {showlegend:true, legend:{orientation:"h", y:1.12, x:0}, margin:{l:40,r:10,t:30,b:30}, shapes:sombras(),
     hovermode:"x unified", yaxis:{title:{text:"sd", font:{size:11}}},
     xaxis:{type:"date", hoverformat:"%Y-%m", nticks:movil()?5:10}});
  plot("p-r2", [{type:"bar", orientation:"h", y:D.etiq, x:c.r2, marker:{color:css("--accent")}, text:c.r2.map(v=>v.toFixed(0)+" %"),
    textposition:"outside", cliponaxis:false, hovertemplate:"%{y}: %{x:.1f} %<extra></extra>"}],
    {margin:{l:50,r:30,t:30,b:36}, title:{text:"R² por capa", font:{size:12}, x:0.02}, xaxis:{range:[0, Math.max(60, ...c.r2)*1.15]}, yaxis:{autorange:"reversed"}});
}
function botonesComp(){ seg("s-comp", porMet(METS[met]).map(c=>c.nombre), cmp, i=>{ cmp = i; dibujaComp(); }); }
seg("s-met", METS, 0, i=>{ met = i; cmp = 0; botonesComp(); dibujaComp(); });
botonesComp(); reg(dibujaComp);

/* ---------- 3. eventos ---------- */
let met2 = 1, ven = 0;
function tablaEv(){
  const cs = porMet(METS[met2]), campo = ven ? "primeros" : "completo";
  let h = "<table><thead><tr><th>Evento</th><th>Meses promediados</th><th>Pico ICEN (°C)</th><th>Mes del pico</th>"+cs.map(c=>"<th>"+c.nombre+"</th>").join("")+"</tr></thead><tbody>";
  D.eventos.forEach(e=>{
    const meses = ven ? e.inicio+" + "+D.n_2026+" m" : e.inicio+" a "+e.fin;
    h += "<tr"+(e.abierto?" class='nuevo'":"")+"><td>"+(e.tipo==="nina"?"La Niña ":"El Niño ")+e.label+(e.abierto?" (en curso)":"")+"</td><td>"+meses+"</td><td>"+f2(e.pico)+"</td><td>"+e.pico_mes+(e.abierto&&e.pico_mes>D.t_fin?"*":"")+"</td>"
      + cs.map(c=>"<td>"+f2(e[campo][c.i])+"</td>").join("")+"</tr>";
  });
  $("t-ev").innerHTML = h+"</tbody></table>";
  $("c-ev").textContent = "Puntaje medio de cada componente (en desviaciones estándar de 1993–"+D.t_hist_fin.slice(0,4)+")"
    + (ven ? " en los primeros "+D.n_2026+" meses de cada evento, la misma etapa que 2026 al cierre de los datos." : " durante cada evento según el ICEN; el de 2026 llega hasta "+D.t_fin+".")
    + " Pico ICEN: el máximo hasta el último mes con ICEN ("+D.icen_ult+")."
    + " * Pico posterior al último mes de GLORYS12 ("+D.t_fin+"): no entra en las medias de los componentes."
    + (ven ? " «–»: eventos que duran menos de "+D.n_2026+" meses (las Niñas, 4 m), cuya ventana incluiría meses posteriores al evento." : "");
}

/* ---------- trayectorias ---------- */
let met3 = 1, cmp3 = 0;
const NMAX = 24;
function dibujaTray(){
  const c = porMet(METS[met3])[cmp3], tr = [], etiquetas = [], m = movil(), g = m ? 0.6 : 0;
  const grises = {"1997-98":"#7b3294", "2015-16":"#1b9e77", "2023-24":"#2166ac"};
  $("p-tray").style.height = m ? "420px" : "340px";
  D.eventos.filter(e=>e.tipo==="nino").forEach(e=>{
    const i0 = D.t.indexOf(e.inicio), x = [], y = [];
    for (let k=0; k<NMAX && i0+k<D.t.length; k++){ x.push(k+1); y.push(c.serie[i0+k]); }
    const dur = e.abierto ? null : e.duracion, color = e.abierto ? css("--new") : (grises[e.label]||css("--muted"));
    const xf = x.slice(), yf = y.slice();                 // extremo de la curva completa, para la etiqueta
    if (e.abierto && D.t_anfc){                      // tramo anfc: punteado, desde el ultimo mes de GLORYS
      const kg = x.filter(k=>D.t[i0+k-1] <= D.t_glorys_fin).length;
      tr.push({x:x.slice(kg-1), y:y.slice(kg-1), mode:"lines+markers", name:e.label+" anfc",
        line:{color:color, width:3+g, dash:"dot"}, marker:{size:m?8:6}, hovertemplate:"%{y:.2f} sd"});
      x.length = kg; y.length = kg;
    }
    tr.push({x, y, mode:"lines", name:e.label, line:{color:color, width:(e.abierto?3.5:1.8)+g}, hovertemplate:"%{y:.2f} sd"});
    if (dur && dur<=NMAX) tr.push({x:[dur], y:[c.serie[i0+dur-1]], mode:"markers", showlegend:false, hoverinfo:"skip",
      marker:{size:m?9:7, symbol:"x", color:color}});
    etiquetas.push({x:xf[xf.length-1], y:yf[yf.length-1], text:"<b>"+e.label+"</b>", showarrow:false, xanchor:"left",
      xshift:6, font:{color:color, size:m?12:11}});
  });
  /* etiquetas al final de cada curva en vez de leyenda; separar las que quedan muy juntas */
  etiquetas.sort((a,b)=>a.y-b.y);
  for (let i=1; i<etiquetas.length; i++){
    const dy = etiquetas[i].y - etiquetas[i-1].y;
    if (Math.abs(etiquetas[i].x - etiquetas[i-1].x) < 3 && dy < 0.45) etiquetas[i].y = etiquetas[i-1].y + 0.45;
  }
  plot("p-tray", tr, {margin:{l:44,r:m?62:66,t:12,b:40}, hovermode:false, annotations:etiquetas,
    xaxis:{title:{text:"meses desde el inicio del evento (ICEN)", font:{size:11}}, dtick:m?6:3, range:[0.5, NMAX+0.5],
           zeroline:false},
    yaxis:{title:{text:"puntaje (sd)", font:{size:11}}},
    shapes:[{type:"line", xref:"x", yref:"y", x0:0.5, x1:NMAX+0.5, y0:0, y1:0, line:{color:css("--rule"), width:1}}]});
  $("c-tray").textContent = METS[met3]+" · "+c.nombre+". Las ✕ marcan el fin de cada evento según el ICEN. 2026 llega hasta "
    + D.t_glorys_fin+" con GLORYS12" + (D.t_anfc ? " y, punteado, hasta "+D.t_fin+" con anfc (exploratorio)" : "")
    + "; el ICEN sigue hasta "+D.icen_ult+"."
    +" Puntajes mensuales, sin suavizar.";
}
function botonesComp3(){ seg("s-comp3", porMet(METS[met3]).map(c=>c.nombre), cmp3, i=>{ cmp3 = i; dibujaTray(); }); }
seg("s-met3", METS, met3, i=>{ met3 = i; cmp3 = 0; botonesComp3(); dibujaTray(); });
botonesComp3(); reg(dibujaTray);
seg("s-met2", METS, met2, i=>{ met2 = i; tablaEv(); });
seg("s-ven", ["Evento completo","Primeros "+D.n_2026+" meses"], 0, i=>{ ven = i; tablaEv(); });
tablaEv();

if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>renders.forEach(f=>f()));
if (window.matchMedia) window.matchMedia("(max-width:760px)").addEventListener("change", ()=>renders.forEach(f=>f()));
</script>
</body>
</html>
"""


def main() -> None:
    """Calcular los datos y escribir la pagina."""
    datos = calcular()
    txt = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(HTML.replace("__DATOS__", txt).replace("__REPO__", REPO), encoding="utf-8")
    print(f"Pagina en {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
