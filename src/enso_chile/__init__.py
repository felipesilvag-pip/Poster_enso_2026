"""Analisis de ENSO en la costa norte de Chile a partir de GLORYS12.

Paquete con la infraestructura de datos (descarga, indices, eventos, anomalias,
cubo) y los metodos de descomposicion usados en el poster (EOF, EEOF/M-SSA y
rotaciones).

La configuracion del proyecto (rutas, cajas, profundidades, periodos) vive en
el modulo ``config`` de la raiz del repositorio, no aqui.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"
