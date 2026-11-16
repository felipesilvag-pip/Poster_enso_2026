"""EOF clasico por descomposicion en valores singulares.

Linea base del proyecto: la descomposicion contra la que se comparan EEOF,
CEOF y las versiones rotadas. Resuelve

.. math::

    X = U \\Sigma V^{\\top},

donde ``X`` es la matriz de datos de forma ``(tiempo, espacio)``, ponderada por
area y centrada en el tiempo. Los patrones espaciales (EOF) son las filas de
:math:`V^{\\top}`, las componentes principales (PC) son las columnas de
:math:`U\\Sigma`, y los autovalores de la matriz de covarianza son
:math:`\\lambda_i = \\sigma_i^2 / (T - 1)`.

Convenciones fijadas
--------------------
**Orientacion.** Aqui la matriz es ``(tiempo, espacio)``: una fila por mes, una
columna por pixel. ``Entendiendo_EOF.py`` usa la transpuesta, ``(espacio,
tiempo)``, y por eso ahi los EOF salen en ``U`` y las PC en
:math:`\\Sigma V^{\\top}`. Son la misma descomposicion leida al revés; la
eleccion de aqui coincide con ``eofs`` y con ``scikit-learn``, y evita
transponer al pasar de ``cube.flatten_space``.

**Centrado.** Siempre en el tiempo, por pixel. La matriz centrada tiene rango a
lo mas ``min(T - 1, N)``: el ultimo autovalor del problema de covarianza es
ruido numerico y se descarta, igual que el ``k = T - 1`` del script original.

**Ponderacion.** ``sqrt(cos(lat))`` aplicada a los datos antes del SVD. El
bloque portado de ``Entendiendo_EOF.py`` no la aplicaba (``COAST.py`` si, via el
paquete ``eofs``); aqui es obligatoria y consistente en todo el proyecto, para
que las varianzas explicadas sean comparables entre cajas. Al mapear un patron
espacial se **divide** por el peso para devolverlo a unidades fisicas, lo que
hace :func:`EOFResult.eofs_as_maps`.

**Signo.** El signo de un par (EOF, PC) es arbitrario en cualquier SVD. Se fija
por convencion para que los resultados sean reproducibles: se elige el signo que
hace positiva la componente de mayor magnitud de cada EOF.

Notes
-----
LEGACY: ``Entendiendo_EOF.py`` l.1-86 (SVD, PCA por covarianza temporal con el
metodo de snapshots, y las cuatro pruebas de equivalencia, hoy en
``tests/test_eof.py``); ``COAST.py`` l.1421-1560 (EOF con el paquete ``eofs`` y
``sqrt(coslat)``).

Este modulo **no** implementa la regla de North et al. (1982) ni la deteccion de
pares degenerados: eso vive en ``degeneracy.py``. Aqui solo se entregan los
autovalores y el numero de grados de libertad necesario para aplicarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr

from .cube import apply_area_weights, area_weights, flatten_space, unflatten_space

__all__ = [
    "Scaling",
    "EOFResult",
    "eof_decomposition",
    "eof_from_dataarray",
    "align_modes",
    "CombinedEOFResult",
    "combined_blocks",
    "combined_eof",
    "combined_eof_from_results",
    "pixel_loadings",
    "scores_from_pattern",
]

#: Escalado del par (EOF, PC).
#:
#: ``'orthonormal'``
#:     EOF ortonormales (``E Eᵀ = I``) y PC con las unidades fisicas de la
#:     variable. Es la convencion del script original y la de ``eofs``.
#: ``'std_pc'``
#:     PC de varianza unitaria y EOF en unidades fisicas. Cada EOF es entonces
#:     el coeficiente de regresion del campo sobre su PC, lo que facilita leer
#:     un mapa en grados Celsius por desviacion estandar de la PC.
Scaling = Literal["orthonormal", "std_pc"]


@dataclass
class EOFResult:
    """Resultado de una descomposicion EOF.

    Attributes
    ----------
    eofs : numpy.ndarray
        Patrones espaciales, forma ``(n_modes, n_space)``, en el espacio
        **ponderado**. Usar :meth:`eofs_as_maps` para obtenerlos en unidades
        fisicas y con su forma espacial.
    pcs : numpy.ndarray
        Componentes principales, forma ``(n_time, n_modes)``.
    eigenvalues : numpy.ndarray
        Autovalores de la matriz de covarianza, ``sigma^2 / (T - 1)``.
    variance_fraction : numpy.ndarray
        Fraccion de la varianza total explicada por cada modo. Suma 1 sobre
        todos los modos del rango completo.
    singular_values : numpy.ndarray
        Valores singulares de la matriz centrada y ponderada.
    mean : numpy.ndarray
        Media temporal por pixel que se removio, en el espacio ponderado.
    rank : int
        Rango de la matriz centrada, ``min(n_time - 1, n_space)``.
    n_time, n_space : int
        Dimensiones de la matriz de datos.
    scaling : str
        Escalado aplicado; ver :data:`Scaling`.
    weights : xarray.DataArray or None
        Pesos de area usados, si la descomposicion vino de un ``DataArray``.
    index : xarray.DataArray or None
        Objeto apilado de ``cube.flatten_space``, necesario para reconstruir
        mapas.
    """

    eofs: np.ndarray
    pcs: np.ndarray
    eigenvalues: np.ndarray
    variance_fraction: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    rank: int
    n_time: int
    n_space: int
    scaling: str
    weights: xr.DataArray | None = field(default=None, repr=False)
    index: xr.DataArray | None = field(default=None, repr=False)

    @property
    def n_modes(self) -> int:
        """Numero de modos retenidos."""
        return self.eofs.shape[0]

    def cumulative_variance(self) -> np.ndarray:
        """Varianza explicada acumulada por los modos retenidos.

        Returns
        -------
        numpy.ndarray
            Suma acumulada de ``variance_fraction``.
        """
        return np.cumsum(self.variance_fraction)

    def reconstruct(self, n_modes: int | None = None) -> np.ndarray:
        """Reconstruir la matriz de datos con los primeros modos.

        Parameters
        ----------
        n_modes : int or None, optional
            Numero de modos a usar. ``None`` usa todos los retenidos.

        Returns
        -------
        numpy.ndarray
            Matriz ``(n_time, n_space)`` en el espacio ponderado, con la media
            temporal repuesta.

        Notes
        -----
        Con todos los modos del rango se recupera la matriz original salvo
        error de redondeo. Truncando en ``k`` modos se obtiene la mejor
        aproximacion de rango ``k`` en norma de Frobenius (Eckart-Young), que es
        la propiedad que justifica interpretar los primeros modos.
        """
        k = self.n_modes if n_modes is None else n_modes
        if not 1 <= k <= self.n_modes:
            raise ValueError(f"n_modes debe estar en [1, {self.n_modes}], no {k}.")

        if self.scaling == "orthonormal":
            approx = self.pcs[:, :k] @ self.eofs[:k]
        else:
            approx = (self.pcs[:, :k] * np.sqrt(self.n_time - 1)) @ (
                self.eofs[:k] / np.sqrt(self.n_time - 1)
            )
        return approx + self.mean

    def eofs_as_maps(
        self, n_modes: int | None = None, unweight: bool = True
    ) -> xr.DataArray:
        """Devolver los patrones espaciales como mapas con coordenadas.

        Parameters
        ----------
        n_modes : int or None, optional
            Numero de modos a mapear.
        unweight : bool, optional
            Dividir por ``sqrt(cos(lat))`` para devolver el patron a unidades
            fisicas. Es lo correcto para graficar e interpretar; dejarlo en
            ``False`` solo sirve para verificar la ortonormalidad, que se
            cumple en el espacio ponderado.

        Returns
        -------
        xarray.DataArray
            Patrones con dimension ``mode`` y las dimensiones espaciales
            originales. Los pixeles de tierra son NaN.

        Raises
        ------
        ValueError
            Si la descomposicion no se hizo desde un ``DataArray`` y por tanto
            no hay coordenadas espaciales que reponer.
        """
        if self.index is None:
            raise ValueError(
                "Esta descomposicion no guarda coordenadas espaciales; usar "
                "eof_from_dataarray para poder mapear los patrones."
            )
        k = self.n_modes if n_modes is None else n_modes

        maps = [
            unflatten_space(self.eofs[i], self.index, name=f"eof{i + 1}")
            for i in range(k)
        ]
        out = xr.concat(maps, dim="mode")
        out = out.assign_coords(mode=np.arange(1, k + 1))

        if unweight and self.weights is not None:
            out = out / self.weights
        out.attrs = {
            "long_name": "patron espacial EOF",
            "scaling": self.scaling,
            "unweighted": str(bool(unweight)),
        }
        return out

    def pcs_as_dataarray(self, n_modes: int | None = None) -> xr.DataArray:
        """Devolver las PC como ``DataArray`` con el eje temporal original.

        Parameters
        ----------
        n_modes : int or None, optional
            Numero de modos a incluir.

        Returns
        -------
        xarray.DataArray
            PC con dimensiones ``(time, mode)``.
        """
        if self.index is None:
            raise ValueError(
                "Esta descomposicion no guarda el eje temporal; usar "
                "eof_from_dataarray."
            )
        k = self.n_modes if n_modes is None else n_modes
        time_dim = self.index.dims[0]
        return xr.DataArray(
            self.pcs[:, :k],
            coords={time_dim: self.index[time_dim], "mode": np.arange(1, k + 1)},
            dims=[time_dim, "mode"],
            name="pc",
            attrs={"scaling": self.scaling},
        )


def _fix_signs(eofs: np.ndarray, pcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fijar el signo de cada par (EOF, PC) de forma reproducible.

    El signo de un par es arbitrario: ``(-E, -P)`` representa exactamente el
    mismo modo que ``(E, P)``. Se elige el que hace positiva la componente de
    mayor magnitud del patron espacial, criterio que no depende del orden de los
    pixeles ni de la libreria de algebra lineal.
    """
    dominante = np.argmax(np.abs(eofs), axis=1)
    signo = np.sign(eofs[np.arange(eofs.shape[0]), dominante])
    signo[signo == 0] = 1.0
    return eofs * signo[:, None], pcs * signo[None, :]


def eof_decomposition(
    matrix: np.ndarray,
    n_modes: int | None = None,
    scaling: Scaling = "orthonormal",
    center: bool = True,
) -> EOFResult:
    """Descomponer una matriz ``(tiempo, espacio)`` en EOF y PC por SVD.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matriz de datos ``(n_time, n_space)``, ya ponderada por area. No puede
        contener NaN.
    n_modes : int or None, optional
        Numero de modos a retener. ``None`` retiene el rango completo.
    scaling : {'orthonormal', 'std_pc'}, optional
        Escalado del par (EOF, PC); ver :data:`Scaling`.
    center : bool, optional
        Remover la media temporal por pixel. Dejarlo en ``False`` solo tiene
        sentido si la matriz ya viene centrada.

    Returns
    -------
    EOFResult
        Resultado de la descomposicion.

    Raises
    ------
    ValueError
        Si la matriz no es bidimensional, contiene NaN, o ``n_modes`` excede el
        rango disponible.

    Notes
    -----
    Las fracciones de varianza se calculan sobre la **suma completa** de
    autovalores del rango, no sobre los modos retenidos: truncar a tres modos no
    hace que esos tres expliquen el 100 por ciento.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Se espera una matriz 2-D, no de forma {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "La matriz tiene NaN o infinitos; descartar los pixeles de tierra "
            "con cube.flatten_space antes de descomponer."
        )

    n_time, n_space = matrix.shape
    if n_time < 2:
        raise ValueError("Se necesitan al menos dos instantes de tiempo.")

    mean = matrix.mean(axis=0, keepdims=True) if center else np.zeros((1, n_space))
    centered = matrix - mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)

    rank = min(n_time - 1, n_space) if center else min(n_time, n_space)
    s, u, vt = s[:rank], u[:, :rank], vt[:rank]

    eigenvalues = s**2 / (n_time - 1)
    variance_fraction = eigenvalues / eigenvalues.sum()

    if scaling == "orthonormal":
        eofs, pcs = vt, u * s
    elif scaling == "std_pc":
        # PC de varianza unitaria; el EOF absorbe la amplitud y queda en las
        # unidades fisicas de la variable.
        eofs, pcs = vt * s[:, None] / np.sqrt(n_time - 1), u * np.sqrt(n_time - 1)
    else:
        raise ValueError(
            f"scaling debe ser 'orthonormal' o 'std_pc', no {scaling!r}."
        )

    eofs, pcs = _fix_signs(eofs, pcs)

    k = rank if n_modes is None else n_modes
    if not 1 <= k <= rank:
        raise ValueError(f"n_modes debe estar en [1, {rank}], no {k}.")

    return EOFResult(
        eofs=eofs[:k],
        pcs=pcs[:, :k],
        eigenvalues=eigenvalues[:k],
        variance_fraction=variance_fraction[:k],
        singular_values=s[:k],
        mean=mean,
        rank=rank,
        n_time=n_time,
        n_space=n_space,
        scaling=scaling,
    )


def align_modes(reference: EOFResult, other: EOFResult) -> EOFResult:
    """Alinear el signo de los modos de una descomposicion con otra.

    La convencion de signo de :func:`eof_decomposition` es reproducible dentro
    de una descomposicion, pero **no** entre descomposiciones distintas: el
    pixel de mayor magnitud puede cambiar al variar la caja, la profundidad o
    el tratamiento de la tendencia, y entonces dos versiones del mismo modo
    aparecen con colores opuestos en un mapa. Esta funcion voltea el par
    (EOF, PC) de ``other`` cuando su patron correlaciona negativamente con el
    homologo de ``reference``.

    Parameters
    ----------
    reference : EOFResult
        Descomposicion que fija el signo.
    other : EOFResult
        Descomposicion a alinear. Debe estar definida sobre el mismo conjunto
        de pixeles.

    Returns
    -------
    EOFResult
        Copia de ``other`` con los signos alineados. No se modifica la entrada.

    Raises
    ------
    ValueError
        Si las dos descomposiciones no comparten el espacio.

    Notes
    -----
    Solo cambia el signo: no reordena modos. Si entre las dos versiones el
    orden de los modos cambia (porque sus varianzas se cruzan), alinear el
    signo no basta y hay que emparejarlos explicitamente por correlacion de
    patrones antes de comparar.
    """
    if reference.n_space != other.n_space:
        raise ValueError(
            f"Las descomposiciones no comparten el espacio: "
            f"{reference.n_space} vs {other.n_space} pixeles."
        )

    k = min(reference.n_modes, other.n_modes)
    signo = np.ones(other.n_modes)
    for i in range(k):
        if float(reference.eofs[i] @ other.eofs[i]) < 0:
            signo[i] = -1.0

    alineado = EOFResult(
        eofs=other.eofs * signo[:, None],
        pcs=other.pcs * signo[None, :],
        eigenvalues=other.eigenvalues,
        variance_fraction=other.variance_fraction,
        singular_values=other.singular_values,
        mean=other.mean,
        rank=other.rank,
        n_time=other.n_time,
        n_space=other.n_space,
        scaling=other.scaling,
        weights=other.weights,
        index=other.index,
    )
    return alineado


def eof_from_dataarray(
    da: xr.DataArray,
    n_modes: int | None = None,
    scaling: Scaling = "orthonormal",
    time_dim: str = "time",
    lat_dim: str = "latitude",
    weight: bool = True,
) -> EOFResult:
    """Descomponer un campo de ``xarray`` en EOF, con ponderacion por area.

    Encadena :func:`enso_chile.cube.apply_area_weights`,
    :func:`enso_chile.cube.flatten_space` y :func:`eof_decomposition`, y guarda
    lo necesario para devolver patrones y PC con coordenadas.

    Parameters
    ----------
    da : xarray.DataArray
        Campo de anomalias con dimension temporal. Los pixeles de tierra (NaN
        en toda la serie) se descartan.
    n_modes : int or None, optional
        Numero de modos a retener.
    scaling : {'orthonormal', 'std_pc'}, optional
        Escalado del par (EOF, PC).
    time_dim : str, optional
        Nombre de la dimension temporal.
    lat_dim : str, optional
        Nombre de la coordenada de latitud, para los pesos.
    weight : bool, optional
        Aplicar ``sqrt(cos(lat))``. Desactivarlo solo para diagnostico: el
        proyecto exige ponderacion consistente en todos los analisis de modos.

    Returns
    -------
    EOFResult
        Resultado con ``weights`` e ``index`` poblados, listo para
        :meth:`EOFResult.eofs_as_maps` y :meth:`EOFResult.pcs_as_dataarray`.

    Notes
    -----
    Un analisis por caja, variable y profundidad. No pasar aqui un campo con
    varias profundidades a la vez si lo que se busca son modos horizontales:
    eso mezcla niveles en un mismo espacio y el resultado no es comparable con
    los EOF de una sola profundidad.
    """
    campo = apply_area_weights(da, lat_dim) if weight else da
    matrix, index = flatten_space(campo, time_dim=time_dim, drop_land=True)

    result = eof_decomposition(matrix, n_modes=n_modes, scaling=scaling)
    result.index = index
    result.weights = area_weights(da, lat_dim) if weight else None
    return result


# --------------------------------------------------------------------------- #
# EOF combinado (varias capas o variables en un solo analisis)
# --------------------------------------------------------------------------- #


@dataclass
class CombinedEOFResult:
    """Resultado de un EOF combinado sobre bloques concatenados.

    Attributes
    ----------
    eigenvalues : numpy.ndarray
        Autovalores ``s**2 / (T - 1)`` de los modos retenidos.
    variance_fraction : numpy.ndarray
        Fraccion del total combinado (suma de todos los autovalores del rango).
        Depende de lo que se incluyo en cada bloque y no es comparable entre
        truncados distintos; para eso usar :meth:`r2_by_block`.
    vectors : numpy.ndarray
        Modos en el espacio de canales, forma ``(n_modes, n_canales)``, norma 1.
    pcs : numpy.ndarray
        PC combinadas ``U * s``, forma ``(T, n_modes)``.
    block_sizes : tuple of int
        Canales de cada bloque, en orden.
    block_scales : numpy.ndarray
        Divisor aplicado a cada bloque (raiz de su varianza total; 1 sin normalizar).
    retained : numpy.ndarray
        Fraccion de la varianza total de cada bloque que representan sus canales:
        1 si el bloque es el campo completo, la varianza de las PC retenidas si
        viene de un prefiltro EOF.
    normalized : bool
        Si los bloques se normalizaron a varianza total 1.
    n_time : int
        Numero de instantes.
    """

    eigenvalues: np.ndarray
    variance_fraction: np.ndarray
    vectors: np.ndarray
    pcs: np.ndarray
    block_sizes: tuple[int, ...]
    block_scales: np.ndarray
    retained: np.ndarray
    normalized: bool
    n_time: int

    @property
    def block_edges(self) -> np.ndarray:
        """Indices de inicio y fin de cada bloque en el espacio de canales."""
        return np.cumsum((0,) + tuple(self.block_sizes))

    def loadings(self, n_modes: int | None = None) -> np.ndarray:
        """Cargas ``V s / sqrt(T - 1)``, forma ``(n_canales, n_modes)``.

        Con :meth:`scores` cumplen ``Z ~ scores @ loadings.T``; es lo que se rota.
        """
        k = len(self.eigenvalues) if n_modes is None else n_modes
        return self.vectors[:k].T * np.sqrt(self.eigenvalues[:k])

    def scores(self, n_modes: int | None = None) -> np.ndarray:
        """PC estandarizadas (varianza 1), forma ``(T, n_modes)``."""
        k = len(self.eigenvalues) if n_modes is None else n_modes
        return self.pcs[:, :k] / np.sqrt(self.eigenvalues[:k])

    def _energia_por_bloque(self, loadings: np.ndarray | None) -> np.ndarray:
        A = self.loadings() if loadings is None else loadings
        e = self.block_edges
        return np.array([(A[a:b] ** 2).sum(0) for a, b in zip(e[:-1], e[1:])]).T

    def block_share(self, loadings: np.ndarray | None = None) -> np.ndarray:
        """Reparto de la energia de cada componente entre bloques (cada fila suma 1).

        Parameters
        ----------
        loadings : numpy.ndarray, optional
            Cargas ``(n_canales, k)``, por ejemplo rotadas; por defecto las propias.
        """
        e = self._energia_por_bloque(loadings)
        return e / e.sum(1, keepdims=True)

    def r2_by_block(self, loadings: np.ndarray | None = None) -> np.ndarray:
        """Fraccion de la varianza **total** de cada bloque explicada por cada componente.

        Parameters
        ----------
        loadings : numpy.ndarray, optional
            Cargas ``(n_canales, k)``. Es varianza explicada solo para componentes
            ortogonales (sin rotar o varimax); en una rotacion oblicua es la suma de
            cargas al cuadrado del patron.

        Returns
        -------
        numpy.ndarray
            Forma ``(k, n_bloques)``, entre 0 y 1. Cada bloque normalizado tiene
            varianza 1, asi que la varianza de la componente en el bloque es la
            fraccion de lo retenido; se multiplica por ``retained``.

        Raises
        ------
        ValueError
            Si los bloques no se normalizaron.
        """
        if not self.normalized:
            raise ValueError("r2_by_block requiere bloques normalizados (normalize=True).")
        return self._energia_por_bloque(loadings) * self.retained[None, :]


def combined_blocks(blocks, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Centrar, normalizar y concatenar bloques ``(T, n_d)``.

    Parameters
    ----------
    blocks : sequence of numpy.ndarray
        Un bloque por capa o variable, todos con el mismo numero de filas (tiempo).
    normalize : bool, optional
        Dividir cada bloque por la raiz de su varianza total para que todos aporten
        varianza 1. Sin esto domina el bloque de mayor varianza (o el de mas pixeles).

    Returns
    -------
    Z : numpy.ndarray
        Matriz concatenada ``(T, sum n_d)``.
    scales : numpy.ndarray
        Divisor aplicado a cada bloque.
    """
    blocks = [np.asarray(b, dtype=float) for b in blocks]
    n_time = blocks[0].shape[0]
    if any(b.ndim != 2 or b.shape[0] != n_time for b in blocks):
        raise ValueError("Todos los bloques deben ser 2-D con el mismo numero de filas.")
    centrados = [b - b.mean(0) for b in blocks]
    scales = np.array([np.sqrt((b ** 2).sum() / (n_time - 1)) if normalize else 1.0 for b in centrados])
    if np.any(scales == 0):
        raise ValueError("Un bloque tiene varianza nula.")
    return np.hstack([b / c for b, c in zip(centrados, scales)]), scales


def combined_eof(blocks, n_modes: int | None = None, normalize: bool = True,
                 retained=None) -> CombinedEOFResult:
    """EOF combinado de varios bloques (capas o variables) concatenados.

    Parameters
    ----------
    blocks : sequence of numpy.ndarray
        Bloques ``(T, n_d)``: el campo ponderado de cada capa, o sus PC.
    n_modes : int or None, optional
        Modos a retener; ``None`` retiene el rango.
    normalize : bool, optional
        Ver :func:`combined_blocks`.
    retained : sequence of float, optional
        Fraccion de la varianza total de cada bloque que representan sus columnas
        (por defecto 1).

    Returns
    -------
    CombinedEOFResult

    Notes
    -----
    Si cada bloque son **todas** las PC (``orthonormal``) de su capa, el resultado es
    identico al EOF del campo apilado: las PC son coordenadas del campo en una base
    ortonormal, y un cambio de base ortonormal por bloques no altera varianzas ni
    covarianzas. Con un prefiltro truncado es la misma descomposicion restringida
    al subespacio retenido.
    """
    Z, scales = combined_blocks(blocks, normalize=normalize)
    n_time = Z.shape[0]
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    rank = min(n_time - 1, Z.shape[1])
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    eigenvalues = s ** 2 / (n_time - 1)
    k = rank if n_modes is None else n_modes
    if not 1 <= k <= rank:
        raise ValueError(f"n_modes debe estar en [1, {rank}], no {k}.")
    vt, pcs = _fix_signs(vt, u * s)
    ret = np.ones(len(scales)) if retained is None else np.asarray(retained, dtype=float)
    return CombinedEOFResult(
        eigenvalues=eigenvalues[:k],
        variance_fraction=(eigenvalues / eigenvalues.sum())[:k],
        vectors=vt[:k],
        pcs=pcs[:, :k],
        block_sizes=tuple(np.asarray(b).shape[1] for b in blocks),
        block_scales=scales,
        retained=ret,
        normalized=normalize,
        n_time=n_time,
    )


def combined_eof_from_results(results, n_pcs: int, n_modes: int | None = None,
                              normalize: bool = True) -> CombinedEOFResult:
    """EOF combinado sobre las primeras ``n_pcs`` PC de cada capa (prefiltro EOF).

    Parameters
    ----------
    results : sequence of EOFResult
        Un EOF por capa, con escalado ``'orthonormal'`` y al menos ``n_pcs`` modos.
    n_pcs : int
        PC retenidas por capa (el proyecto fija 6).
    n_modes : int or None, optional
        Modos combinados a retener.
    normalize : bool, optional
        Ver :func:`combined_blocks`.
    """
    for r in results:
        if r.scaling != "orthonormal":
            raise ValueError("El prefiltro requiere EOF con escalado 'orthonormal'.")
        if r.n_modes < n_pcs:
            raise ValueError(f"Un EOF tiene {r.n_modes} modos; se piden {n_pcs}.")
    return combined_eof([r.pcs[:, :n_pcs] for r in results], n_modes=n_modes, normalize=normalize,
                        retained=[r.variance_fraction[:n_pcs].sum() for r in results])


def pixel_loadings(loadings: np.ndarray, results, n_pcs: int) -> np.ndarray:
    """Llevar cargas por canal (PC de cada capa) a cargas por pixel, apiladas por capa.

    Parameters
    ----------
    loadings : numpy.ndarray
        Cargas ``(n_capas * n_pcs, k)`` de un EOF combinado con prefiltro.
    results : sequence of EOFResult
        Los EOF por capa del prefiltro, en el mismo orden.
    n_pcs : int
        PC por capa.

    Returns
    -------
    numpy.ndarray
        Cargas ``(sum pixeles, k)`` en el espacio ponderado de cada capa
        normalizada: la matriz que se rota con pixeles como filas. La
        transformacion es una isometria por bloques: conserva los productos
        internos entre columnas.
    """
    return np.vstack([r.eofs[:n_pcs].T @ loadings[d * n_pcs:(d + 1) * n_pcs]
                      for d, r in enumerate(results)])


def scores_from_pattern(Z: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Puntajes por minimos cuadrados de los datos sobre un patron, estandarizados.

    Parameters
    ----------
    Z : numpy.ndarray
        Datos centrados ``(T, n_canales)``, por ejemplo de :func:`combined_blocks`.
    pattern : numpy.ndarray
        Patron ``(n_canales, k)``: cargas sin rotar o rotadas (ortogonal u oblicua).

    Returns
    -------
    numpy.ndarray
        ``Z P (P^T P)^{-1}`` con cada columna llevada a varianza 1. Para un patron
        rotado coincide con los puntajes rotados (``F R`` en una rotacion ortogonal,
        ``F (T^{-1})^T`` en una oblicua), porque el residuo del truncado es
        ortogonal al subespacio de las cargas. Sirve para proyectar meses o eventos.
    """
    F = Z @ pattern @ np.linalg.inv(pattern.T @ pattern)
    return (F - F.mean(0)) / F.std(0, ddof=1)
