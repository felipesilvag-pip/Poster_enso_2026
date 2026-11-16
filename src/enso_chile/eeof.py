"""EOF extendidos (EEOF) / SSA multicanal (M-SSA).

Descomposicion de ``K`` series (por ejemplo las PC de un prefiltro EOF) con ``L``
rezagos: matriz de trayectoria Hankel por bloques, SVD, reconstruccion por
promedio de antidiagonales y agrupacion de modos por W-correlacion.

Decisiones
----------
* Estimador de trayectoria de Broomhead & King (1986), no el Toeplitz de Vautard &
  Ghil (1989): SVD de la matriz de trayectoria de los canales centrados.
* Orden de columnas por **bloques de rezago**: ``X[i, l * K + k] = Z[i + l, k]``.
* Autovalores ``s**2 / N`` con ``N = T - L + 1``; ST-PC ``U * s``, de modo que
  ``X = st_pcs @ V``.
* Agrupacion por W-correlacion (Golyandina, Nekrutkin & Zhigljavsky 2001); el
  criterio de North et al. (1982) queda como alternativa.

Referencias generales: Ghil et al. (2002), Rev. Geophys.; Hannachi, Jolliffe &
Stephenson (2007), Int. J. Climatol.
"""

# LEGACY: Entendiendo_EOF.py, ejecutar_y_visualizar_eeof (l. 590), reconstruct_rc
# (l. 916 y 2602-2615) y W-correlacion (compute_w_correlation_matrix, l. 965-980;
# calc_w_correlation, l. 2638-2646). Misma definicion de W-correlacion; aqui sin
# salto entre rezagos (STEP = 1) y sin valor absoluto.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "trajectory_matrix",
    "eeof",
    "reconstruction_weights",
    "reconstruct_components",
    "wcorr_matrix",
    "find_pairs",
    "find_pairs_north",
    "pair_lagged_corr",
    "diagonal_average",
    "IOSSAResult",
    "iterative_ossa",
    "match_groups_by_subspace",
    "derivssa",
    "DEFAULT_BANDS",
    "band_fractions",
    "group_by_band",
    "spectral_purity",
]


def trajectory_matrix(Z: np.ndarray, L: int) -> np.ndarray:
    """Construir la matriz de trayectoria (Hankel por bloques) de ``K`` series.

    Parameters
    ----------
    Z : numpy.ndarray
        Series de forma ``(T, K)``.
    L : int
        Largo de la ventana (numero de rezagos), ``1 <= L <= T``.

    Returns
    -------
    numpy.ndarray
        Matriz ``(T - L + 1, L * K)`` con ``X[i, l * K + k] = Z[i + l, k]``: el
        bloque ``l`` son los ``K`` canales en el rezago ``l``.

    Raises
    ------
    ValueError
        Si ``Z`` no es 2D o ``L`` esta fuera de rango.

    Notes
    -----
    Con ``L = 1`` la matriz es ``Z`` y el EEOF se reduce al EOF clasico.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z debe ser 2D (T, K), no {Z.ndim}D.")
    T, K = Z.shape
    if not 1 <= L <= T:
        raise ValueError(f"L debe estar en [1, {T}], no {L}.")
    N = T - L + 1
    ventanas = np.lib.stride_tricks.sliding_window_view(Z, L, axis=0)  # (N, K, L)
    return np.ascontiguousarray(ventanas.transpose(0, 2, 1)).reshape(N, L * K)


def eeof(Z: np.ndarray, L: int, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descomponer ``Z`` en EEOF / M-SSA.

    Parameters
    ----------
    Z : numpy.ndarray
        Series de forma ``(T, K)``. Cada canal se centra antes de descomponer.
    L : int
        Largo de la ventana.
    n_modes : int
        Modos a retener, entre 1 y ``min(T - L + 1, L * K)``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Autovalores ``s**2 / N``, de mayor a menor.
    st_eofs : numpy.ndarray
        EEOF, forma ``(n_modes, K, L)`` (modo, canal, rezago), norma 1 por modo.
    st_pcs : numpy.ndarray
        ST-PC ``U * s``, forma ``(T - L + 1, n_modes)``; ortogonales entre si.
    """
    Z = np.asarray(Z, dtype=float)
    T, K = Z.shape
    X = trajectory_matrix(Z - Z.mean(0), L)
    N = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if not 1 <= n_modes <= len(s):
        raise ValueError(f"n_modes debe estar en [1, {len(s)}], no {n_modes}.")
    eigenvalues = s[:n_modes] ** 2 / N
    st_eofs = Vt[:n_modes].reshape(n_modes, L, K).transpose(0, 2, 1)
    st_pcs = U[:, :n_modes] * s[:n_modes]
    return eigenvalues, st_eofs, st_pcs


def reconstruction_weights(T: int, L: int) -> np.ndarray:
    """Numero de veces que cada instante entra en la matriz de trayectoria.

    Parameters
    ----------
    T : int
        Largo de la serie.
    L : int
        Largo de la ventana.

    Returns
    -------
    numpy.ndarray
        ``w[t] = min(t + 1, L*, K*, T - t)`` con ``L* = min(L, N)``,
        ``K* = max(L, N)`` y ``N = T - L + 1``: ``1, 2, ..., L*, ..., 2, 1``. Son
        los pesos del promedio por antidiagonal y de la W-correlacion.
    """
    N = T - L + 1
    if N < 1:
        raise ValueError(f"L = {L} es mayor que T = {T}.")
    t = np.arange(T)
    return np.minimum.reduce([t + 1, np.full(T, min(L, N)), np.full(T, max(L, N)), T - t]).astype(float)


def reconstruct_components(st_eofs: np.ndarray, st_pcs: np.ndarray, modes: list[int],
                           T: int) -> np.ndarray:
    """Reconstruir las series de un grupo de modos (promedio por antidiagonal).

    Parameters
    ----------
    st_eofs : numpy.ndarray
        EEOF, forma ``(n_modes, K, L)``.
    st_pcs : numpy.ndarray
        ST-PC, forma ``(T - L + 1, n_modes)``.
    modes : list of int
        Modos (base 0) a sumar; los dos miembros de un par oscilatorio van juntos.
    T : int
        Largo de la serie original.

    Returns
    -------
    numpy.ndarray
        Componente reconstruido (RC), forma ``(T, K)``. Con todos los modos del
        rango se recupera ``Z - Z.mean(0)`` salvo redondeo.

    Notes
    -----
    El elemento ``(i, l, k)`` de la matriz reconstruida aporta al instante ``i + l``;
    cada instante se promedia sobre sus aportes. Los extremos tienen menos aportes y,
    por tanto, mas incertidumbre.
    """
    modes = list(modes)
    _, K, L = st_eofs.shape
    N = st_pcs.shape[0]
    if N != T - L + 1:
        raise ValueError(f"st_pcs tiene {N} filas, pero T - L + 1 = {T - L + 1}.")
    V = st_eofs[modes].transpose(0, 2, 1).reshape(len(modes), L * K)
    R = (st_pcs[:, modes] @ V).reshape(N, L, K)
    rec = np.zeros((T, K))
    for l in range(L):
        rec[l:l + N] += R[:, l, :]
    return rec / reconstruction_weights(T, L)[:, None]


def wcorr_matrix(rcs: np.ndarray, L: int) -> np.ndarray:
    """Matriz de W-correlacion entre componentes reconstruidos.

    Parameters
    ----------
    rcs : numpy.ndarray
        RC de cada modo (o grupo), forma ``(n, T, K)``.
    L : int
        Largo de la ventana con que se calcularon.

    Returns
    -------
    numpy.ndarray
        Matriz simetrica ``(n, n)`` con
        ``rho_w(a, b) = (a, b)_w / sqrt((a, a)_w (b, b)_w)`` y
        ``(a, b)_w = sum_t w_t sum_k a[t, k] b[t, k]``, con ``w_t`` de
        :func:`reconstruction_weights`.

    Notes
    -----
    No es Pearson: no centra y pondera cada instante por cuantas veces entra en la
    matriz de trayectoria. ``|rho_w|`` alto indica componentes no separables, que
    conviene agrupar; bajo, separables. Los RC de los dos modos de una oscilacion
    salen en fase u oposicion (``|rho_w| ~ 1``) aunque sus ST-PC esten en cuadratura.
    """
    n, T, _ = rcs.shape
    w = reconstruction_weights(T, L)
    A = (rcs * np.sqrt(w)[None, :, None]).reshape(n, -1)
    G = A @ A.T
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def find_pairs(wcorr: np.ndarray, threshold: float = 0.7,
               isolation: float = 0.6) -> list[tuple[int, int]]:
    """Identificar pares oscilatorios por W-correlacion.

    Parameters
    ----------
    wcorr : numpy.ndarray
        Salida de :func:`wcorr_matrix` para modos elementales consecutivos.
    threshold : float, optional
        ``|rho_w|`` minimo entre ``i`` e ``i + 1``.
    isolation : float, optional
        ``|rho_w|`` maximo de cada miembro con cualquier otro modo.

    Returns
    -------
    list of tuple of int
        Pares ``(i, i + 1)`` en base 0.

    Notes
    -----
    Los umbrales son una convencion de trabajo, no un estandar. Confirmar con la
    cuadratura de las ST-PC (:func:`pair_lagged_corr`) y con que ambos modos
    compartan frecuencia.
    """
    A = np.abs(wcorr)
    n = A.shape[0]
    pares = []
    for i in range(n - 1):
        if A[i, i + 1] < threshold:
            continue
        otros = [m for m in range(n) if m not in (i, i + 1)]
        if not otros or (A[i, otros].max() < isolation and A[i + 1, otros].max() < isolation):
            pares.append((i, i + 1))
    return pares


def find_pairs_north(eigenvalues: np.ndarray, n_eff: float) -> list[tuple[int, int]]:
    """Pares de autovalores no separables segun North et al. (1982).

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Autovalores en orden descendente.
    n_eff : float
        Numero efectivo de datos independientes.

    Returns
    -------
    list of tuple of int
        Pares ``(i, i + 1)`` que no se separan entre si pero si del modo anterior y
        del siguiente, con ``delta_i = lambda_i * sqrt(2 / n_eff)``.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    delta = lam * np.sqrt(2.0 / n_eff)
    separado = lam[:-1] - lam[1:] > delta[:-1]
    return [(i, i + 1) for i in range(len(lam) - 2)
            if not separado[i] and (i == 0 or separado[i - 1]) and separado[i + 1]]


def pair_lagged_corr(st_pcs: np.ndarray, i: int, j: int,
                     max_lag: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Correlacion cruzada entre dos ST-PC.

    Parameters
    ----------
    st_pcs : numpy.ndarray
        ST-PC, forma ``(N, n_modes)``.
    i, j : int
        Modos (base 0).
    max_lag : int, optional
        Rezago maximo.

    Returns
    -------
    lags : numpy.ndarray
        De ``-max_lag`` a ``max_lag``.
    corr : numpy.ndarray
        ``corr[k] = corr(st_pcs[t, i], st_pcs[t + lags[k], j])``. Un par oscilatorio
        de periodo ``P`` tiene corr ~ 0 en rezago 0 y un maximo de ``|corr|`` cerca
        de ``+-P/4``.
    """
    a, b = st_pcs[:, i], st_pcs[:, j]
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.corrcoef(a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)])[0, 1]
                     for k in lags])
    return lags, corr


def diagonal_average(Y: np.ndarray, L: int, K: int, T: int) -> np.ndarray:
    """Promediar por antidiagonales una matriz con la forma de la trayectoria.

    Parameters
    ----------
    Y : numpy.ndarray
        Matriz ``(T - L + 1, L * K)`` en el orden de :func:`trajectory_matrix`
        (no necesita ser Hankel).
    L : int
        Largo de la ventana.
    K : int
        Numero de canales.
    T : int
        Largo de la serie.

    Returns
    -------
    numpy.ndarray
        Series ``(T, K)``: la proyeccion de ``Y`` sobre las matrices Hankel por
        bloques (la operacion ``T^{-1} H`` de Golyandina et al.). Es la inversa de
        :func:`trajectory_matrix` sobre matrices que ya son Hankel.
    """
    N = T - L + 1
    if Y.shape != (N, L * K):
        raise ValueError(f"Y debe tener forma {(N, L * K)}, no {Y.shape}.")
    R = Y.reshape(N, L, K)
    rec = np.zeros((T, K))
    for l in range(L):
        rec[l:l + N] += R[:, l, :]
    return rec / reconstruction_weights(T, L)[:, None]


@dataclass
class IOSSAResult:
    """Resultado de :func:`iterative_ossa`.

    Attributes
    ----------
    series : list of numpy.ndarray
        Los dos componentes reconstruidos, cada uno ``(T, K)``; suman el RC del grupo.
    n_iter : int
        Iteraciones realizadas.
    sigmas : numpy.ndarray
        Valores singulares de la ultima ``(L, R)``-SVD.
    assignment : list of int
        Componente (0 o 1) de cada termino de la ``(L, R)``-SVD.
    converged : bool
        Si se alcanzo la tolerancia antes de ``max_iter``.
    lr_wcorr : float
        W-correlacion ``(L, R)`` entre los dos componentes (Golyandina & Shlemov 2015,
        ec. 12): la medida de separabilidad adecuada para componentes oblicuos. Cerca
        de 0 = bien separados. La W-correlacion clasica no tiene por que serlo.
    """

    series: list
    n_iter: int
    sigmas: np.ndarray
    assignment: list
    converged: bool
    lr_wcorr: float


def iterative_ossa(Z: np.ndarray, L: int, group: list[int], J1: list[int], kappa: float | None = None,
                   eps: float = 1e-5, max_iter: int = 300):
    """Iterative Oblique SSA anidado (Golyandina & Shlemov 2015) sobre un grupo de modos.

    Separa en dos componentes un subespacio que el SSA basico mezcla (por ejemplo,
    una oscilacion y una tendencia, o dos senos de frecuencias cercanas): busca
    productos internos ``(L, R)`` en los que ambos componentes sean bi-ortogonales.

    Parameters
    ----------
    Z : numpy.ndarray
        Series ``(T, K)``; cada canal se centra.
    L : int
        Largo de la ventana.
    group : list of int
        Modos (base 0) del SSA basico que forman el subespacio a refinar.
    J1 : list of int
        Subconjunto de ``group`` que inicializa el primer componente; el resto del
        grupo inicializa el segundo.
    kappa : float or None, optional
        Factor de separacion de la correccion sigma (algoritmo 5); ``None`` la
        desactiva. Con datos ruidosos suele hacer falta para converger (2 es un
        valor razonable).
    eps : float, optional
        Tolerancia: se detiene cuando el cambio cuadratico medio de ambos
        componentes es menor que ``eps**2``.
    max_iter : int, optional
        Iteraciones maximas.

    Returns
    -------
    IOSSAResult
        Componentes, iteraciones, valores singulares, asignacion, convergencia y
        W-correlacion ``(L, R)`` entre componentes.

    Notes
    -----
    Algoritmos 1 (``(L, R)``-SVD reducida a una SVD ordinaria con ``O_L = U^+`` y
    ``O_R = V^+``), 3 (iteracion), 4 (productos internos a partir de los espacios de
    trayectoria de cada componente hankelizado y truncado a su rango, proyectados
    sobre los espacios fila y columna del grupo) y 5 (correccion sigma). Cada termino
    de la ``(L, R)``-SVD se asigna al componente donde concentra su energia, lo que
    equivale a la agrupacion del articulo cuando la separacion es buena.
    """
    Z = np.asarray(Z, dtype=float)
    T, K = Z.shape
    group = list(group)
    if not J1 or not set(J1) < set(group):
        raise ValueError("J1 debe ser un subconjunto propio y no vacio de group.")
    X = trajectory_matrix(Z - Z.mean(0), L)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UI, sI, VI = U[:, group], s[group], Vt[group].T
    Y = (UI * sI) @ VI.T
    P_col, P_row = UI @ UI.T, VI @ VI.T
    J = [[group.index(i) for i in J1], [group.index(i) for i in group if i not in J1]]
    previas = [diagonal_average((UI[:, j] * sI[j]) @ VI[:, j].T, L, K, T) for j in J]
    converged = False
    for it in range(1, max_iter + 1):
        Uh, Vh, lams = [], [], []
        for m in range(2):
            u, ss, vt = np.linalg.svd(trajectory_matrix(previas[m], L), full_matrices=False)
            r_m = len(J[m])
            Uh.append(P_col @ u[:, :r_m])
            Vh.append(P_row @ vt[:r_m].T)
            lams.append(ss[:r_m] ** 2)
        if kappa is not None and lams[0][-1] < kappa ** 2 * lams[1][0]:
            mu = kappa * np.sqrt(lams[1][0] / lams[0][-1])
            Uh[1], Vh[1] = Uh[1] * np.sqrt(mu), Vh[1] * np.sqrt(mu)
        Uhat, Vhat = np.hstack(Uh), np.hstack(Vh)
        Uc, sc, Vct = np.linalg.svd(np.linalg.pinv(Uhat) @ Y @ np.linalg.pinv(Vhat).T)
        P, Q = Uhat @ Uc, Vhat @ Vct.T
        r1 = len(J[0])
        assignment = [0 if (Uc[:r1, i] ** 2).sum() >= (Uc[r1:, i] ** 2).sum() else 1 for i in range(len(sc))]
        partes = [np.zeros_like(Y), np.zeros_like(Y)]
        for i, g in enumerate(assignment):
            partes[g] += sc[i] * np.outer(P[:, i], Q[:, i])
        nuevas = [diagonal_average(p, L, K, T) for p in partes]
        cambio = max(((nuevas[m] - previas[m]) ** 2).sum() / T for m in range(2))
        previas = nuevas
        if cambio < eps ** 2:
            converged = True
            break
    O_L, O_R = np.linalg.pinv(Uhat), np.linalg.pinv(Vhat)
    A = [O_L @ trajectory_matrix(c, L) @ O_R.T for c in previas]
    lr = float((A[0] * A[1]).sum() / np.sqrt((A[0] ** 2).sum() * (A[1] ** 2).sum()))
    return IOSSAResult(series=previas, n_iter=it, sigmas=sc, assignment=assignment,
                       converged=converged, lr_wcorr=lr)


def match_groups_by_subspace(st_eofs: np.ndarray, references: list[np.ndarray],
                             n_candidates: int | None = None) -> tuple[list[list[int]], np.ndarray]:
    """Asignar modos de una descomposicion a grupos de referencia por semejanza de subespacio.

    Sirve para reconocer los mismos grupos (por ejemplo ENSO y lento) en otra descomposicion
    (una replica bootstrap, otro periodo) sin depender de umbrales de periodo, del orden de los
    modos ni de su signo.

    Parameters
    ----------
    st_eofs : numpy.ndarray
        EEOF candidatas, forma ``(n, K, L)``.
    references : list of numpy.ndarray
        Un subespacio por grupo: matrices ``(r_g, K * L)`` de filas ortonormales en el orden
        por bloques de rezago (``st_eofs[idx].transpose(0, 2, 1).reshape(r_g, -1)``).
    n_candidates : int, optional
        Modos candidatos (los primeros); por defecto todos.

    Returns
    -------
    groups : list of list of int
        Para cada grupo, los ``r_g`` modos (base 0) que mas se proyectan sobre su subespacio.
        Se asignan en el orden de ``references``, sin repetir modos.
    similarity : numpy.ndarray
        Energia media de proyeccion de los modos elegidos sobre su subespacio (1 = identico,
        0 = ortogonal), una por grupo.
    """
    n = st_eofs.shape[0] if n_candidates is None else n_candidates
    V = st_eofs[:n].transpose(0, 2, 1).reshape(n, -1)
    usados: set[int] = set()
    groups, similarity = [], []
    for S in references:
        energia = ((V @ S.T) ** 2).sum(1)
        elegidos = [int(m) for m in np.argsort(-energia) if m not in usados][:S.shape[0]]
        usados.update(elegidos)
        groups.append(sorted(elegidos))
        similarity.append(float(energia[elegidos].mean()))
    return groups, np.array(similarity)


def derivssa(st_eofs: np.ndarray, st_pcs: np.ndarray, modes: list[int], gamma: float = np.inf,
             normalize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DerivSSA anidado (Golyandina & Shlemov 2015, algoritmo 6) sobre un grupo de modos.

    Refina la descomposicion del subespacio de ``modes`` (el de senal) para que
    componentes de frecuencias distintas pero contribucion parecida dejen de
    mezclarse: la diferencia temporal multiplica la amplitud de un seno de periodo
    ``P`` por ``2 sin(pi / P)`` sin cambiar su subespacio, y rompe la igualdad de
    valores singulares (falta de separabilidad fuerte).

    Parameters
    ----------
    st_eofs : numpy.ndarray
        EEOF ``(n_modes, K, L)`` de :func:`eeof`.
    st_pcs : numpy.ndarray
        ST-PC ``(N, n_modes)`` de :func:`eeof`.
    modes : list of int
        Modos (base 0) del subespacio a refinar. Debe excluir el ruido: la
        diferencia realza las frecuencias altas.
    gamma : float, optional
        Peso de la derivada; ``numpy.inf`` usa solo la derivada. Con
        ``normalize=True`` el resultado no depende de ``gamma``.
    normalize : bool, optional
        ``True`` (por defecto de Rssa ``fossa``): las ST-PC se blanquean antes de
        diferenciar; los modos salen ordenados por frecuencia decreciente y las
        ST-PC nuevas son ortonormales (las EEOF, oblicuas). ``False``: version del
        articulo, ``SVD([Y : gamma Phi(Y)])``; EEOF nuevas ortonormales.

    Returns
    -------
    st_eofs_new : numpy.ndarray
        EEOF nuevas ``(len(modes), K, L)``.
    st_pcs_new : numpy.ndarray
        ST-PC nuevas ``(N, len(modes))``. Con :func:`reconstruct_components` sobre
        todos los modos nuevos se recupera el RC del grupo original.
    eigenvalues : numpy.ndarray
        Autovalores del problema ``r x r`` (orden decreciente).

    Notes
    -----
    Formulacion climatologica (filas = tiempo): ``Y = P E^T``, ``Phi`` = diferencias
    entre filas consecutivas. Con ``M = Y_t^T Y_t + gamma^2 dY_t^T dY_t`` (``Y_t = P``
    o ``P / s``) y ``M = C Lambda C^T``, las ST-PC nuevas son ``Y_t C`` y las EEOF
    ``E C`` (o ``E S C``). Equivale a la SVD de ``Z`` del algoritmo 6 (proposicion 1)
    y al codigo de Rssa ``fossa.ssa`` (``R/ossa.R``). La adaptacion a la trayectoria
    climatologica es propia: el articulo define el algoritmo para cualquier ``Y``
    (observacion 6). Validado en el sandbox 05 (ejemplo de la sec. 4.3).
    """
    modes = list(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes debe ser una lista no vacia y sin repetidos.")
    P = np.asarray(st_pcs, dtype=float)[:, modes]
    E = np.asarray(st_eofs, dtype=float)[modes]
    s = np.linalg.norm(P, axis=0)
    Yt = P / s if normalize else P
    dY = np.diff(Yt, axis=0)
    M = dY.T @ dY if np.isinf(gamma) else Yt.T @ Yt + gamma ** 2 * dY.T @ dY
    lam, C = np.linalg.eigh(M)
    orden = np.argsort(lam)[::-1]
    lam, C = lam[orden], C[:, orden]
    if normalize:
        return np.einsum("jkl,j,ji->ikl", E, s, C), Yt @ C, lam
    return np.einsum("jkl,ji->ikl", E, C), P @ C, lam


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "decadal": (84.0, np.inf),
    "ENSO": (24.0, 84.0),
    "cuasi-bienal": (15.0, 24.0),
    "resto": (0.0, 15.0),
}
"""Bandas de periodo (meses, ``(min, max]``) acordadas para el EEOF combinado."""


def band_fractions(x: np.ndarray, bands: dict | None = None) -> dict[str, float]:
    """Fraccion de la potencia del periodograma de ``x`` en cada banda.

    Parameters
    ----------
    x : numpy.ndarray
        Serie 1D (se centra; la frecuencia 0 se excluye).
    bands : dict, optional
        ``{nombre: (periodo_min, periodo_max)}`` en pasos de tiempo; por defecto
        :data:`DEFAULT_BANDS`.

    Returns
    -------
    dict
        Fraccion por banda; suman 1 si las bandas cubren todos los periodos.
    """
    bands = DEFAULT_BANDS if bands is None else bands
    x = np.asarray(x, dtype=float)
    f = np.fft.rfftfreq(len(x))[1:]
    pot = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    per, pot = 1 / f, pot / pot.sum()
    return {b: float(pot[(per > a) & (per <= c)].sum()) for b, (a, c) in bands.items()}


def group_by_band(st_pcs: np.ndarray, bands: dict | None = None, wcorr: np.ndarray | None = None,
                  threshold: float = 0.6) -> dict[str, list[int]]:
    """Asigna cada modo a la banda donde su ST-PC concentra mas potencia.

    Parameters
    ----------
    st_pcs : numpy.ndarray
        ST-PC ``(N, n)``.
    bands : dict, optional
        Bandas de periodo; por defecto :data:`DEFAULT_BANDS`.
    wcorr : numpy.ndarray, optional
        W-correlacion ``(n, n)``. Si se da, los modos conectados por
        ``|W| >= threshold`` (componentes conexas) forman un bloque que va entero a
        la banda donde suma mas potencia (regla para el SSA basico, que no parte
        pares). Para modos de :func:`derivssa` **no** usarla: con modos oblicuos
        ordenados por frecuencia encadena bandas vecinas (sandbox 05 [11]).
    threshold : float, optional
        Umbral de ``|W|`` para unir modos.

    Returns
    -------
    dict
        ``{banda: [modos base 0]}``.
    """
    bands = DEFAULT_BANDS if bands is None else bands
    n = st_pcs.shape[1]
    fr = [band_fractions(st_pcs[:, m], bands) for m in range(n)]
    padre = list(range(n))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    if wcorr is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(wcorr[i, j]) >= threshold:
                    padre[raiz(i)] = raiz(j)
    banda = [""] * n
    for r in {raiz(i) for i in range(n)}:
        miembros = [i for i in range(n) if raiz(i) == r]
        suma = {b: sum(fr[i][b] for i in miembros) for b in bands}
        for i in miembros:
            banda[i] = max(suma, key=suma.get)
    return {b: [m for m in range(n) if banda[m] == b] for b in bands}


def spectral_purity(component: np.ndarray, bands: dict | None = None) -> dict[str, float]:
    """Fraccion de la potencia de un componente multicanal en cada banda.

    Suma los periodogramas de los canales pesados por su varianza, de modo que los
    canales con mas senal dominan. La pureza de un componente es la fraccion en su
    propia banda.

    Parameters
    ----------
    component : numpy.ndarray
        Componente ``(T, K)`` (recortar antes a los meses observados si la serie
        fue extendida).
    bands : dict, optional
        Bandas de periodo; por defecto :data:`DEFAULT_BANDS`.

    Returns
    -------
    dict
        Fraccion por banda (suman 1 si las bandas cubren todos los periodos).
    """
    bands = DEFAULT_BANDS if bands is None else bands
    x = np.asarray(component, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    tot = {b: 0.0 for b in bands}
    for c in range(x.shape[1]):
        w = x[:, c].var()
        if w == 0:
            continue
        for b, v in band_fractions(x[:, c], bands).items():
            tot[b] += w * v
    s = sum(tot.values())
    if s == 0:
        raise ValueError("El componente es identicamente nulo.")
    return {b: v / s for b, v in tot.items()}


def _ar_aic_fit(x: np.ndarray, p_max: int) -> np.ndarray:
    """Coeficientes AR(p) por minimos cuadrados, con p <= p_max elegido por AIC."""
    mejor = None
    for p in range(1, p_max + 1):
        X = np.column_stack([x[p - j - 1:len(x) - j - 1] for j in range(p)])
        y = x[p:]
        a, *_ = np.linalg.lstsq(X, y, rcond=None)
        s2 = float(((y - X @ a) ** 2).mean())
        aic = len(y) * np.log(max(s2, 1e-300)) + 2 * p
        if mejor is None or aic < mejor[0]:
            mejor = (aic, a)
    return mejor[1]


def extend_ar(Z: np.ndarray, M: int, p_max: int = 24) -> np.ndarray:
    """Extender cada canal ``M`` pasos hacia adelante y hacia atras con un AR(p).

    Por canal se resta la media, se ajusta un AR(p) por minimos cuadrados con
    ``p <= p_max`` elegido por AIC y se pronostica sin ruido ``M`` pasos; hacia el
    pasado se hace lo mismo sobre la serie invertida (con su propio ajuste). Se usa
    antes del EEOF para reducir el error de borde: el pronostico decae a la media
    en 1-2 anios, asi que en la practica es un anio de continuacion mas relleno
    climatologico. No se interpreta: se recorta despues de la descomposicion.

    Parameters
    ----------
    Z : numpy.ndarray
        Canales ``(T, C)``.
    M : int
        Pasos a agregar en cada extremo.
    p_max : int, optional
        Orden maximo del AR.

    Returns
    -------
    numpy.ndarray
        Serie extendida ``(T + 2M, C)``; las filas ``M:M+T`` son ``Z``.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z debe ser 2-D (tiempo, canal).")
    T, C = Z.shape
    if T <= 2 * p_max:
        raise ValueError("La serie es demasiado corta para p_max.")
    out = np.empty((T + 2 * M, C))
    for c in range(C):
        media = Z[:, c].mean()
        x = Z[:, c] - media
        lados = []
        for serie in (x, x[::-1]):
            a = _ar_aic_fit(serie, p_max)
            p = len(a)
            ext = list(serie[-p:])
            for _ in range(M):
                ext.append(float(np.dot(a, ext[-1:-p - 1:-1])))
            lados.append(np.array(ext[p:]))
        out[:, c] = np.r_[lados[1][::-1], x, lados[0]] + media
    return out
