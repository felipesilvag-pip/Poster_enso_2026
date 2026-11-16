"""Rotaciones de cargas: orthomax, bifactor y objetivo (ortogonales); promax, oblimin, geomin (oblicuas).

Se rotan **cargas** (autovectores escalados por la raiz del autovalor), no
autovectores unitarios. Con cargas ``A`` (p x k) y puntajes estandarizados ``F``
(T x k) tales que ``X ~ F A^T``:

* varimax: ``B = A R`` con ``R`` ortogonal y puntajes ``F R``;
* promax: ``B = A T`` con ``T`` no ortogonal y puntajes ``F (T^{-1})^T``; los
  factores quedan correlacionados (``Phi``).

En promax la suma de cargas al cuadrado de un factor no es su varianza explicada.
Referencias: Kaiser (1958), Hendrickson & White (1964), Jennrich & Sampson (1966),
Yates (1987), Richman (1986), Jennrich (2001, 2002), Bernaards & Jennrich (2005),
Schmid & Leiman (1957), Browne (2001), Jennrich & Bentler (2011).
"""

# LEGACY: Entendiendo_EOF.py, varimax_rotation (l. 274), varimax (l. 2469) y
# promax (l. 2492). El varimax estructurado para M-SSA no tiene antecedente en los scripts viejos.

from __future__ import annotations

import numpy as np

__all__ = ["varimax", "promax", "oblimin", "geomin", "bifactor", "schmid_leiman", "target_rotation",
           "structured_varimax", "structured_promax"]


def varimax(A: np.ndarray, gamma: float = 1.0, max_sweeps: int = 1000, tol: float = 1e-10,
            normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Rotacion varimax por rotaciones planas sucesivas (algoritmo original de Kaiser).

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)``.
    gamma : float, optional
        1 es varimax; 0 es quartimax (familia orthomax).
    max_sweeps : int, optional
        Barridos maximos sobre todos los pares de columnas.
    tol : float, optional
        Angulo (radianes) por debajo del cual se considera convergido.
    normalize : bool, optional
        Normalizacion de Kaiser: filas divididas por su comunalidad antes de rotar.

    Returns
    -------
    B : numpy.ndarray
        Cargas rotadas ``A R``.
    R : numpy.ndarray
        Matriz de rotacion ortogonal ``(k, k)``.

    Notes
    -----
    Maximiza la suma sobre columnas de la varianza de las cargas al cuadrado. Para
    cada par de columnas ``(x, y)`` el angulo optimo es cerrado (Kaiser 1958):
    ``tan(4 phi) = (D - 2 gamma A B / p) / (C - gamma (A^2 - B^2) / p)``, con
    ``u = x^2 - y^2``, ``v = 2 x y``, ``A = sum u``, ``B = sum v``,
    ``C = sum(u^2 - v^2)``, ``D = 2 sum(u v)``. Cada paso no disminuye el criterio.
    Se prefiere al algoritmo de punto fijo con SVD porque este puede oscilar entre
    dos rotaciones en configuraciones simetricas. Las comunalidades no cambian.
    """
    A = np.asarray(A, dtype=float)
    p, k = A.shape
    h = np.sqrt((A ** 2).sum(1)) if normalize else np.ones(p)
    h = np.where(h == 0, 1.0, h)
    B = A / h[:, None]
    R = np.eye(k)
    for _ in range(max_sweeps):
        max_phi = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                x, y = B[:, i].copy(), B[:, j].copy()
                u, v = x ** 2 - y ** 2, 2 * x * y
                a_, b_ = u.sum(), v.sum()
                num = 2 * (u * v).sum() - 2 * gamma * a_ * b_ / p
                den = (u ** 2 - v ** 2).sum() - gamma * (a_ ** 2 - b_ ** 2) / p
                phi = 0.25 * np.arctan2(num, den)
                if abs(phi) < tol:
                    continue
                c, s = np.cos(phi), np.sin(phi)
                B[:, i], B[:, j] = c * x + s * y, -s * x + c * y
                ri, rj = R[:, i].copy(), R[:, j].copy()
                R[:, i], R[:, j] = c * ri + s * rj, -s * ri + c * rj
                max_phi = max(max_phi, abs(phi))
        if max_phi < tol:
            break
    return B * h[:, None], R


def promax(A: np.ndarray, power: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotacion promax, con la misma construccion que ``stats::promax`` de R.

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)``.
    power : int, optional
        Exponente del objetivo (``m``); 4 es lo usual.

    Returns
    -------
    B : numpy.ndarray
        Cargas de patron oblicuas ``A T``.
    T : numpy.ndarray
        Transformacion total desde ``A``.
    Phi : numpy.ndarray
        Correlacion entre factores, ``(U^T U)^{-1}``.

    Notes
    -----
    Parte de varimax, construye ``Q = B_v |B_v|^(power - 1)``, ajusta ``U`` por
    minimos cuadrados (``B_v U ~ Q``) y escala sus columnas para que ``Phi`` tenga
    diagonal 1.
    """
    Bv, R = varimax(A)
    Q = Bv * np.abs(Bv) ** (power - 1)
    U = np.linalg.lstsq(Bv, Q, rcond=None)[0]
    U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))[None, :]
    Phi = np.linalg.inv(U.T @ U)
    return Bv @ U, R @ U, Phi


def _oblimin_criterion(L: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    """Criterio oblimin directo y su gradiente respecto de ``L`` (como ``GPArotation``)."""
    p, k = L.shape
    X = (L ** 2) @ (np.ones((k, k)) - np.eye(k))
    if delta != 0:
        X = X - (delta / p) * X.sum(0, keepdims=True)
    return float((L ** 2 * X).sum() / 4), L * X


def _geomin_criterion(L: np.ndarray, eps: float) -> tuple[float, np.ndarray]:
    """Criterio geomin y su gradiente respecto de ``L`` (como ``GPArotation``)."""
    k = L.shape[1]
    L2 = L ** 2 + eps
    pro = np.exp(np.log(L2).sum(1) / k)
    return float(pro.sum()), (2.0 / k) * (L / L2) * pro[:, None]


def _gpa_oblique(A: np.ndarray, criterion, T0: np.ndarray, max_iter: int,
                 tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Proyeccion del gradiente para rotaciones oblicuas (Jennrich 2002).

    Minimiza ``criterion(L)`` con ``L = A (T^T)^{-1}`` y columnas de ``T`` de
    norma 1. Devuelve ``L``, ``T`` y el valor del criterio.
    """
    T = T0.copy()
    L = A @ np.linalg.inv(T).T
    f, Gq = criterion(L)
    G = -(L.T @ Gq @ np.linalg.inv(T)).T
    alpha = 1.0
    for _ in range(max_iter):
        Gp = G - T * (T * G).sum(0)[None, :]
        s = np.linalg.norm(Gp)
        if s < tol:
            break
        alpha *= 2.0
        for _ in range(20):
            X = T - alpha * Gp
            Tt = X / np.linalg.norm(X, axis=0)[None, :]
            Lt = A @ np.linalg.inv(Tt).T
            ft, Gqt = criterion(Lt)
            if ft < f - 0.5 * s ** 2 * alpha:
                break
            alpha /= 2.0
        T, L, f, Gq = Tt, Lt, ft, Gqt
        G = -(L.T @ Gq @ np.linalg.inv(T)).T
    return L, T, f


def _oblique_rotation(A: np.ndarray, criterion, normalize: bool, n_starts: int,
                      random_state: int, max_iter: int,
                      tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotacion oblicua por GPA con varios arranques; convencion de :func:`promax`."""
    A = np.asarray(A, dtype=float)
    p, k = A.shape
    h = np.sqrt((A ** 2).sum(1)) if normalize else np.ones(p)
    h = np.where(h == 0, 1.0, h)
    An = A / h[:, None]
    rng = np.random.default_rng(random_state)
    starts = [np.eye(k)] + [np.linalg.qr(rng.normal(size=(k, k)))[0] for _ in range(n_starts - 1)]
    best = None
    for T0 in starts:
        L, Tg, f = _gpa_oblique(An, criterion, T0, max_iter, tol)
        if best is None or f < best[2]:
            best = (L, Tg, f)
    L, Tg, _ = best
    return L * h[:, None], np.linalg.inv(Tg).T, Tg.T @ Tg


def oblimin(A: np.ndarray, delta: float = 0.0, normalize: bool = True, n_starts: int = 10,
            random_state: int = 0, max_iter: int = 1000,
            tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotacion oblimin directa (Jennrich & Sampson 1966); ``delta = 0`` es quartimin.

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)``.
    delta : float, optional
        Parametro de oblicuidad; valores negativos acercan los factores a la
        ortogonalidad. 0 (quartimin) es lo usual.
    normalize : bool, optional
        Normalizacion de Kaiser (filas por su comunalidad), igual que :func:`varimax`.
    n_starts : int, optional
        Arranques (identidad mas ``n_starts - 1`` rotaciones aleatorias); se
        conserva el de menor criterio.
    random_state : int, optional
        Semilla de los arranques aleatorios.
    max_iter, tol : optional
        Iteraciones maximas y tolerancia sobre la norma del gradiente proyectado.

    Returns
    -------
    B : numpy.ndarray
        Cargas de patron oblicuas ``A T``.
    T : numpy.ndarray
        Transformacion desde ``A``; los puntajes rotados son ``F (T^{-1})^T``.
    Phi : numpy.ndarray
        Correlacion entre factores, con diagonal 1.

    Notes
    -----
    Minimiza ``sum_{j<l} [sum_i b_ij^2 b_il^2 - (delta/p) sum_i b_ij^2 sum_i b_il^2]``
    por proyeccion del gradiente (Jennrich 2002; Bernaards & Jennrich 2005), con
    el criterio y el gradiente de ``GPArotation::vgQ.oblimin``. Cumple
    ``B Phi B^T = A A^T``.
    """
    return _oblique_rotation(A, lambda L: _oblimin_criterion(L, delta), normalize,
                             n_starts, random_state, max_iter, tol)


def geomin(A: np.ndarray, eps: float = 0.01, normalize: bool = True, n_starts: int = 10,
           random_state: int = 0, max_iter: int = 1000,
           tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotacion geomin oblicua (Yates 1987).

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)``.
    eps : float, optional
        Constante que evita el logaritmo de cero; 0.01 como ``GPArotation``.
    normalize, n_starts, random_state, max_iter, tol : optional
        Como en :func:`oblimin`. Geomin tiene minimos locales: conviene mas de
        un arranque.

    Returns
    -------
    B, T, Phi : numpy.ndarray
        Como en :func:`oblimin`.

    Notes
    -----
    Minimiza ``sum_i (prod_j (b_ij^2 + eps))^(1/k)``, que es pequeno cuando cada
    fila tiene al menos una carga cercana a cero.
    """
    return _oblique_rotation(A, lambda L: _geomin_criterion(L, eps), normalize,
                             n_starts, random_state, max_iter, tol)


def _gpa_orthogonal(A: np.ndarray, criterion, T0: np.ndarray, max_iter: int,
                    tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Proyeccion del gradiente para rotaciones ortogonales (Jennrich 2001).

    Minimiza ``criterion(L)`` con ``L = A T`` y ``T`` ortogonal; la proyeccion
    sobre las matrices ortogonales se hace con la SVD (factor polar).
    """
    T = T0.copy()
    L = A @ T
    f, Gq = criterion(L)
    G = A.T @ Gq
    alpha = 1.0
    for _ in range(max_iter):
        M = T.T @ G
        Gp = G - T @ ((M + M.T) / 2)
        s = np.linalg.norm(Gp)
        if s < tol:
            break
        alpha *= 2.0
        for _ in range(30):
            U, _, Vt = np.linalg.svd(T - alpha * Gp)
            Tt = U @ Vt
            Lt = A @ Tt
            ft, Gqt = criterion(Lt)
            if ft < f - 0.5 * s ** 2 * alpha:
                break
            alpha /= 2.0
        T, L, f, Gq = Tt, Lt, ft, Gqt
        G = A.T @ Gq
    return L, T, f


def _biquartimin_criterion(L: np.ndarray) -> tuple[float, np.ndarray]:
    """Bi-quartimin: quartimin solo entre las columnas de grupo (``GPArotation::vgQ.bifactor``)."""
    Lg = L[:, 1:]
    Lg2 = Lg ** 2
    N = np.ones((Lg.shape[1],) * 2) - np.eye(Lg.shape[1])
    Gq = np.zeros_like(L)
    Gq[:, 1:] = 4 * Lg * (Lg2 @ N)
    return float((Lg2 * (Lg2 @ N)).sum()), Gq


def bifactor(A: np.ndarray, oblique: bool = False, normalize: bool = False, n_starts: int = 20,
             random_state: int = 0, max_iter: int = 2000, tol: float = 1e-8):
    """Rotacion bifactor exploratoria con el criterio bi-quartimin (Jennrich & Bentler 2011, 2012).

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)`` con ``k >= 3`` (un factor general y al menos dos de grupo).
    oblique : bool, optional
        ``False``: rotacion ortogonal (``GPArotation::bifactorT``). ``True``: oblicua
        (``bifactorQ``; Jennrich & Bentler 2012): los grupos pueden correlacionarse.
    normalize : bool, optional
        Normalizacion de Kaiser. Por defecto ``False``, como ``GPArotation``.
    n_starts : int, optional
        Arranques (identidad mas rotaciones aleatorias); el criterio tiene minimos
        locales y ``T = I`` es punto estacionario cuando ``A`` viene de un EOF.
    random_state, max_iter, tol : optional
        Semilla, iteraciones maximas y tolerancia del gradiente proyectado.

    Returns
    -------
    B : numpy.ndarray
        Cargas (de patron, si es oblicua) ``A T``; la columna 0 es el factor general.
    T : numpy.ndarray
        Transformacion ``(k, k)``; ortogonal, o con puntajes ``F (T^{-1})^T`` si es oblicua.
    Phi : numpy.ndarray
        Solo si ``oblique=True``: correlacion entre factores, con diagonal 1.

    Notes
    -----
    El criterio es quartimin sobre las columnas 1..k-1, ``sum_i sum_{j != l >= 1}
    b_ij^2 b_il^2``; la columna 0 no entra y por eso es el factor general. Con un
    solo grupo el criterio es identicamente cero (se exige ``k >= 3``). En el caso
    oblicuo, si existe una rotacion con estructura bifactor exacta la solucion no
    es unica (Jennrich & Bentler 2012, teorema 2); con datos reales el general
    resulta incorrelacionado con los grupos (su teorema 1).
    """
    A = np.asarray(A, dtype=float)
    p, k = A.shape
    if k < 3:
        raise ValueError("bifactor necesita k >= 3 (un general y al menos dos de grupo).")
    if oblique:
        B, T, Phi = _oblique_rotation(A, _biquartimin_criterion, normalize, n_starts,
                                      random_state, max_iter, tol)
        s = np.where(B.sum(0) < 0, -1.0, 1.0)
        return B * s, T * s, Phi * np.outer(s, s)
    h = np.sqrt((A ** 2).sum(1)) if normalize else np.ones(p)
    h = np.where(h == 0, 1.0, h)
    An = A / h[:, None]
    rng = np.random.default_rng(random_state)
    starts = [np.eye(k)] + [np.linalg.qr(rng.normal(size=(k, k)))[0] for _ in range(n_starts - 1)]
    best = None
    for T0 in starts:
        L, T, f = _gpa_orthogonal(An, _biquartimin_criterion, T0, max_iter, tol)
        if best is None or f < best[2]:
            best = (L, T, f)
    T = best[1]
    T = T * np.where((A @ T).sum(0) < 0, -1.0, 1.0)
    return A @ T, T


def schmid_leiman(B: np.ndarray, Phi: np.ndarray, max_iter: int = 500,
                  tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Transformacion de Schmid & Leiman (1957) desde una solucion oblicua.

    Parameters
    ----------
    B : numpy.ndarray
        Cargas de patron oblicuas ``(p, k)``, ``k >= 3``.
    Phi : numpy.ndarray
        Correlacion entre factores ``(k, k)``.
    max_iter, tol : optional
        Iteraciones del factor principal de un factor sobre ``Phi``.

    Returns
    -------
    L : numpy.ndarray
        Cargas ``(p, k + 1)``: columna 0 el factor general (segundo orden) y
        columnas 1..k los factores de grupo residualizados, todos ortogonales.
    gamma : numpy.ndarray
        Cargas de segundo orden de cada factor oblicuo en el general.

    Notes
    -----
    Se ajusta un factor a ``Phi`` por factores principales iterados,
    ``Phi ~ gamma gamma^T + diag(1 - gamma^2)``; luego ``L = [B gamma,
    B diag(sqrt(1 - gamma^2))]``. Si ``Phi`` tiene exactamente un factor,
    ``L L^T = B Phi B^T``. La carga general de cada fila es proporcional a sus
    cargas de grupo (restriccion de proporcionalidad, Jennrich & Bentler 2011).
    """
    B = np.asarray(B, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    k = Phi.shape[0]
    if k < 3:
        raise ValueError("Schmid-Leiman necesita al menos 3 factores de primer orden.")
    h2 = 1 - 1 / np.diag(np.linalg.inv(Phi))   # comunalidades iniciales (SMC)
    for _ in range(max_iter):
        R = Phi.copy()
        np.fill_diagonal(R, h2)
        lam, V = np.linalg.eigh(R)
        gamma = V[:, -1] * np.sqrt(max(lam[-1], 0.0))
        nuevo = np.clip(gamma ** 2, 0.0, 1.0)
        if np.abs(nuevo - h2).max() < tol:
            break
        h2 = nuevo
    gamma = gamma * (1.0 if gamma.sum() >= 0 else -1.0)
    gamma = np.clip(gamma, -1.0, 1.0)
    L = np.column_stack([B @ gamma, B * np.sqrt(1 - gamma ** 2)[None, :]])
    return L, gamma


def target_rotation(A: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None,
                    oblique: bool = False, n_starts: int = 10, random_state: int = 0,
                    max_iter: int = 2000, tol: float = 1e-8):
    """Rotacion hacia una matriz objetivo parcialmente especificada (Browne 2001).

    Parameters
    ----------
    A : numpy.ndarray
        Cargas ``(p, k)``.
    target : numpy.ndarray
        Objetivo ``(p, k)``; solo cuentan las celdas con peso 1.
    weights : numpy.ndarray or None, optional
        Mascara ``(p, k)`` de 0/1 con las celdas especificadas (por ejemplo los
        ceros esperados). ``None`` especifica todas.
    oblique : bool, optional
        ``False``: ``T`` ortogonal. ``True``: oblicua, como :func:`oblimin`.
    n_starts, random_state, max_iter, tol : optional
        Como en :func:`oblimin`.

    Returns
    -------
    B, T : numpy.ndarray
        Si es ortogonal.
    B, T, Phi : numpy.ndarray
        Si es oblicua (misma convencion que :func:`promax`).

    Notes
    -----
    Minimiza ``|| W o (B - H) ||_F^2`` (``GPArotation::vgQ.pst``).
    """
    A = np.asarray(A, dtype=float)
    H = np.asarray(target, dtype=float)
    W = np.ones_like(H) if weights is None else np.asarray(weights, dtype=float)

    def crit(L: np.ndarray) -> tuple[float, np.ndarray]:
        D = W * (L - H)
        return float((D ** 2).sum()), 2 * D

    if oblique:
        return _oblique_rotation(A, crit, False, n_starts, random_state, max_iter, tol)
    k = A.shape[1]
    rng = np.random.default_rng(random_state)
    starts = [np.eye(k)] + [np.linalg.qr(rng.normal(size=(k, k)))[0] for _ in range(n_starts - 1)]
    best = min((_gpa_orthogonal(A, crit, T0, max_iter, tol) for T0 in starts), key=lambda r: r[2])
    return A @ best[1], best[1]


def structured_varimax(B: np.ndarray, n_channels: int, n_lags: int, gamma: float = 1.0,
                       normalize: bool = True, max_sweeps: int = 1000,
                       tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Varimax para M-SSA (Groth & Ghil 2011; Groth & Ghil 2015, apendice A).

    El varimax clasico aplicado a las ST-EOF maximiza tambien la varianza en los rezagos y
    rompe los pares oscilatorios. Esta version primero suma sobre los rezagos las cargas al
    cuadrado de cada canal (indice de participacion ``e_dk^2 = sum_m e_dk(m)^2``) y maximiza la
    varianza, entre canales, de esa participacion: solo simplifica la estructura espacial.

    Parameters
    ----------
    B : numpy.ndarray
        ST-EOF escaladas por sus valores singulares (``E Sigma``), forma
        ``(n_lags * n_channels, k)`` en el orden por bloques de rezago (fila ``l * K + d``),
        como las columnas de ``Vt.T * s`` de la matriz de trayectoria.
    n_channels, n_lags : int
        Canales ``K`` y rezagos ``L``.
    gamma : float, optional
        1 es varimax.
    normalize : bool, optional
        Normalizar cada canal por su participacion total (analogo a Kaiser).
    max_sweeps : int, optional
        Barridos maximos sobre todos los pares.
    tol : float, optional
        Angulo (radianes) bajo el cual se considera convergido.

    Returns
    -------
    B_rot : numpy.ndarray
        Cargas rotadas ``B R``.
    R : numpy.ndarray
        Rotacion ortogonal ``(k, k)``. Los EEOF rotados son ``E R``, las ST-PC ``A R`` y los
        autovalores ``diag(R' Lambda R)``.

    Notes
    -----
    Con ``u = A_i - A_j`` y ``v = 2 C_ij`` (``A`` = participacion por canal, ``C`` = producto
    cruzado sumado sobre rezagos), la participacion rotada tiene la misma forma que una carga al
    cuadrado en el varimax clasico, asi que el angulo optimo de cada par sale de la formula de
    Kaiser. Con ``n_lags = 1`` se reduce exactamente a :func:`varimax`.
    """
    B = np.asarray(B, dtype=float)
    p, k = B.shape
    if p != n_lags * n_channels:
        raise ValueError(f"B tiene {p} filas; se esperaban n_lags * n_channels = {n_lags * n_channels}.")
    B3 = B.reshape(n_lags, n_channels, k).copy()
    h2 = (B3 ** 2).sum(axis=(0, 2)) if normalize else np.ones(n_channels)
    h2 = np.where(h2 == 0, 1.0, h2)
    R = np.eye(k)
    for _ in range(max_sweeps):
        max_phi = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                bi, bj = B3[:, :, i].copy(), B3[:, :, j].copy()
                u = ((bi ** 2).sum(0) - (bj ** 2).sum(0)) / h2
                v = 2 * (bi * bj).sum(0) / h2
                a_, b_ = u.sum(), v.sum()
                num = 2 * (u * v).sum() - 2 * gamma * a_ * b_ / n_channels
                den = (u ** 2 - v ** 2).sum() - gamma * (a_ ** 2 - b_ ** 2) / n_channels
                phi = 0.25 * np.arctan2(num, den)
                if abs(phi) < tol:
                    continue
                c, s = np.cos(phi), np.sin(phi)
                B3[:, :, i], B3[:, :, j] = c * bi + s * bj, -s * bi + c * bj
                ri, rj = R[:, i].copy(), R[:, j].copy()
                R[:, i], R[:, j] = c * ri + s * rj, -s * ri + c * rj
                max_phi = max(max_phi, abs(phi))
        if max_phi < tol:
            break
    return B3.reshape(p, k), R


def structured_promax(B: np.ndarray, n_channels: int, n_lags: int, power: float = 4.0,
                      normalize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promax estructurado para M-SSA (construccion propia; sin referencia publicada conocida).

    Analogo oblicuo de :func:`structured_varimax`. El promax clasico afila cada carga
    (``q = b |b|^(m-1)``); aplicado entrada por entrada a las ST-EOF afilaria tambien la
    estructura de rezagos y romperia los pares oscilatorios. Aqui el objetivo escala **todos
    los rezagos de un canal por el mismo factor**, que depende solo de la participacion del
    canal ``e_dk^2 = sum_m b_dk(m)^2``:

    ``q_dk(m) = b_dk(m) * (e_dk^2)^((m - 1) / 2)``,

    de modo que la participacion del objetivo es ``(e_dk^2)^m`` (el promax clasico aplicado a
    las participaciones) y la forma temporal dentro de cada canal se conserva.

    Parameters
    ----------
    B : numpy.ndarray
        ST-EOF escaladas por sus valores singulares, ``(n_lags * n_channels, k)``, en el orden
        por bloques de rezago (fila ``l * K + d``), como en :func:`structured_varimax`.
    n_channels, n_lags : int
        Canales ``K`` y rezagos ``L``.
    power : float, optional
        Exponente ``m`` del objetivo (4 es lo usual; 2 es mas suave).
    normalize : bool, optional
        Normalizacion por canal del varimax estructurado de partida.

    Returns
    -------
    B_p : numpy.ndarray
        Patrones oblicuos ``B T``.
    T : numpy.ndarray
        Transformacion total desde ``B`` (``R U``). ST-EOF rotadas ``E T`` (con ``B = E Sigma``,
        ``E Sigma T``), ST-PC estandarizadas ``F (T^{-1})^T``.
    Phi : numpy.ndarray
        Correlacion entre componentes, ``(U^T U)^{-1}``, con diagonal 1.

    Notes
    -----
    Con ``n_lags = 1`` se reduce exactamente a :func:`promax`. La reconstruccion total no
    cambia: ``F B^T = [F (T^{-1})^T] [B T]^T``. Al ser oblicuo, las varianzas de los
    componentes no suman la del subespacio y su W-correlacion no es nula.
    """
    B = np.asarray(B, dtype=float)
    p, k = B.shape
    if p != n_lags * n_channels:
        raise ValueError(f"B tiene {p} filas; se esperaban n_lags * n_channels = {n_lags * n_channels}.")
    Bv, R = structured_varimax(B, n_channels, n_lags, normalize=normalize)
    B3 = Bv.reshape(n_lags, n_channels, k)
    e2 = (B3 ** 2).sum(axis=0)                                   # participacion (K, k)
    Q = (B3 * (e2[None, :, :] ** ((power - 1) / 2))).reshape(p, k)
    U = np.linalg.lstsq(Bv, Q, rcond=None)[0]
    U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))[None, :]
    Phi = np.linalg.inv(U.T @ U)
    return Bv @ U, R @ U, Phi
