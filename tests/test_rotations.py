"""Tests de las rotaciones varimax y promax."""

from __future__ import annotations

import numpy as np
import pytest

from enso_chile.eeof import eeof
from enso_chile.rotations import (
    bifactor,
    geomin,
    oblimin,
    promax,
    schmid_leiman,
    structured_promax,
    structured_varimax,
    target_rotation,
    varimax,
)

RNG = np.random.default_rng(42)


def _criterio_varimax(B: np.ndarray) -> float:
    """Suma sobre columnas de la varianza de las cargas al cuadrado."""
    return float(np.var(B ** 2, axis=0).sum())


def _estructura_simple() -> np.ndarray:
    """Cargas en bloques (cada variable carga en un solo factor), rotadas 30 grados."""
    A = np.zeros((8, 2))
    A[:4, 0] = [0.9, 0.8, 0.85, 0.7]
    A[4:, 1] = [0.75, 0.9, 0.8, 0.85]
    th = np.deg2rad(30)
    return A @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), A


def test_varimax_rotacion_ortogonal_y_comunalidades() -> None:
    A = RNG.normal(size=(20, 4))
    B, R = varimax(A)

    np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-10)
    np.testing.assert_allclose((B ** 2).sum(1), (A ** 2).sum(1), rtol=1e-10)
    np.testing.assert_allclose(B, A @ R, atol=1e-10)


def test_varimax_recupera_la_estructura_simple() -> None:
    """Deshace una rotacion de 30 grados de una estructura en bloques (salvo signo y orden)."""
    A_rot, A_true = _estructura_simple()
    B, _ = varimax(A_rot)
    sim = np.abs(B.T @ A_true) / np.outer(np.linalg.norm(B, axis=0), np.linalg.norm(A_true, axis=0))

    assert np.sort(sim.max(1)).min() > 0.999
    assert _criterio_varimax(B) >= _criterio_varimax(A_rot)


def test_promax_correlacion_entre_factores() -> None:
    A = RNG.normal(size=(15, 3))
    B, T, Phi = promax(A)

    np.testing.assert_allclose(np.diag(Phi), 1.0, atol=1e-10)
    np.testing.assert_allclose(Phi, Phi.T, atol=1e-10)
    np.testing.assert_allclose(B, A @ T, atol=1e-8)


def test_promax_en_estructura_ortogonal_da_factores_casi_incorrelacionados() -> None:
    A_rot, _ = _estructura_simple()
    _, _, Phi = promax(A_rot)
    assert abs(Phi[0, 1]) == pytest.approx(0.0, abs=0.05)


def _estructura_simple_oblicua(rho: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patron en bloques con tres factores correlacionados, mezclado por una rotacion."""
    P = np.zeros((12, 3))
    P[:4, 0] = [0.9, 0.8, 0.85, 0.7]
    P[4:8, 1] = [0.75, 0.9, 0.8, 0.85]
    P[8:, 2] = [0.8, 0.7, 0.9, 0.75]
    Phi = np.array([[1.0, rho, 0.2], [rho, 1.0, -0.3], [0.2, -0.3, 1.0]])
    Q = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    return P @ np.linalg.cholesky(Phi) @ Q, P, Phi


def _alinear(B: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Orden y signo de las columnas de ``B`` que mejor coinciden con ``P``."""
    sim = B.T @ P
    orden = np.abs(sim).argmax(0)
    return orden, np.sign(sim[orden, np.arange(P.shape[1])])


@pytest.mark.parametrize("rot", [oblimin, geomin])
def test_rotacion_oblicua_conserva_la_comunalidad_y_phi_es_una_correlacion(rot) -> None:
    A = RNG.normal(size=(20, 3))
    B, T, Phi = rot(A, n_starts=3)

    np.testing.assert_allclose(B, A @ T, atol=1e-8)
    np.testing.assert_allclose(np.diag(Phi), 1.0, atol=1e-10)
    np.testing.assert_allclose(Phi, Phi.T, atol=1e-10)
    np.testing.assert_allclose(B @ Phi @ B.T, A @ A.T, atol=1e-8)


@pytest.mark.parametrize("rot, tol", [(oblimin, 1e-3), (geomin, 0.03)])
def test_rotacion_oblicua_recupera_patron_y_correlaciones(rot, tol) -> None:
    """Con estructura simple exacta, quartimin (minimo 0) y geomin recuperan P y Phi."""
    A, P, Phi = _estructura_simple_oblicua()
    B, _, Phi_est = rot(A, normalize=False, tol=1e-9)
    orden, signo = _alinear(B, P)
    B_al = B[:, orden] * signo
    Phi_al = Phi_est[np.ix_(orden, orden)] * np.outer(signo, signo)

    np.testing.assert_allclose(B_al, P, atol=tol)
    np.testing.assert_allclose(Phi_al, Phi, atol=tol)


def _general_mas_grupos() -> np.ndarray:
    """Patron con un factor general y dos de grupo (superficie, fondo); filas distintas."""
    return np.array([[0.70, 0.55, 0.0], [0.60, 0.60, 0.0], [0.50, 0.45, 0.0], [0.65, 0.0, 0.0],
                     [0.55, 0.0, 0.0], [0.60, 0.0, 0.50], [0.50, 0.0, 0.55], [0.45, 0.0, 0.40]])


def test_bifactor_recupera_general_y_grupos_y_conserva_comunalidades() -> None:
    L = _general_mas_grupos()
    Q = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    A = L @ Q
    B, T = bifactor(A, tol=1e-10)

    np.testing.assert_allclose(T.T @ T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose((B ** 2).sum(1), (A ** 2).sum(1), rtol=1e-10)
    np.testing.assert_allclose(B[:, 0], L[:, 0], atol=1e-4)          # general en la columna 0
    grupos = B[:, 1:][:, np.argsort(-np.abs(B[0, 1:]))] * np.sign(B[[0, 5], 1:].sum(0))[np.argsort(-np.abs(B[0, 1:]))]
    np.testing.assert_allclose(np.abs(grupos), L[:, 1:], atol=1e-4)


def test_bifactor_oblicuo_conserva_la_comunalidad_y_mejora_el_criterio() -> None:
    from enso_chile.rotations import _biquartimin_criterion

    A = np.random.default_rng(42).normal(size=(20, 4))
    B, T, Phi = bifactor(A, oblique=True, n_starts=10, tol=1e-10)
    Bt, _ = bifactor(A, n_starts=10, tol=1e-10)

    np.testing.assert_allclose(B, A @ T, atol=1e-8)
    np.testing.assert_allclose(np.diag(Phi), 1.0, atol=1e-10)
    np.testing.assert_allclose(B @ Phi @ B.T, A @ A.T, atol=1e-8)
    assert _biquartimin_criterion(B)[0] <= _biquartimin_criterion(Bt)[0] + 1e-10


def test_bifactor_oblicuo_deja_el_general_incorrelacionado_con_los_grupos() -> None:
    """Teorema 1 de Jennrich & Bentler (2012): sin estructura bifactor exacta, phi_r1 = 0."""
    rng = np.random.default_rng(42)
    for _ in range(3):
        _, _, Phi = bifactor(rng.normal(size=(20, 4)), oblique=True, n_starts=10, tol=1e-10)
        np.testing.assert_allclose(Phi[0, 1:], 0.0, atol=1e-4)


def test_bifactor_exige_al_menos_dos_factores_de_grupo() -> None:
    with pytest.raises(ValueError):
        bifactor(RNG.normal(size=(6, 2)))


def test_schmid_leiman_recupera_segundo_orden_y_reproduce_la_covarianza() -> None:
    P = np.zeros((9, 3))
    P[:3, 0], P[3:6, 1], P[6:, 2] = [0.8, 0.7, 0.6], [0.75, 0.65, 0.7], [0.6, 0.8, 0.7]
    gamma = np.array([0.8, 0.6, 0.7])
    Phi = np.outer(gamma, gamma)
    np.fill_diagonal(Phi, 1.0)                                      # Phi con un solo factor exacto
    L, g = schmid_leiman(P, Phi)

    np.testing.assert_allclose(g, gamma, atol=1e-6)
    np.testing.assert_allclose(L @ L.T, P @ Phi @ P.T, atol=1e-8)
    np.testing.assert_allclose(L[:, 0], P @ gamma, atol=1e-6)


@pytest.mark.parametrize("oblique", [False, True])
def test_rotacion_objetivo_recupera_una_rotacion_conocida(oblique) -> None:
    A_rot, A_true = _estructura_simple()
    W = (A_true == 0).astype(float)                                 # solo se especifican los ceros
    out = target_rotation(A_rot, np.zeros_like(A_true), weights=W, oblique=oblique, tol=1e-10)
    B = out[0]
    sim = np.abs(B.T @ A_true) / np.outer(np.linalg.norm(B, axis=0), np.linalg.norm(A_true, axis=0))
    assert np.sort(sim.max(1)).min() > 0.999
    if oblique:
        assert abs(out[2][0, 1]) < 1e-3


def test_oblimin_en_estructura_ortogonal_da_factores_incorrelacionados() -> None:
    A_rot, A_true = _estructura_simple()
    B, _, Phi = oblimin(A_rot)
    assert abs(Phi[0, 1]) == pytest.approx(0.0, abs=1e-3)
    sim = np.abs(B.T @ A_true) / np.outer(np.linalg.norm(B, axis=0), np.linalg.norm(A_true, axis=0))
    assert np.sort(sim.max(1)).min() > 0.999



# --------------------------------------------------------------------------- #
# Varimax estructurado para M-SSA
# --------------------------------------------------------------------------- #


def test_varimax_estructurado_con_un_rezago_es_el_varimax_clasico() -> None:
    A = RNG.normal(size=(12, 3))
    B_s, R_s = structured_varimax(A, n_channels=12, n_lags=1)
    B_v, R_v = varimax(A)
    np.testing.assert_allclose(B_s, B_v, atol=1e-10)
    np.testing.assert_allclose(R_s, R_v, atol=1e-10)


def test_varimax_estructurado_es_ortogonal_y_conserva_participaciones() -> None:
    K, L, k = 4, 6, 3
    B = RNG.normal(size=(L * K, k))
    B_r, R = structured_varimax(B, n_channels=K, n_lags=L)
    np.testing.assert_allclose(R.T @ R, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(B_r, B @ R, atol=1e-10)
    part = lambda M: (M.reshape(L, K, k) ** 2).sum(axis=(0, 2))  # participacion total de cada canal
    np.testing.assert_allclose(part(B_r), part(B), rtol=1e-10)


def test_varimax_estructurado_localiza_oscilaciones_por_canal_sin_romper_pares() -> None:
    """Dos canales con oscilaciones propias de igual varianza: el M-SSA las mezcla; la rotacion
    estructurada deja cada componente concentrado en un canal."""
    t = np.arange(400)
    rng = np.random.default_rng(3)
    Z = np.column_stack([np.sin(2 * np.pi * t / 20), np.sin(2 * np.pi * t / 33 + 1)])  # una por canal
    Z = Z + 0.05 * rng.normal(size=Z.shape)
    L, K = 40, 2
    _, ee, pc = eeof(Z, L, n_modes=4)
    sv = np.sqrt((pc ** 2).sum(0))
    E = ee.transpose(0, 2, 1).reshape(4, -1).T  # (L*K, 4), orden por bloques de rezago
    B_r, R = structured_varimax(E * sv, n_channels=K, n_lags=L)
    part = (B_r.reshape(L, K, 4) ** 2).sum(0)
    share = part / part.sum(0)
    assert share.max(0).min() > 0.95  # cada componente rotada vive en un solo canal
    # los pares se conservan: las dos componentes de cada canal tienen la misma frecuencia
    canal = share.argmax(0)
    assert sorted(np.bincount(canal, minlength=2).tolist()) == [2, 2]


# --------------------------------------------------------------------------- #
# Promax estructurado
# --------------------------------------------------------------------------- #


def test_promax_estructurado_con_un_rezago_es_el_promax_clasico() -> None:
    A = RNG.normal(size=(12, 3))
    Bs, Ts, Ps = structured_promax(A, n_channels=12, n_lags=1)
    Bp, Tp, Pp = promax(A)
    np.testing.assert_allclose(Bs, Bp, atol=1e-8)
    np.testing.assert_allclose(Ts, Tp, atol=1e-8)
    np.testing.assert_allclose(Ps, Pp, atol=1e-8)


def test_promax_estructurado_conserva_la_reconstruccion_y_phi_es_una_correlacion() -> None:
    K, L, k = 4, 6, 3
    B = RNG.normal(size=(L * K, k))
    F = RNG.normal(size=(50, k))
    B_p, T, Phi = structured_promax(B, n_channels=K, n_lags=L)
    np.testing.assert_allclose(B_p, B @ T, atol=1e-10)
    np.testing.assert_allclose((F @ np.linalg.inv(T).T) @ B_p.T, F @ B.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(Phi), 1.0, atol=1e-10)
    np.testing.assert_allclose(Phi, Phi.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(Phi) > 0)


def test_promax_estructurado_localiza_y_no_rompe_pares_con_oscilaciones_independientes() -> None:
    """Como el test del varimax estructurado: oscilaciones independientes, una por canal. El
    promax estructurado debe localizarlas igual, conservar los pares y dar Phi casi diagonal."""
    t = np.arange(400)
    rng = np.random.default_rng(3)
    Z = np.column_stack([np.sin(2 * np.pi * t / 20), np.sin(2 * np.pi * t / 33 + 1)])
    Z = Z + 0.05 * rng.normal(size=Z.shape)
    L, K = 40, 2
    _, ee, pc = eeof(Z, L, n_modes=4)
    sv = np.sqrt((pc ** 2).sum(0))
    E = ee.transpose(0, 2, 1).reshape(4, -1).T
    B_p, T, Phi = structured_promax(E * sv, n_channels=K, n_lags=L)
    part = (B_p.reshape(L, K, 4) ** 2).sum(0)
    share = part / part.sum(0)
    assert share.max(0).min() > 0.95
    canal = share.argmax(0)
    assert sorted(np.bincount(canal, minlength=2).tolist()) == [2, 2]
    otro = canal[:, None] != canal[None, :]
    assert np.abs(Phi[otro]).max() < 0.1  # componentes de canales distintos casi incorrelacionados
