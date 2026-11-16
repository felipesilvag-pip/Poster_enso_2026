"""Tests de EEOF / M-SSA: matriz de trayectoria, SVD, reconstruccion y W-correlacion."""

from __future__ import annotations

import numpy as np
import pytest

from enso_chile.eof import combined_blocks, eof_decomposition
from enso_chile.eeof import (
    band_fractions,
    derivssa,
    group_by_band,
    spectral_purity,
    diagonal_average,
    iterative_ossa,
    match_groups_by_subspace,
    eeof,
    extend_ar,
    find_pairs,
    find_pairs_north,
    pair_lagged_corr,
    reconstruct_components,
    reconstruction_weights,
    trajectory_matrix,
    wcorr_matrix,
)

RNG = np.random.default_rng(42)


def _seno(T: int = 240, periodo: float = 12.0, fase: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * np.arange(T) / periodo + fase)


# --------------------------------------------------------------------------- #
# Matriz de trayectoria
# --------------------------------------------------------------------------- #


def test_trayectoria_forma_y_estructura_hankel() -> None:
    """X[i, l*K + k] = Z[i + l, k] en todas las posiciones."""
    T, K, L = 30, 3, 7
    Z = RNG.normal(size=(T, K))
    X = trajectory_matrix(Z, L)

    assert X.shape == (T - L + 1, L * K)
    i, l, k = np.meshgrid(np.arange(T - L + 1), np.arange(L), np.arange(K), indexing="ij")
    assert np.array_equal(X[i, l * K + k], Z[i + l, k])


def test_trayectoria_con_L_1_es_la_misma_matriz() -> None:
    Z = RNG.normal(size=(20, 4))
    assert np.array_equal(trajectory_matrix(Z, 1), Z)


@pytest.mark.parametrize("L", [0, 31])
def test_trayectoria_L_fuera_de_rango(L: int) -> None:
    with pytest.raises(ValueError):
        trajectory_matrix(RNG.normal(size=(30, 2)), L)


# --------------------------------------------------------------------------- #
# Descomposicion y reconstruccion
# --------------------------------------------------------------------------- #


def test_reconstruccion_exacta_con_todos_los_modos() -> None:
    T, K, L = 60, 3, 12
    Z = RNG.normal(size=(T, K)) + 5.0
    rango = min(T - L + 1, L * K)
    _, ee, pc = eeof(Z, L, n_modes=rango)
    rec = reconstruct_components(ee, pc, list(range(rango)), T)

    np.testing.assert_allclose(rec, Z - Z.mean(0), atol=1e-10)


def test_st_pcs_ortogonales_y_eeof_unitarias() -> None:
    _, ee, pc = eeof(RNG.normal(size=(80, 3)), 10, n_modes=6)
    G = pc.T @ pc

    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-8)
    np.testing.assert_allclose((ee ** 2).sum(axis=(1, 2)), 1.0, atol=1e-12)


def test_autovalores_descendentes_y_suma_de_varianza() -> None:
    T, L = 50, 8
    Z = RNG.normal(size=(T, 2))
    rango = min(T - L + 1, L * 2)
    lam, _, _ = eeof(Z, L, n_modes=rango)
    X = trajectory_matrix(Z - Z.mean(0), L)

    assert np.all(np.diff(lam) <= 1e-12)
    assert lam.sum() == pytest.approx((X ** 2).sum() / X.shape[0])


def test_pesos_de_reconstruccion() -> None:
    """1, 2, ..., L*, ..., 2, 1 y suman N * L."""
    T, L = 20, 5
    w = reconstruction_weights(T, L)

    assert list(w[:6]) == [1, 2, 3, 4, 5, 5]
    assert list(w[-5:]) == [5, 4, 3, 2, 1]
    assert w.sum() == (T - L + 1) * L


# --------------------------------------------------------------------------- #
# W-correlacion y pares
# --------------------------------------------------------------------------- #


def _wcorr_literal(F1: np.ndarray, F2: np.ndarray, L: int) -> float:
    """Definicion de Golyandina et al. (2001), escrita sin vectorizar."""
    T = F1.shape[0]
    N = T - L + 1
    w = np.array([min(i + 1, min(L, N), max(L, N), T - i) for i in range(T)], dtype=float)
    ip = lambda a, b: float(np.sum(w[:, None] * a * b))
    return ip(F1, F2) / np.sqrt(ip(F1, F1) * ip(F2, F2))


def test_wcorr_coincide_con_la_definicion() -> None:
    T, L = 70, 15
    rcs = RNG.normal(size=(4, T, 2))
    W = wcorr_matrix(rcs, L)

    for i in range(4):
        for j in range(4):
            assert W[i, j] == pytest.approx(_wcorr_literal(rcs[i], rcs[j], L))
    np.testing.assert_allclose(np.diag(W), 1.0)
    np.testing.assert_allclose(W, W.T)


def test_wcorr_no_es_pearson() -> None:
    """No centra: dos series con la misma media no nula y ruido distinto tienen W-corr alta."""
    T, L = 100, 20
    a = 3.0 + RNG.normal(size=(T, 1))
    b = 3.0 + RNG.normal(size=(T, 1))
    W = wcorr_matrix(np.stack([a, b]), L)

    assert abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]) < 0.3
    assert W[0, 1] > 0.8


def test_seno_puro_da_un_par_aislado_en_cuadratura() -> None:
    """Un seno de 12 meses produce un par (0, 1) con |W-corr| ~ 1 y ST-PC en cuadratura."""
    T, L, P = 240, 36, 12
    Z = (_seno(T, P) + 0.3 * _seno(T, 60.0))[:, None] + 0.05 * RNG.normal(size=(T, 1))
    lam, ee, pc = eeof(Z, L, n_modes=6)
    rcs = np.stack([reconstruct_components(ee, pc, [m], T) for m in range(6)])
    W = wcorr_matrix(rcs, L)

    assert (0, 1) in find_pairs(W)
    assert abs(W[0, 1]) > 0.95
    lags, c = pair_lagged_corr(pc, 0, 1, max_lag=P)
    assert abs(c[len(c) // 2]) < 0.1
    assert abs(abs(lags[np.argmax(np.abs(c))]) - P / 4) <= 1


def test_find_pairs_north() -> None:
    lam = np.array([10.0, 5.0, 4.9, 1.0, 0.9, 0.85])
    assert find_pairs_north(lam, n_eff=1000) == [(1, 2)]


# --------------------------------------------------------------------------- #
# Propiedades adicionales (L = 1, rezagos, aditividad, combinado)
# --------------------------------------------------------------------------- #


def test_eeof_con_L_1_es_el_eof_clasico() -> None:
    """Con un solo rezago el EEOF es el EOF: mismos patrones y autovalores (salvo N vs T-1)."""
    T = 50
    Z = RNG.normal(size=(T, 4))
    lam, ee, _ = eeof(Z, 1, n_modes=4)
    r = eof_decomposition(Z)

    np.testing.assert_allclose(lam * T / (T - 1), r.eigenvalues[:4], rtol=1e-10)
    np.testing.assert_allclose(np.abs(ee[:, :, 0]), np.abs(r.eofs[:4]), atol=1e-10)


def test_canal_retrasado_aparece_como_rezago_en_la_st_eof() -> None:
    """Si el canal 2 es el canal 1 retrasado d meses, su ST-EOF es la del canal 1 desplazada d."""
    T, L, d = 400, 24, 4
    rng = np.random.default_rng(1)
    x = np.zeros(T + d)
    e = rng.normal(size=T + d)
    for t in range(1, T + d):
        x[t] = 0.9 * x[t - 1] + e[t]
    Z = np.column_stack([x[d:], x[:-d]])  # z2[t] = z1[t - d]
    _, ee, _ = eeof(Z, L, n_modes=1)
    v1, v2 = ee[0, 0, :L - d], ee[0, 1, d:]

    assert abs(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) > 0.95


def test_reconstruccion_es_aditiva_por_grupos() -> None:
    T, L = 80, 10
    Z = RNG.normal(size=(T, 3))
    _, ee, pc = eeof(Z, L, n_modes=8)
    juntos = reconstruct_components(ee, pc, [0, 1, 2, 3], T)
    por_partes = reconstruct_components(ee, pc, [0, 1], T) + reconstruct_components(ee, pc, [2, 3], T)

    np.testing.assert_allclose(juntos, por_partes, atol=1e-12)


def test_eeof_combinado_invariante_a_la_escala_de_una_capa() -> None:
    """Con bloques normalizados, escalar una capa no cambia el EEOF combinado."""
    capa_a, capa_b = RNG.normal(size=(90, 3)), RNG.normal(size=(90, 2))
    Za, _ = combined_blocks([capa_a, capa_b])
    Zb, _ = combined_blocks([capa_a, 50.0 * capa_b])

    np.testing.assert_allclose(Za, Zb, atol=1e-12)
    np.testing.assert_allclose(eeof(Za, 12, 5)[0], eeof(Zb, 12, 5)[0], rtol=1e-10)


# --------------------------------------------------------------------------- #
# Promedio diagonal e Iterative O-SSA
# --------------------------------------------------------------------------- #


def test_promedio_diagonal_invierte_la_trayectoria() -> None:
    T, K, L = 40, 3, 9
    Z = RNG.normal(size=(T, K))
    np.testing.assert_allclose(diagonal_average(trajectory_matrix(Z, L), L, K, T), Z, atol=1e-12)


def _senos_cercanos():
    """Ejemplo de Golyandina & Shlemov (2015, sec. 3.4): frecuencias 0.065 y 0.06, N = 150, L = 70."""
    n = np.arange(150)
    s1, s2 = np.sin(2 * np.pi * 0.065 * n), 1.2 * np.sin(2 * np.pi * 0.06 * n)
    return (s1 + s2)[:, None], s1, s2


def _corr_media(a: np.ndarray, b: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    c = lambda x, y: abs(np.corrcoef(x, y)[0, 1])
    return max(c(a, s1) + c(b, s2), c(a, s2) + c(b, s1)) / 2


@pytest.mark.parametrize("kappa", [None, 2.0])
def test_iossa_separa_senos_de_frecuencias_cercanas(kappa) -> None:
    """El SSA basico los mezcla; Iterative O-SSA los separa casi exactamente."""
    Z, s1, s2 = _senos_cercanos()
    _, ee, pc = eeof(Z, 70, n_modes=4)
    basico = _corr_media(reconstruct_components(ee, pc, [0, 1], 150)[:, 0],
                         reconstruct_components(ee, pc, [2, 3], 150)[:, 0], s1, s2)
    res = iterative_ossa(Z, 70, [0, 1, 2, 3], [0, 1], kappa=kappa)
    a, b = res.series

    assert basico < 0.7
    assert res.converged
    assert _corr_media(a[:, 0], b[:, 0], s1, s2) > 0.99


def test_iossa_conserva_la_suma_del_grupo() -> None:
    """Los dos componentes suman exactamente el RC del grupo (la (L, R)-SVD es exacta)."""
    Z, _, _ = _senos_cercanos()
    Z = Z + 0.1 * RNG.normal(size=Z.shape)
    _, ee, pc = eeof(Z, 70, n_modes=4)
    a, b = iterative_ossa(Z, 70, [0, 1, 2, 3], [0, 1], kappa=2.0, max_iter=50).series
    np.testing.assert_allclose(a + b, reconstruct_components(ee, pc, [0, 1, 2, 3], 150), atol=1e-8)


@pytest.mark.parametrize("J1", [[], [0, 1, 2, 3], [5]])
def test_iossa_rechaza_particiones_invalidas(J1) -> None:
    Z, _, _ = _senos_cercanos()
    with pytest.raises(ValueError):
        iterative_ossa(Z, 70, [0, 1, 2, 3], J1)


def test_w_correlacion_lr_mide_la_separacion_oblicua() -> None:
    """Tras separar los senos, la W-correlacion (L, R) es ~0 aunque la clasica no lo sea
    (Golyandina & Shlemov 2015, sec. 3.4.1: clasica ~ -0.44, (L, R) ~ 0)."""
    Z, _, _ = _senos_cercanos()
    res = iterative_ossa(Z, 70, [0, 1, 2, 3], [0, 1])
    clasica = wcorr_matrix(np.stack(res.series), 70)[0, 1]

    assert abs(res.lr_wcorr) < 0.05
    assert abs(clasica) > 0.2


# --------------------------------------------------------------------------- #
# Emparejamiento de grupos por subespacio
# --------------------------------------------------------------------------- #


def _como_vectores(ee: np.ndarray, idx: list[int]) -> np.ndarray:
    return ee[idx].transpose(0, 2, 1).reshape(len(idx), -1)


def test_subespacio_reconoce_los_grupos_aunque_cambien_orden_y_signo() -> None:
    Z = RNG.normal(size=(120, 3))
    _, ee, _ = eeof(Z, 12, n_modes=6)
    refs = [_como_vectores(ee, [0, 1]), _como_vectores(ee, [3, 4])]
    perm = [4, 2, 0, 5, 3, 1]
    signos = np.array([1, -1, 1, -1, -1, 1])[:, None, None]
    grupos, sim = match_groups_by_subspace(ee[perm] * signos, refs)

    assert grupos == [sorted([perm.index(0), perm.index(1)]), sorted([perm.index(3), perm.index(4)])]
    np.testing.assert_allclose(sim, 1.0, atol=1e-10)


def test_subespacio_tolera_ruido() -> None:
    """Un seno fuerte sigue reconociendose como el mismo par tras cambiar el ruido."""
    t = np.arange(240)
    base = np.column_stack([np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24 + 0.3)])
    _, ee_a, _ = eeof(base + 0.2 * np.random.default_rng(1).normal(size=base.shape), 30, 6)
    _, ee_b, _ = eeof(base + 0.2 * np.random.default_rng(2).normal(size=base.shape), 30, 6)
    grupos, sim = match_groups_by_subspace(ee_b, [_como_vectores(ee_a, [0, 1])])

    assert grupos == [[0, 1]]
    assert sim[0] > 0.9


# --------------------------------------------------------------------------- #
# DerivSSA, bandas y pureza espectral
# --------------------------------------------------------------------------- #


def _senos_iguales() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejemplo de Golyandina & Shlemov (2015, sec. 4.3): periodos 10 y 15, igual amplitud."""
    n = np.arange(1, 151)
    s10, s15 = np.sin(2 * np.pi * n / 10), np.sin(2 * np.pi * n / 15)
    return (s10 + s15)[:, None], s10 - s10.mean(), s15 - s15.mean()


def _w_entre(ee: np.ndarray, pc: np.ndarray, g1: list[int], g2: list[int]) -> float:
    rcs = np.stack([reconstruct_components(ee, pc, g, 150) for g in (g1, g2)])
    return abs(wcorr_matrix(rcs, 70)[0, 1])


@pytest.mark.parametrize("gamma, normalize", [(10.0, False), (100.0, False), (np.inf, True)])
def test_derivssa_separa_senos_de_igual_amplitud(gamma, normalize) -> None:
    """Articulo: basico W = 0.92; DerivSSA (gamma = 10) W = 0.01."""
    Z, s10, s15 = _senos_iguales()
    _, ee, pc = eeof(Z, 70, n_modes=4)
    ee2, pc2, _ = derivssa(ee, pc, [0, 1, 2, 3], gamma, normalize)

    assert _w_entre(ee, pc, [0, 1], [2, 3]) > 0.8
    assert _w_entre(ee2, pc2, [0, 1], [2, 3]) < 0.02
    rapido = reconstruct_components(ee2, pc2, [0, 1], 150)[:, 0]
    assert np.sqrt(((rapido - s10) ** 2).mean()) < 0.08  # la frecuencia alta sale primero


@pytest.mark.parametrize("normalize", [True, False])
def test_derivssa_conserva_la_suma_del_grupo(normalize) -> None:
    Z = np.column_stack([_seno(300, 44, f) + _seno(300, 118, 2 * f) for f in (0.0, 1.0, 2.0)])
    Z = Z + 0.2 * RNG.normal(size=Z.shape)
    _, ee, pc = eeof(Z, 60, n_modes=8)
    ee2, pc2, _ = derivssa(ee, pc, [0, 1, 2, 3, 5], 3.0, normalize)
    np.testing.assert_allclose(reconstruct_components(ee2, pc2, list(range(5)), 300),
                               reconstruct_components(ee, pc, [0, 1, 2, 3, 5], 300), atol=1e-10)


def test_derivssa_normalizado_no_depende_de_gamma_y_da_st_pc_ortonormales() -> None:
    Z, _, _ = _senos_iguales()
    Z = Z + 0.3 * RNG.normal(size=Z.shape)
    _, ee, pc = eeof(Z, 70, n_modes=6)
    a = derivssa(ee, pc, list(range(6)), 1.0, True)
    b = derivssa(ee, pc, list(range(6)), np.inf, True)
    for m in range(6):  # el signo de cada modo es arbitrario; el RC elemental no
        np.testing.assert_allclose(reconstruct_components(a[0], a[1], [m], 150),
                                   reconstruct_components(b[0], b[1], [m], 150), atol=1e-8)
    np.testing.assert_allclose(a[1].T @ a[1], np.eye(6), atol=1e-10)


def test_derivssa_articulo_da_eeof_ortonormales() -> None:
    Z, _, _ = _senos_iguales()
    _, ee, pc = eeof(Z, 70, n_modes=4)
    ee2, _, _ = derivssa(ee, pc, [0, 1, 2, 3], 10.0, False)
    V = ee2.reshape(4, -1)
    np.testing.assert_allclose(V @ V.T, np.eye(4), atol=1e-10)


@pytest.mark.parametrize("modes", [[], [0, 0]])
def test_derivssa_rechaza_modos_invalidos(modes) -> None:
    Z, _, _ = _senos_iguales()
    _, ee, pc = eeof(Z, 70, n_modes=4)
    with pytest.raises(ValueError):
        derivssa(ee, pc, modes)


def test_fracciones_de_banda_de_un_seno() -> None:
    fr = band_fractions(_seno(528, 44))  # 12 ciclos exactos: sin fuga espectral
    assert fr["ENSO"] > 0.999
    assert sum(fr.values()) == pytest.approx(1.0)


def test_group_by_band_asigna_por_potencia_y_une_por_w() -> None:
    T = 528
    pcs = np.column_stack([_seno(T, 132), _seno(T, 132, 1.5), _seno(T, 44), _seno(T, 44, 1.5), _seno(T, 12)])
    g = group_by_band(pcs)
    assert g == {"decadal": [0, 1], "ENSO": [2, 3], "cuasi-bienal": [], "resto": [4]}
    W = np.eye(5)
    W[3, 4] = W[4, 3] = 0.9  # el bloque {3, 4} suma mas potencia en ENSO que en resto? empate -> 1 y 1
    W[2, 4] = W[4, 2] = 0.9  # bloque {2, 3, 4}: ENSO 2, resto 1 -> ENSO
    assert group_by_band(pcs, wcorr=W)["ENSO"] == [2, 3, 4]


def test_pureza_pondera_canales_por_varianza() -> None:
    T = 528
    comp = np.column_stack([3 * _seno(T, 44), _seno(T, 16)])
    p = spectral_purity(comp)
    assert p["ENSO"] == pytest.approx(0.9, abs=1e-6)
    assert p["cuasi-bienal"] == pytest.approx(0.1, abs=1e-6)
    with pytest.raises(ValueError):
        spectral_purity(np.zeros((T, 2)))


def test_extend_ar_conserva_el_tramo_observado_y_la_forma() -> None:
    rng = np.random.default_rng(42)
    Z = rng.normal(size=(200, 3)) + np.array([1.0, -2.0, 0.5])
    ext = extend_ar(Z, 30, p_max=6)
    assert ext.shape == (260, 3)
    np.testing.assert_allclose(ext[30:230], Z, atol=1e-12)


def test_extend_ar_continua_una_oscilacion_en_ambos_sentidos() -> None:
    n, M, per = 300, 24, 44.0
    t = np.arange(-M, n + M)
    x = 2.0 + np.sin(2 * np.pi * t / per)
    ext = extend_ar(x[M:M + n, None], M, p_max=4)[:, 0]
    np.testing.assert_allclose(ext, x, atol=1e-6)


def test_extend_ar_ruido_blanco_decae_a_la_media() -> None:
    rng = np.random.default_rng(0)
    Z = 5.0 + rng.normal(size=(400, 1))
    ext = extend_ar(Z, 60, p_max=3)
    assert abs(ext[-1, 0] - Z.mean()) < 0.05 and abs(ext[0, 0] - Z.mean()) < 0.05


def test_extend_ar_rechaza_series_cortas() -> None:
    with pytest.raises(ValueError):
        extend_ar(np.ones((20, 1)), 5, p_max=24)
