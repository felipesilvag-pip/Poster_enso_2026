"""Tests del EOF de linea base.

El nucleo son las cuatro pruebas de equivalencia de ``Entendiendo_EOF.py``
l.63-86, que ahi se imprimian por consola y aqui se verifican de forma
automatica: ortonormalidad de los modos, e igualdad de autovalores, patrones
espaciales y amplitudes temporales entre el SVD directo y el PCA clasico por
matriz de covarianza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import config
from enso_chile.cube import apply_area_weights, flatten_space
from enso_chile.eof import (
    align_modes,
    combined_blocks,
    combined_eof,
    combined_eof_from_results,
    eof_decomposition,
    eof_from_dataarray,
    pixel_loadings,
    scores_from_pattern,
)
from enso_chile.rotations import promax, varimax

SEED = config.RANDOM_SEED


def _matrix(n_time: int = 60, n_space: int = 40) -> np.ndarray:
    """Matriz con dos modos impuestos mas ruido, en convencion (tiempo, espacio)."""
    rng = np.random.default_rng(SEED)
    t = np.arange(n_time)

    pc1 = np.sin(2 * np.pi * t / 24)
    pc2 = np.cos(2 * np.pi * t / 7)
    e1 = np.exp(-((np.arange(n_space) - 10) ** 2) / 50)
    e2 = np.exp(-((np.arange(n_space) - 30) ** 2) / 50)

    return (
        6.0 * np.outer(pc1, e1)
        + 2.0 * np.outer(pc2, e2)
        + 0.1 * rng.normal(size=(n_time, n_space))
    )


def _field(n_time: int = 60, n_lat: int = 8, n_lon: int = 6) -> xr.DataArray:
    """Campo sintetico con coordenadas, dentro de la caja Norte Grande."""
    rng = np.random.default_rng(SEED)
    t = np.arange(n_time)
    pc = np.sin(2 * np.pi * t / 30)

    lat = np.linspace(-27.0, -18.0, n_lat)
    lon = np.linspace(-74.0, -69.0, n_lon)
    patron = np.exp(-((lat[:, None] + 22.0) ** 2) / 8.0) * np.ones(n_lon)

    values = pc[:, None, None] * patron + 0.2 * rng.normal(
        size=(n_time, n_lat, n_lon)
    )
    return xr.DataArray(
        values,
        coords={
            "time": pd.date_range("1993-01-01", periods=n_time, freq="MS"),
            "latitude": lat,
            "longitude": lon,
        },
        dims=["time", "latitude", "longitude"],
        name="thetao_anom",
    )


def _pca_clasico(matrix: np.ndarray, k: int):
    """PCA por matriz de covarianza temporal (metodo de snapshots).

    Reproduce ``Entendiendo_EOF.py`` l.38-61 en convencion (tiempo, espacio),
    para contrastar contra el SVD.
    """
    n_time = matrix.shape[0]
    x = matrix - matrix.mean(axis=0, keepdims=True)

    cov = (x @ x.T) / (n_time - 1)
    lam, vec = np.linalg.eigh(cov)
    orden = np.argsort(lam)[::-1][:k]
    lam, vec = lam[orden], vec[:, orden]

    pcs = vec * np.sqrt((n_time - 1) * lam)
    eofs = (pcs.T @ x) / ((n_time - 1) * lam[:, None])
    return lam, eofs, pcs


# --------------------------------------------------------------------------- #
# Las cuatro pruebas de equivalencia del script original
# --------------------------------------------------------------------------- #


def test_modos_son_ortonormales() -> None:
    """``E Eᵀ = I``: los patrones espaciales son ortonormales."""
    res = eof_decomposition(_matrix())
    producto = res.eofs @ res.eofs.T

    np.testing.assert_allclose(producto, np.eye(res.n_modes), atol=1e-10)


def test_autovalores_identicos_svd_y_pca() -> None:
    """Los autovalores del SVD y del PCA por covarianza coinciden."""
    matrix = _matrix()
    res = eof_decomposition(matrix)
    lam, _, _ = _pca_clasico(matrix, res.n_modes)

    np.testing.assert_allclose(res.eigenvalues, lam, rtol=1e-8, atol=1e-12)


def test_patrones_espaciales_identicos_svd_y_pca() -> None:
    """Los patrones coinciden salvo el signo, que es arbitrario en ambos."""
    matrix = _matrix()
    res = eof_decomposition(matrix)
    _, eofs, _ = _pca_clasico(matrix, res.n_modes)

    # Solo los modos bien separados: en los de varianza casi nula el signo y la
    # orientacion son numericamente inestables.
    k = int((res.variance_fraction > 1e-6).sum())
    np.testing.assert_allclose(np.abs(res.eofs[:k]), np.abs(eofs[:k]), atol=1e-7)


def test_amplitudes_temporales_identicas_svd_y_pca() -> None:
    """Las PC coinciden salvo el signo."""
    matrix = _matrix()
    res = eof_decomposition(matrix)
    _, _, pcs = _pca_clasico(matrix, res.n_modes)

    k = int((res.variance_fraction > 1e-6).sum())
    np.testing.assert_allclose(
        np.abs(res.pcs[:, :k]), np.abs(pcs[:, :k]), atol=1e-7
    )


# --------------------------------------------------------------------------- #
# Propiedades algebraicas
# --------------------------------------------------------------------------- #


def test_pcs_estan_descorrelacionadas() -> None:
    """La covarianza entre PC distintas es nula: la base es ortogonal."""
    res = eof_decomposition(_matrix())
    cov = res.pcs.T @ res.pcs

    fuera = cov - np.diag(np.diag(cov))
    assert np.abs(fuera).max() < 1e-8 * np.abs(np.diag(cov)).max()


def test_rango_es_t_menos_uno() -> None:
    """Centrar en el tiempo reduce el rango a ``min(T - 1, N)``.

    Es el ``k = T - 1`` del script original: el ultimo autovalor del problema
    de covarianza es ruido numerico y no debe retenerse.
    """
    res = eof_decomposition(_matrix(n_time=30, n_space=50))
    assert res.rank == 29

    ancho = eof_decomposition(_matrix(n_time=60, n_space=12))
    assert ancho.rank == 12


def test_varianzas_suman_uno() -> None:
    """Las fracciones de varianza del rango completo suman exactamente 1."""
    res = eof_decomposition(_matrix())
    assert res.variance_fraction.sum() == pytest.approx(1.0)
    assert res.cumulative_variance()[-1] == pytest.approx(1.0)


def test_truncar_no_reescala_las_varianzas() -> None:
    """Retener 3 modos no hace que esos 3 expliquen el 100 por ciento."""
    matrix = _matrix()
    completo = eof_decomposition(matrix)
    truncado = eof_decomposition(matrix, n_modes=3)

    np.testing.assert_allclose(
        truncado.variance_fraction, completo.variance_fraction[:3]
    )
    assert truncado.variance_fraction.sum() < 1.0


def test_varianza_total_se_conserva() -> None:
    """La suma de autovalores es la varianza total de la matriz centrada."""
    matrix = _matrix()
    res = eof_decomposition(matrix)
    centrada = matrix - matrix.mean(axis=0, keepdims=True)
    total = (centrada**2).sum() / (matrix.shape[0] - 1)

    assert res.eigenvalues.sum() == pytest.approx(total, rel=1e-10)


def test_autovalores_decrecen() -> None:
    res = eof_decomposition(_matrix())
    assert np.all(np.diff(res.eigenvalues) <= 1e-12)


# --------------------------------------------------------------------------- #
# Reconstruccion
# --------------------------------------------------------------------------- #


def test_reconstruccion_completa_recupera_los_datos() -> None:
    """Con todos los modos del rango se recupera la matriz original."""
    matrix = _matrix()
    res = eof_decomposition(matrix)

    np.testing.assert_allclose(res.reconstruct(), matrix, atol=1e-10)


def test_reconstruccion_completa_tambien_con_pc_estandarizadas() -> None:
    """El escalado no altera la reconstruccion."""
    matrix = _matrix()
    res = eof_decomposition(matrix, scaling="std_pc")

    np.testing.assert_allclose(res.reconstruct(), matrix, atol=1e-10)


def test_truncar_da_la_mejor_aproximacion_de_ese_rango() -> None:
    """Eckart-Young: ninguna matriz de rango k se acerca mas en Frobenius.

    Se contrasta contra 30 perturbaciones aleatorias de rango k construidas a
    partir de modos mezclados, que no deben mejorar el error.
    """
    matrix = _matrix()
    res = eof_decomposition(matrix)
    centrada = matrix - res.mean
    k = 2

    error_eof = np.linalg.norm(centrada - (res.pcs[:, :k] @ res.eofs[:k]))

    rng = np.random.default_rng(SEED)
    for _ in range(30):
        mezcla = rng.normal(size=(k, res.n_modes))
        base = mezcla @ res.eofs
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        proyeccion = centrada @ base.T @ base
        assert np.linalg.norm(centrada - proyeccion) >= error_eof - 1e-9


def test_error_de_reconstruccion_decrece_con_mas_modos() -> None:
    matrix = _matrix()
    res = eof_decomposition(matrix)
    errores = [
        np.linalg.norm(matrix - res.reconstruct(k)) for k in (1, 2, 5, 10)
    ]
    assert all(np.diff(errores) < 0)


# --------------------------------------------------------------------------- #
# Recuperacion de modos impuestos
# --------------------------------------------------------------------------- #


def test_recupera_el_modo_dominante_impuesto() -> None:
    """El primer modo reproduce la PC que se impuso, con su periodo de 24 meses."""
    matrix = _matrix()
    res = eof_decomposition(matrix)

    t = np.arange(matrix.shape[0])
    pc1_real = np.sin(2 * np.pi * t / 24)
    r = abs(np.corrcoef(res.pcs[:, 0], pc1_real)[0, 1])

    assert r > 0.99
    assert res.variance_fraction[0] > 0.8


def test_separa_el_segundo_modo() -> None:
    """El segundo modo recupera la segunda senal, no una mezcla de la primera."""
    matrix = _matrix()
    res = eof_decomposition(matrix)

    t = np.arange(matrix.shape[0])
    pc2_real = np.cos(2 * np.pi * t / 7)
    assert abs(np.corrcoef(res.pcs[:, 1], pc2_real)[0, 1]) > 0.95


# --------------------------------------------------------------------------- #
# Signo y reproducibilidad
# --------------------------------------------------------------------------- #


def test_signo_es_reproducible() -> None:
    """Dos corridas sobre los mismos datos dan exactamente el mismo signo."""
    matrix = _matrix()
    a = eof_decomposition(matrix)
    b = eof_decomposition(matrix)

    np.testing.assert_array_equal(a.eofs, b.eofs)
    np.testing.assert_array_equal(a.pcs, b.pcs)


def test_signo_fija_positiva_la_componente_dominante() -> None:
    """La componente de mayor magnitud de cada EOF es positiva por convencion."""
    res = eof_decomposition(_matrix())
    for i in range(res.n_modes):
        dominante = np.argmax(np.abs(res.eofs[i]))
        assert res.eofs[i, dominante] > 0


def test_signo_invariante_al_voltear_la_entrada() -> None:
    """Cambiar el signo de los datos no cambia el patron, solo la PC."""
    matrix = _matrix()
    a = eof_decomposition(matrix)
    b = eof_decomposition(-matrix)

    np.testing.assert_allclose(a.eofs, b.eofs, atol=1e-10)
    np.testing.assert_allclose(a.pcs, -b.pcs, atol=1e-10)


# --------------------------------------------------------------------------- #
# Escalados
# --------------------------------------------------------------------------- #


def test_pc_estandarizada_tiene_varianza_unitaria() -> None:
    """Con ``std_pc`` cada PC tiene varianza 1 y el EOF lleva las unidades."""
    res = eof_decomposition(_matrix(), scaling="std_pc")
    assert res.pcs.var(axis=0, ddof=1).max() == pytest.approx(1.0, abs=1e-8)


def test_los_dos_escalados_dan_la_misma_varianza_explicada() -> None:
    """El escalado es una reparametrizacion: no cambia los autovalores."""
    matrix = _matrix()
    a = eof_decomposition(matrix, scaling="orthonormal")
    b = eof_decomposition(matrix, scaling="std_pc")

    np.testing.assert_allclose(a.variance_fraction, b.variance_fraction)
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues)


def test_escalado_invalido_falla() -> None:
    with pytest.raises(ValueError, match="scaling"):
        eof_decomposition(_matrix(), scaling="varimax")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Validaciones de entrada
# --------------------------------------------------------------------------- #


def test_nan_en_la_matriz_falla_con_mensaje_util() -> None:
    """Un NaN debe fallar apuntando a flatten_space, no romper el SVD."""
    matrix = _matrix()
    matrix[3, 4] = np.nan

    with pytest.raises(ValueError, match="flatten_space"):
        eof_decomposition(matrix)


def test_matriz_no_bidimensional_falla() -> None:
    with pytest.raises(ValueError, match="2-D"):
        eof_decomposition(np.zeros((4, 5, 6)))


def test_n_modes_excesivo_falla() -> None:
    with pytest.raises(ValueError, match="n_modes"):
        eof_decomposition(_matrix(n_time=20, n_space=10), n_modes=50)


# --------------------------------------------------------------------------- #
# Integracion con xarray
# --------------------------------------------------------------------------- #


def test_desde_dataarray_aplica_la_ponderacion() -> None:
    """``eof_from_dataarray`` pondera por area antes del SVD."""
    campo = _field()
    con_peso = eof_from_dataarray(campo, n_modes=3)

    manual, _ = flatten_space(apply_area_weights(campo))
    esperado = eof_decomposition(manual, n_modes=3)

    np.testing.assert_allclose(con_peso.eigenvalues, esperado.eigenvalues)
    assert con_peso.weights is not None


def test_ponderar_cambia_el_resultado() -> None:
    """La ponderacion no es cosmetica: altera las varianzas explicadas."""
    campo = _field()
    con = eof_from_dataarray(campo, n_modes=3, weight=True)
    sin = eof_from_dataarray(campo, n_modes=3, weight=False)

    assert con.variance_fraction[0] != pytest.approx(
        sin.variance_fraction[0], abs=1e-12
    )


def test_mapas_tienen_las_dimensiones_espaciales() -> None:
    """Los patrones vuelven a su malla, con una dimension ``mode``."""
    res = eof_from_dataarray(_field(), n_modes=3)
    mapas = res.eofs_as_maps()

    assert set(mapas.dims) == {"mode", "latitude", "longitude"}
    assert list(mapas.mode.values) == [1, 2, 3]


def test_mapas_desponderados_estan_en_unidades_fisicas() -> None:
    """Desponderar divide por ``sqrt(cos(lat))`` y cambia el patron."""
    res = eof_from_dataarray(_field(), n_modes=2)
    fisico = res.eofs_as_maps(unweight=True)
    ponderado = res.eofs_as_maps(unweight=False)

    razon = (ponderado / fisico).isel(mode=0).mean(dim="longitude")
    np.testing.assert_allclose(
        razon.values, res.weights.values, rtol=1e-10
    )


def test_ortonormalidad_se_cumple_en_el_espacio_ponderado() -> None:
    """La ortonormalidad vive en el espacio ponderado, no en el fisico.

    Documenta por que ``eofs_as_maps`` tiene el argumento ``unweight``: el
    patron que se grafica ya no es ortonormal, y eso es correcto.
    """
    res = eof_from_dataarray(_field(), n_modes=5)
    np.testing.assert_allclose(res.eofs @ res.eofs.T, np.eye(5), atol=1e-10)


def test_pcs_as_dataarray_conserva_el_eje_temporal() -> None:
    campo = _field()
    res = eof_from_dataarray(campo, n_modes=2)
    pcs = res.pcs_as_dataarray()

    assert pcs.dims == ("time", "mode")
    np.testing.assert_array_equal(pcs.time.values, campo.time.values)


def test_tierra_vuelve_como_nan_en_los_mapas() -> None:
    """Los pixeles de tierra no entran al SVD y reaparecen como NaN."""
    campo = _field()
    campo[:, 0, 0] = np.nan

    res = eof_from_dataarray(campo, n_modes=2)
    mapas = res.eofs_as_maps()

    assert res.n_space == campo.sizes["latitude"] * campo.sizes["longitude"] - 1
    assert bool(mapas.isel(mode=0).sel(latitude=campo.latitude[0], longitude=campo.longitude[0]).isnull())


def test_mapear_sin_coordenadas_falla() -> None:
    """Una descomposicion desde matriz cruda no puede mapear patrones."""
    res = eof_decomposition(_matrix())
    with pytest.raises(ValueError, match="eof_from_dataarray"):
        res.eofs_as_maps()


# --------------------------------------------------------------------------- #
# Alineacion de signos entre descomposiciones
# --------------------------------------------------------------------------- #


def test_alinear_corrige_el_signo_opuesto() -> None:
    """Dos versiones del mismo modo quedan con el mismo signo.

    Sin esto, comparar dos descomposiciones en un mapa divergente muestra el
    mismo patron en colores opuestos y sugiere una diferencia que no existe.
    """
    matrix = _matrix()
    ref = eof_decomposition(matrix, n_modes=3)

    volteado = eof_decomposition(matrix, n_modes=3)
    volteado.eofs = volteado.eofs * np.array([-1.0, 1.0, -1.0])[:, None]
    volteado.pcs = volteado.pcs * np.array([-1.0, 1.0, -1.0])[None, :]

    alineado = align_modes(ref, volteado)

    np.testing.assert_allclose(alineado.eofs, ref.eofs, atol=1e-12)
    np.testing.assert_allclose(alineado.pcs, ref.pcs, atol=1e-12)


def test_alinear_no_altera_lo_que_ya_coincide() -> None:
    matrix = _matrix()
    ref = eof_decomposition(matrix, n_modes=3)
    alineado = align_modes(ref, eof_decomposition(matrix, n_modes=3))

    np.testing.assert_array_equal(alineado.eofs, ref.eofs)


def test_alinear_no_muta_la_entrada() -> None:
    """La descomposicion original queda intacta."""
    matrix = _matrix()
    ref = eof_decomposition(matrix, n_modes=2)
    otro = eof_decomposition(matrix, n_modes=2)
    otro.eofs = -otro.eofs
    copia = otro.eofs.copy()

    align_modes(ref, otro)

    np.testing.assert_array_equal(otro.eofs, copia)


def test_alinear_conserva_las_varianzas() -> None:
    """Cambiar el signo no toca autovalores ni varianza explicada."""
    matrix = _matrix()
    ref = eof_decomposition(matrix, n_modes=3)
    otro = eof_decomposition(matrix * 2.0, n_modes=3)
    alineado = align_modes(ref, otro)

    np.testing.assert_array_equal(alineado.eigenvalues, otro.eigenvalues)
    np.testing.assert_array_equal(
        alineado.variance_fraction, otro.variance_fraction
    )


def test_alinear_preserva_la_reconstruccion() -> None:
    """Voltear un par (EOF, PC) no cambia el campo que reconstruye."""
    matrix = _matrix()
    ref = eof_decomposition(matrix)
    otro = eof_decomposition(matrix)
    otro.eofs = -otro.eofs
    otro.pcs = -otro.pcs

    np.testing.assert_allclose(
        align_modes(ref, otro).reconstruct(), matrix, atol=1e-10
    )


def test_alinear_espacios_distintos_falla() -> None:
    ref = eof_decomposition(_matrix(n_space=40), n_modes=2)
    otro = eof_decomposition(_matrix(n_space=25), n_modes=2)

    with pytest.raises(ValueError, match="no comparten el espacio"):
        align_modes(ref, otro)


# --------------------------------------------------------------------------- #
# EOF combinado
# --------------------------------------------------------------------------- #


def _capas(n_time: int = 80, sizes: tuple[int, ...] = (20, 20, 20), seed: int = SEED) -> list[np.ndarray]:
    """Capas con una senal comun, una propia de cada capa y ruido."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    comun = np.sin(2 * np.pi * t / 24)
    propias = [np.cos(2 * np.pi * t / 9), np.sin(2 * np.pi * t / 13 + 1), np.cos(2 * np.pi * t / 31)]
    return [np.outer(comun, rng.normal(size=p)) + np.outer(propias[d % 3], rng.normal(size=p))
            + 0.3 * rng.normal(size=(n_time, p)) for d, p in enumerate(sizes)]


def _abs_corr_columnas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - a.mean(0)) / np.linalg.norm(a - a.mean(0), axis=0)
    b = (b - b.mean(0)) / np.linalg.norm(b - b.mean(0), axis=0)
    return np.abs((a * b).sum(0))


def test_combinado_con_todas_las_pc_equivale_al_campo_apilado() -> None:
    """(1) Prefiltro con todas las PC = EOF del campo concatenado (autovalores y PC)."""
    capas = _capas()
    directo = combined_eof(capas)
    res = [eof_decomposition(c) for c in capas]
    prefiltro = combined_eof_from_results(res, n_pcs=20)

    np.testing.assert_allclose(prefiltro.eigenvalues, directo.eigenvalues, rtol=1e-10)
    np.testing.assert_allclose(prefiltro.retained, 1.0)
    np.testing.assert_allclose(_abs_corr_columnas(prefiltro.pcs, directo.pcs), 1.0, atol=1e-8)


def test_combinado_es_invariante_a_la_escala_de_una_capa() -> None:
    """(2) Con normalizacion, multiplicar una capa por 100 no cambia nada; sin ella, si."""
    capas = _capas()
    escaladas = [capas[0] * 100.0] + capas[1:]
    a, b = combined_eof(capas), combined_eof(escaladas)

    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-10)
    np.testing.assert_allclose(_abs_corr_columnas(a.pcs, b.pcs), 1.0, atol=1e-8)
    sin_norm = combined_eof(escaladas, normalize=False, n_modes=3)
    assert sin_norm.block_share()[0, 0] > 0.99  # sin normalizar domina la capa escalada


def test_combinado_es_invariante_al_orden_de_las_capas() -> None:
    """(3) El SVD no conoce la profundidad: permutar capas da los mismos modos."""
    capas = _capas()
    a, b = combined_eof(capas), combined_eof(capas[::-1])

    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-10)
    np.testing.assert_allclose(_abs_corr_columnas(a.pcs, b.pcs), 1.0, atol=1e-8)
    np.testing.assert_allclose(a.block_share(), b.block_share()[:, ::-1], atol=1e-10)


def test_capas_independientes_no_se_mezclan() -> None:
    """(4) Sin covarianza entre capas, cada modo combinado vive en una sola capa."""
    n = 120
    t = np.arange(n)
    rng = np.random.default_rng(SEED)
    base = lambda k: np.sin(2 * np.pi * k * t / n)  # senos de frecuencia entera: ortogonales
    capa_a = np.outer(base(1), rng.normal(size=10)) + 0.5 * np.outer(base(2), rng.normal(size=10))
    capa_b = np.outer(base(3), rng.normal(size=8)) + 0.5 * np.outer(base(4), rng.normal(size=8))
    comb = combined_eof([capa_a, capa_b], n_modes=4)

    assert np.all(comb.block_share().max(1) > 0.999)


def test_senal_comun_se_reparte_igual_entre_capas() -> None:
    """(5) La misma senal con igual relacion senal/ruido en cada capa: el modo 1 la captura
    y reparte su energia por igual, aunque las capas tengan amplitudes muy distintas."""
    n = 200
    rng = np.random.default_rng(SEED)
    s = np.sin(2 * np.pi * np.arange(n) / 30)
    capas = []
    for amp in (1.0, 10.0, 100.0):
        senal = amp * np.outer(s, rng.normal(size=15))
        capas.append(senal + 0.5 * senal.std() * rng.normal(size=senal.shape))
    comb = combined_eof(capas, n_modes=3)

    assert _abs_corr_columnas(comb.pcs[:, :1], s[:, None])[0] > 0.95
    np.testing.assert_allclose(comb.block_share()[0], 1 / 3, atol=0.05)


def test_contabilidad_del_r2_por_capa() -> None:
    """(6) Con todos los modos el R2 de cada capa suma lo retenido, y el promedio entre
    capas del R2 de un modo es su fraccion del total combinado."""
    capas = _capas()
    comb = combined_eof(capas)
    r2 = comb.r2_by_block()
    np.testing.assert_allclose(r2.sum(0), 1.0, rtol=1e-10)
    np.testing.assert_allclose(r2.mean(1), comb.variance_fraction, rtol=1e-10)

    res = [eof_decomposition(c) for c in capas]
    pre = combined_eof_from_results(res, n_pcs=5)
    np.testing.assert_allclose(pre.r2_by_block().sum(0), pre.retained, rtol=1e-10)
    with pytest.raises(ValueError):
        combined_eof(capas, normalize=False).r2_by_block()


def test_cargas_por_pixel_son_una_isometria() -> None:
    """(7) Pasar de cargas por PC a cargas por pixel conserva los productos internos."""
    capas = _capas(sizes=(20, 15, 12))
    res = [eof_decomposition(c) for c in capas]
    comb = combined_eof_from_results(res, n_pcs=5)
    A = comb.loadings(3)
    L = pixel_loadings(A, res, n_pcs=5)

    assert L.shape == (47, 3)
    np.testing.assert_allclose(L.T @ L, A.T @ A, atol=1e-12)


def test_rotaciones_conservan_el_campo_ajustado() -> None:
    """(8) Varimax conserva varianza total y comunalidades; varimax y promax reproducen
    el mismo campo ajustado F A^T."""
    comb = combined_eof(_capas())
    A, F = comb.loadings(4), comb.scores(4)
    ajustado = F @ A.T

    B, R = varimax(A)
    assert (B ** 2).sum() == pytest.approx((A ** 2).sum())
    np.testing.assert_allclose((B ** 2).sum(1), (A ** 2).sum(1), rtol=1e-10)
    np.testing.assert_allclose((F @ R) @ B.T, ajustado, atol=1e-10)

    P, Tm, _ = promax(A)
    np.testing.assert_allclose((F @ np.linalg.inv(Tm).T) @ P.T, ajustado, atol=1e-8)


def test_proyeccion_por_minimos_cuadrados_da_los_puntajes_rotados() -> None:
    """(9) Proyectar los datos sobre el patron rotado reproduce los puntajes rotados."""
    capas = _capas()
    Z, _ = combined_blocks(capas)
    comb = combined_eof(capas)
    A, F = comb.loadings(4), comb.scores(4)
    estandarizar = lambda M: (M - M.mean(0)) / M.std(0, ddof=1)

    _, R = varimax(A)
    np.testing.assert_allclose(scores_from_pattern(Z, A @ R), estandarizar(F @ R), atol=1e-8)
    _, Tm, _ = promax(A)
    np.testing.assert_allclose(scores_from_pattern(Z, A @ Tm),
                               estandarizar(F @ np.linalg.inv(Tm).T), atol=1e-8)


def test_bloques_con_filas_distintas_fallan() -> None:
    with pytest.raises(ValueError):
        combined_blocks([np.ones((10, 2)), np.ones((9, 2))])
