"""Tests del recorte, la ponderacion por area y el aplanado del cubo."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import config
from enso_chile.cube import (
    apply_area_weights,
    area_weights,
    flatten_space,
    open_cube,
    save_cube,
    select_box,
    select_depths,
    unflatten_space,
    water_mask,
)


def _grid(n_time: int = 12) -> xr.DataArray:
    """Malla sintetica de 1/12 de grado que cubre las dos cajas."""
    latitude = np.round(np.arange(-33.5, -17.999, 1 / 12), 4)
    longitude = np.round(np.arange(-78.0, -68.999, 1 / 12), 4)
    time = pd.date_range("1993-01-01", periods=n_time, freq="MS")

    rng = np.random.default_rng(config.RANDOM_SEED)
    values = rng.normal(size=(n_time, latitude.size, longitude.size))

    return xr.DataArray(
        values,
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        dims=["time", "latitude", "longitude"],
        name="thetao_anom",
    )


# --------------------------------------------------------------------------- #
# Apertura
# --------------------------------------------------------------------------- #


def test_open_cube_falla_con_ruta_inexistente(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        open_cube(tmp_path / "no_existe.nc")


# --------------------------------------------------------------------------- #
# Recorte
# --------------------------------------------------------------------------- #


def test_recorte_respeta_los_limites_de_la_caja() -> None:
    """Ningun pixel queda fuera de la caja pedida."""
    out = select_box(_grid(), config.BOXES["norte_grande"])

    assert float(out.latitude.min()) >= -27.0
    assert float(out.latitude.max()) <= -18.0
    assert float(out.longitude.min()) >= -74.0


def test_las_dos_cajas_comparten_la_fila_de_27_grados() -> None:
    """Sin el argumento, ``.sel(slice)`` incluye -27.0 en ambas cajas.

    Documenta el motivo de ``exclude_north_edge``: el solapamiento existe y es
    de una fila completa de longitudes.
    """
    grande = select_box(_grid(), config.BOXES["norte_grande"])
    chico = select_box(_grid(), config.BOXES["norte_chico"])

    compartidas = set(np.round(grande.latitude.values, 4)) & set(
        np.round(chico.latitude.values, 4)
    )
    assert compartidas == {-27.0}


def test_exclude_north_edge_elimina_el_solapamiento() -> None:
    """Con el argumento activo, las dos cajas son disjuntas."""
    grande = select_box(_grid(), config.BOXES["norte_grande"])
    chico = select_box(
        _grid(), config.BOXES["norte_chico"], exclude_north_edge=True
    )

    assert float(chico.latitude.max()) < -27.0
    assert not set(np.round(grande.latitude.values, 4)) & set(
        np.round(chico.latitude.values, 4)
    )


def test_las_dos_cajas_cubren_el_dominio_sin_huecos() -> None:
    """Partir en -27.0 no pierde ninguna latitud entre las dos cajas."""
    grid = _grid()
    dominio = grid.sel(latitude=slice(-32.5, -18.0), longitude=slice(-74.0, -69.0))
    grande = select_box(grid, config.BOXES["norte_grande"])
    chico = select_box(grid, config.BOXES["norte_chico"], exclude_north_edge=True)

    assert chico.sizes["latitude"] + grande.sizes["latitude"] == dominio.sizes["latitude"]


def test_caja_vacia_falla() -> None:
    """Limites invertidos dejan la caja vacia y eso es un error."""
    with pytest.raises(ValueError, match="vacia"):
        select_box(_grid(), {"lat": (-18.0, -27.0), "lon": (-74.0, -69.0)})


def test_recorte_es_perezoso(tmp_path) -> None:
    """Abrir y recortar no materializa datos en memoria."""
    path = save_cube(_grid().to_dataset(), tmp_path / "cubo.nc")

    with open_cube(path) as ds:
        recortado = select_box(ds, config.BOXES["norte_grande"])
        assert isinstance(
            recortado["thetao_anom"].variable._data,
            type(ds["thetao_anom"].variable._data),
        )


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #


def test_guardar_y_releer_conserva_los_datos(tmp_path) -> None:
    """Un cubo guardado y vuelto a abrir es identico al original.

    Tambien ejercita el bloqueo temporal de ``distributed``: sin el, cualquier
    ``to_netcdf`` falla en este equipo con ``SSLError`` al importar
    ``dask.distributed``.
    """
    original = _grid().to_dataset()
    path = save_cube(original, tmp_path / "cubo.nc")

    with open_cube(path) as vuelto:
        xr.testing.assert_allclose(vuelto["thetao_anom"], original["thetao_anom"])


def test_no_sobrescribe_sin_permiso(tmp_path) -> None:
    """Regenerar un cubo no debe destruir el anterior por accidente."""
    path = save_cube(_grid().to_dataset(), tmp_path / "cubo.nc")

    with pytest.raises(FileExistsError):
        save_cube(_grid().to_dataset(), path)

    assert save_cube(_grid().to_dataset(), path, overwrite=True) == path


def test_guardar_restaura_sys_modules(tmp_path) -> None:
    """El centinela de ``distributed`` no queda en ``sys.modules`` despues."""
    save_cube(_grid().to_dataset(), tmp_path / "cubo.nc")
    assert sys.modules.get("distributed", "ausente") is not None


def test_compresion_reduce_el_tamanio(tmp_path) -> None:
    """La compresion zlib por defecto achica el archivo de forma apreciable."""
    constante = _grid() * 0.0
    comprimido = save_cube(constante.to_dataset(), tmp_path / "zip.nc")
    plano = save_cube(constante.to_dataset(), tmp_path / "plano.nc", compress=False)

    assert comprimido.stat().st_size < plano.stat().st_size


# --------------------------------------------------------------------------- #
# Profundidades
# --------------------------------------------------------------------------- #


def _with_depth() -> xr.DataArray:
    niveles = [0.494, 1.541, 25.211, 92.326, 155.851, 222.475]
    return _grid(3).expand_dims(depth=niveles).transpose("time", "depth", "latitude", "longitude")


def test_selecciona_los_niveles_de_trabajo_en_orden() -> None:
    """Se devuelven exactamente las cuatro profundidades de ``config``."""
    out = select_depths(_with_depth(), config.DEPTHS_WORK)

    assert list(np.round(out.depth.values, 3)) == list(config.DEPTHS_WORK)


def test_profundidad_inexistente_falla_en_vez_de_aproximar() -> None:
    """Pedir 100 m exactos falla: el nivel real es 92.326 m.

    Es deliberado: ``method='nearest'`` sustituiria el nivel en silencio y una
    figura quedaria rotulada con una profundidad que no es la del dato.
    """
    with pytest.raises(ValueError, match="no existe en el cubo"):
        select_depths(_with_depth(), [100.0])


def test_tolerancia_configurable_admite_el_nivel_cercano() -> None:
    """Con tolerancia amplia, 100 m resuelve al nivel de 92.326 m."""
    out = select_depths(_with_depth(), [100.0], tol=10.0)
    assert float(out.depth.values[0]) == pytest.approx(92.326)


def test_redondeo_permite_sel_con_los_literales_de_config() -> None:
    """``.sel(depth=0.494)`` funciona tras el redondeo de la coordenada.

    En el cubo real ``depth`` es ``float32`` y el nivel de superficie vale
    0.49399998784, asi que sin redondeo el literal de ``config.DEPTHS_WORK``
    levanta ``KeyError``.
    """
    sucia = np.asarray(config.DEPTHS_WORK, dtype=np.float32).astype(np.float64)
    da = _with_depth().isel(depth=slice(0, 4)).assign_coords(depth=sucia)

    with pytest.raises(KeyError):
        da.sel(depth=0.494)

    out = select_depths(da, config.DEPTHS_WORK, tol=1e-4)

    assert out.depth.dtype == np.float64
    for d in config.DEPTHS_WORK:
        assert float(out.sel(depth=d).depth) == d


def test_redondeo_usa_el_nivel_real_y_no_el_pedido() -> None:
    """Pedir 100 m con tolerancia amplia etiqueta 92.326, no 100.

    La etiqueta nunca debe mentir sobre la profundidad del dato.
    """
    out = select_depths(_with_depth(), [100.0], tol=10.0)

    assert float(out.depth.values[0]) == 92.326
    assert "92.32" in out.depth.attrs["depth_original"]


def test_redondeo_desactivable() -> None:
    """Con ``round_decimals=None`` la coordenada queda intacta."""
    da = _with_depth()
    out = select_depths(da, config.DEPTHS_WORK, round_decimals=None)

    np.testing.assert_array_equal(out.depth.values, np.array(config.DEPTHS_WORK))


# --------------------------------------------------------------------------- #
# Mascara y pesos
# --------------------------------------------------------------------------- #


def test_mascara_excluye_cobertura_parcial() -> None:
    """Un pixel con un solo mes ausente no cuenta como agua."""
    da = _grid()
    da[:, 0, 0] = np.nan   # tierra
    da[3, 1, 1] = np.nan   # cobertura parcial

    mask = water_mask(da)

    assert not bool(mask[0, 0])
    assert not bool(mask[1, 1])
    assert bool(mask[2, 2])
    assert "time" not in mask.dims


def test_pesos_son_sqrt_cos_lat() -> None:
    """El peso es exactamente la raiz del coseno de la latitud."""
    da = _grid()
    pesos = area_weights(da)
    esperado = np.sqrt(np.cos(np.deg2rad(da.latitude.values)))

    np.testing.assert_allclose(pesos.values, esperado)


def test_pesos_decrecen_hacia_el_polo() -> None:
    """A 32.5 grados sur el peso es menor que a 18 grados sur."""
    pesos = area_weights(_grid())
    assert float(pesos.sel(latitude=-32.5, method="nearest")) < float(
        pesos.sel(latitude=-18.0, method="nearest")
    )


def test_pesos_estan_entre_cero_y_uno() -> None:
    pesos = area_weights(_grid())
    assert float(pesos.min()) > 0.0
    assert float(pesos.max()) <= 1.0


def test_aplicar_pesos_anota_la_ponderacion() -> None:
    """El campo ponderado registra que lleva ``sqrt_coslat``."""
    da = _grid()
    ponderado = apply_area_weights(da)

    assert ponderado.attrs["area_weighting"] == "sqrt_coslat"
    assert ponderado.name == da.name
    np.testing.assert_allclose(
        ponderado.isel(latitude=0).values,
        da.isel(latitude=0).values * float(area_weights(da)[0]),
    )


# --------------------------------------------------------------------------- #
# Aplanado
# --------------------------------------------------------------------------- #


def test_aplanado_da_matriz_tiempo_por_espacio() -> None:
    """La matriz tiene una fila por mes y una columna por pixel de agua."""
    da = select_box(_grid(), config.BOXES["norte_grande"])
    matrix, index = flatten_space(da)

    assert matrix.shape == (da.sizes["time"], da.sizes["latitude"] * da.sizes["longitude"])
    assert index.dims == ("time", "pixel")


def test_aplanado_descarta_solo_las_columnas_de_tierra() -> None:
    """Se eliminan las columnas sin serie completa, no filas de tiempo."""
    da = _grid()
    da[:, 0, 0] = np.nan

    matrix, _ = flatten_space(da)

    assert matrix.shape[0] == da.sizes["time"]
    assert matrix.shape[1] == da.sizes["latitude"] * da.sizes["longitude"] - 1
    assert np.isfinite(matrix).all()


def test_aplanado_no_submuestrea() -> None:
    """Todos los pixeles de agua llegan a la matriz (sin submuestreo)."""
    da = _grid()
    matrix, _ = flatten_space(da)

    assert matrix.shape[1] == int(water_mask(da).sum())
    assert not config.ALLOW_PIXEL_SUBSAMPLING


def test_ida_y_vuelta_reconstruye_el_campo() -> None:
    """Aplanar y desaplanar devuelve el campo original."""
    da = select_box(_grid(), config.BOXES["norte_chico"], exclude_north_edge=True)
    matrix, index = flatten_space(da)
    vuelta = unflatten_space(matrix, index, name=da.name).transpose(*da.dims)

    xr.testing.assert_allclose(vuelta, da)


def test_ida_y_vuelta_repone_la_tierra_como_nan() -> None:
    """Los pixeles descartados reaparecen como NaN, no como ceros."""
    da = _grid()
    da[:, 0, 0] = np.nan

    matrix, index = flatten_space(da)
    vuelta = unflatten_space(matrix, index, name=da.name)

    assert bool(vuelta.sel(latitude=da.latitude[0], longitude=da.longitude[0]).isnull().all())


def test_patron_espacial_se_desaplana_a_mapa() -> None:
    """Un vector de pesos espaciales (un EOF) vuelve a su malla 2-D."""
    da = select_box(_grid(), config.BOXES["norte_grande"])
    _, index = flatten_space(da)
    patron = np.arange(index.sizes["pixel"], dtype=float)

    mapa = unflatten_space(patron, index, name="eof1")

    assert set(mapa.dims) == {"latitude", "longitude"}
    assert mapa.sizes["latitude"] == da.sizes["latitude"]
