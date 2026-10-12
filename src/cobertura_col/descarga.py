"""Descarga de imagenes Sentinel-2 y construccion de un compuesto RGB libre de nubes.

Flujo:
  1. Buscar escenas Sentinel-2 L2A en el catalogo STAC publico de Element84 (AWS).
  2. Elegir las escenas con menos nubes de los ultimos meses.
  3. Definir una grilla comun en el sistema UTM de la zona (10 m).
  4. Leer las bandas roja/verde/azul (B04/B03/B02) y la mascara SCL de cada escena,
     recortandolas a la zona y reproyectandolas a la grilla comun.
  5. Quitar nubes y sombras con la mascara SCL y promediar por pixel (mediana),
     que es la forma clasica de lidiar con cielos nublados como los de Colombia.
  6. Convertir a RGB de 8 bits con un estiramiento percentil (2%-98%).
"""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# Configuracion de GDAL para lecturas HTTP por rangos (debe ir antes de importar rasterio)
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

import numpy as np
import rasterio
from pystac_client import Client
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .zona import Zona

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLECCION = "sentinel-2-l2a"
RESOLUCION = 10.0  # metros por pixel

# Clases de la mascara SCL que consideramos no observables
SCL_INVALIDAS = {0, 1, 3, 8, 9, 10, 11}  # nodata, saturado, nube alta, nube media,
# cirrus, sombra de nube, nieve


@dataclass
class Compuesto:
    """Resultado de la etapa de descarga."""

    rgb: np.ndarray  # (H, W, 3) uint8
    mascara: np.ndarray  # (H, W) bool, True = pixel observado (sin nube)
    transform: Affine
    crs: str  # EPSG:xxxxx
    escenas: list[dict] = field(default_factory=list)
    fraccion_valida: float = 0.0
    bbox_utm: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]


def avisar(mensaje: str) -> None:
    print(f"  [i] {mensaje}")


# --------------------------------------------------------------------- STAC


def buscar_escenas(
    bbox: tuple[float, float, float, float],
    max_nubes: float = 40.0,
    max_escenas: int = 5,
    meses: int = 12,
) -> list:
    """Devuelve las escenas menos nubladas de los ultimos `meses` meses."""
    cliente = Client.open(STAC_URL)
    inicio = (datetime.now(timezone.utc) - timedelta(days=30 * meses)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fin = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = dict(collections=[COLECCION], bbox=list(bbox), datetime=f"{inicio}/{fin}")

    items = []
    # Intento 1: filtro y orden en el servidor (extensiones query/sortby)
    try:
        items = list(
            cliente.search(
                **params,
                query={"eo:cloud_cover": {"lt": max_nubes}},
                sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                max_items=max_escenas,
            ).items()
        )
    except Exception:
        items = []

    # Intento 2: traer las primeras y ordenar en local
    if not items:
        candidatos = list(cliente.search(**params, max_items=200).items())
        candidatos = [
            i for i in candidatos if i.properties.get("eo:cloud_cover", 100.0) <= max_nubes
        ]
        candidatos.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100.0))
        items = candidatos[:max_escenas]

    # Ultimo recurso: aceptar cualquier nubosidad
    if not items:
        avisar(
            f"No hubo escenas con <= {max_nubes:.0f}% de nubes en {meses} meses; "
            "buscando con cualquier nubosidad."
        )
        candidatos = list(cliente.search(**params, max_items=200).items())
        candidatos.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100.0))
        items = candidatos[:max_escenas]
        if not items:
            raise RuntimeError(
                "No se encontraron escenas Sentinel-2 para esa zona. "
                "Revisa la conexion o amplia el rango de fechas."
            )
    return items


# ------------------------------------------------------------------- grilla


def grilla_zona(zona: Zona, resolucion: float = RESOLUCION) -> tuple[Affine, str, int, int]:
    """Grilla regular de salida en el UTM de la zona que contiene al bounding box."""
    epsg = zona.epsg_utm()
    crs = f"EPSG:{epsg}"
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Densificamos los bordes para que la transformacion entre zonas UTM sea precisa
    minlon, minlat, maxlon, maxlat = zona.bbox
    xs, ys = [], []
    for t in np.linspace(0.0, 1.0, 21):
        for lon, lat in (
            (minlon + t * (maxlon - minlon), minlat),
            (minlon + t * (maxlon - minlon), maxlat),
            (minlon, minlat + t * (maxlat - minlat)),
            (maxlon, minlat + t * (maxlat - minlat)),
        ):
            x, y = tf.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    ancho = max(1, int(math.ceil((maxx - minx) / resolucion)))
    alto = max(1, int(math.ceil((maxy - miny) / resolucion)))
    transform = from_origin(minx, maxy, resolucion, resolucion)
    return transform, crs, ancho, alto


# ------------------------------------------------------- lectura de bandas


def _leer_reproyectar(
    href: str,
    transform_dst: Affine,
    crs_dst: str,
    shape_dst: tuple[int, int],
    resampling: Resampling,
    nodata_src: float,
    nodata_dst: float,
) -> np.ndarray:
    """Lee la ventana de un GeoTIFF remoto (peticiones HTTP por rangos) y la reproyecta."""
    alto, ancho = shape_dst
    bounds_dst = array_bounds(alto, ancho, transform_dst)

    destino = np.full(shape_dst, nodata_dst, dtype="float32")
    with rasterio.open(f"/vsicurl/{href}") as src:
        bounds_src = transform_bounds(crs_dst, src.crs, *bounds_dst, densify_pts=21)
        win = from_bounds(*bounds_src, transform=src.transform)
        win = Window(
            col_off=int(math.floor(win.col_off)),
            row_off=int(math.floor(win.row_off)),
            width=int(math.ceil(win.width)) + 1,
            height=int(math.ceil(win.height)) + 1,
        )
        datos = src.read(
            1, window=win, boundless=True, fill_value=nodata_src, out_dtype="float32"
        )
        transform_src = src.window_transform(win)
        crs_src = src.crs

    # La lectura boundless genera un dataset virtual sin georreferencia en rasterio;
    # el aviso solo ensucia la salida, aqui el transform explicito se pasa abajo.
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        reproject(
            source=datos,
            destination=destino,
            src_transform=transform_src,
            src_crs=crs_src,
            src_nodata=nodata_src,
            dst_transform=transform_dst,
            dst_crs=crs_dst,
            dst_nodata=nodata_dst,
            resampling=resampling,
        )
    return destino


# ---------------------------------------------------------------- compuesto


def _procesar_escena(item, transform: Affine, crs: str, forma: tuple[int, int]) -> np.ndarray | None:
    """Lee una escena completa (3 bandas + SCL) y devuelve el stack (3,H,W) con NaN donde no hay dato."""
    try:
        href_r = item.assets["red"].href
        href_g = item.assets["green"].href
        href_b = item.assets["blue"].href
        href_scl = item.assets["scl"].href
    except KeyError as exc:
        avisar(f"La escena {item.id} no trae la banda esperada ({exc}); se omite.")
        return None

    try:
        dn = [
            _leer_reproyectar(
                href, transform, crs, forma, Resampling.bilinear, nodata_src=0.0, nodata_dst=0.0
            )
            for href in (href_r, href_g, href_b)
        ]
        scl = _leer_reproyectar(
            href_scl, transform, crs, forma, Resampling.nearest, nodata_src=0.0, nodata_dst=0.0
        )
    except Exception as exc:
        avisar(f"No se pudo leer la escena {item.id} ({exc}); se omite.")
        return None

    observado = ~np.isin(np.rint(scl).astype(np.int16), list(SCL_INVALIDAS))
    capas = []
    for banda in dn:
        ref = banda * 0.0001 - 0.1  # factor de escala/offset de Sentinel-2 L2A
        ref = np.where(banda > 0, ref, np.nan)
        capas.append(np.where(observado, ref, np.nan))
    return np.stack(capas, axis=0)


def construir_compuesto(
    zona: Zona,
    max_nubes: float = 40.0,
    max_escenas: int = 5,
    meses: int = 12,
    transform: Affine | None = None,
    crs: str | None = None,
    forma: tuple[int, int] | None = None,
) -> Compuesto:
    """Descarga las escenas y devuelve un RGB de 8 bits mediano entre escenas."""
    if transform is None or crs is None or forma is None:
        transform, crs, ancho, alto = grilla_zona(zona)
        forma = (alto, ancho)
    else:
        ancho, alto = forma[1], forma[0]

    avisar(f"Buscando escenas Sentinel-2 en {STAC_URL} ...")
    items = buscar_escenas(zona.bbox, max_nubes=max_nubes, max_escenas=max_escenas, meses=meses)
    avisar(f"{len(items)} escena(s) seleccionada(s):")
    for it in items:
        nubes = it.properties.get("eo:cloud_cover", float("nan"))
        fecha = it.properties.get("datetime", "")[:10]
        avisar(f"  - {it.id}  fecha={fecha}  nubes={nubes:.1f}%")

    avisar("Leyendo bandas B04/B03/B02 y mascara SCL (en paralelo) ...")
    stack = []  # lista de (3, H, W) float32 con NaN en pixeles invalidos
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(items)))) as pool:
        futuros = {
            pool.submit(_procesar_escena, it, transform, crs, forma): it for it in items
        }
        for futuro in as_completed(futuros):
            resultado = futuro.result()
            if resultado is not None:
                stack.append(resultado)

    if not stack:
        raise RuntimeError("Ninguna escena pudo leerse. Revisa la conexion.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixeles sin datos en todas las escenas
        media = np.nanmedian(np.stack(stack, axis=0), axis=0)  # (3, H, W)

    mascara = ~np.isnan(media[0])
    fraccion = float(mascara.mean()) if mascara.size else 0.0
    avisar(f"Pixeles observados tras el compuesto mediano: {fraccion * 100:.1f}%")
    if fraccion < 0.8:
        avisar(
            "Aviso: menos del 80% del area quedo libre de nubes; "
            "el mapa tendra huecos (se marcan como 'sin dato')."
        )

    # Estiramiento percentil 2%-98% por banda -> RGB uint8
    rgb = np.zeros((forma[0], forma[1], 3), dtype=np.uint8)
    for i in range(3):
        valores = media[i][mascara]
        if valores.size == 0:
            continue
        lo, hi = np.percentile(valores, [2.0, 98.0])
        if hi - lo < 1e-6:
            lo, hi = float(valores.min()), float(valores.max())
            if hi - lo < 1e-6:
                hi = lo + 1.0
        norm = np.clip((media[i] - lo) / (hi - lo), 0.0, 1.0)
        rgb[..., i] = (norm * 255.0).astype(np.uint8)

    metadatos = [
        {
            "id": it.id,
            "fecha": it.properties.get("datetime", ""),
            "nubes_pct": it.properties.get("eo:cloud_cover"),
            "url": it.assets.get("red").href if "red" in it.assets else None,
        }
        for it in items
    ]

    return Compuesto(
        rgb=rgb,
        mascara=mascara,
        transform=transform,
        crs=crs,
        escenas=metadatos,
        fraccion_valida=fraccion,
        bbox_utm=tuple(array_bounds(forma[0], forma[1], transform)),  # type: ignore[arg-type]
    )
