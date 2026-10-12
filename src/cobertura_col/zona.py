"""Definicion y validacion de la zona de estudio.

Una zona se puede indicar de tres formas:
  * --bbox minlon,minlat,maxlon,maxlat   (coordenadas geograficas en grados)
  * --coords lat,lon --radio-km 3        (punto central + radio)
  * --geojson archivo.geojson            (poligono o conjunto de poligonos)

Toda zona se valida contra un bounding box aproximado de Colombia para evitar
consultas que no tendrian sentido.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

# Bounding box aproximado de Colombia continental e insular (minlon, minlat, maxlon, maxlat)
LIM_COLOMBIA = (-79.6, -4.3, -66.9, 13.6)


@dataclass
class Zona:
    """Zona de estudio en coordenadas geograficas (EPSG:4326)."""

    nombre: str
    bbox: tuple[float, float, float, float]  # minlon, minlat, maxlon, maxlat
    geometria: object | None = None  # shapely geometry opcional (para recortar)

    def __post_init__(self) -> None:
        minlon, minlat, maxlon, maxlat = self.bbox
        if not (-180.0 <= minlon < maxlon <= 180.0 and -90.0 <= minlat < maxlat <= 90.0):
            raise ValueError(f"Bounding box invalido: {self.bbox}")
        if maxlon - minlon > 2.0 or maxlat - minlat > 2.0:
            # ~2 grados es un area enorme; el programa esta pensado para zonas locales
            raise ValueError(
                "La zona es demasiado grande (mas de 2 grados por lado). "
                "Usa una zona mas pequena; el limite practico son ~50 km por lado."
            )
        if not box(*self.bbox).intersects(box(*LIM_COLOMBIA)):
            raise ValueError(
                f"La zona {self.bbox} queda fuera de Colombia. "
                f"El pais esta aproximadamente entre {LIM_COLOMBIA}."
            )

    # ------------------------------------------------------------------ propiedades
    @property
    def centro(self) -> tuple[float, float]:
        minlon, minlat, maxlon, maxlat = self.bbox
        return ((minlon + maxlon) / 2.0, (minlat + maxlat) / 2.0)

    @property
    def ancho_grados(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def alto_grados(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def epsg_utm(self) -> int:
        """EPSG del sistema UTM que corresponde al centro de la zona (Colombia esta en el hemisferio norte)."""
        lon, lat = self.centro
        zona = int(math.floor((lon + 180.0) / 6.0)) + 1
        return 32600 + zona  # 326xx = UTM norte (WGS84)

    def etiqueta(self) -> str:
        """Nombre corto de la zona, util para nombrar carpetas de salida."""
        if self.nombre:
            return self._sanear(self.nombre)
        lon, lat = self.centro
        return f"zona_{lat:.3f}_{lon:.3f}".replace("-", "m")

    @staticmethod
    def _sanear(texto: str) -> str:
        keep = []
        for ch in texto.strip():
            keep.append(ch if (ch.isalnum() or ch in "-_") else "_")
        return "".join(keep).strip("_") or "zona"


# ---------------------------------------------------------------------- constructores


def desde_bbox(bbox: str, nombre: str = "") -> Zona:
    """--bbox minlon,minlat,maxlon,maxlat"""
    partes = [p.strip() for p in bbox.split(",")]
    if len(partes) != 4:
        raise ValueError("--bbox espera 4 valores: minlon,minlat,maxlon,maxlat")
    minlon, minlat, maxlon, maxlat = (float(p) for p in partes)
    if maxlon <= minlon or maxlat <= minlat:
        raise ValueError("En --bbox el maximo debe ser mayor que el minimo en ambos ejes.")
    return Zona(nombre=nombre, bbox=(minlon, minlat, maxlon, maxlat))


def desde_coords(lat: float, lon: float, radio_km: float, nombre: str = "") -> Zona:
    """--coords lat,lon --radio-km r  -> bounding box cuadrado que contiene al circulo."""
    if not (-4.3 <= lat <= 13.6 and -79.6 <= lon <= -66.9):
        raise ValueError(f"El punto ({lat}, {lon}) queda fuera de Colombia.")
    if radio_km <= 0:
        raise ValueError("--radio-km debe ser positivo.")
    dlat = radio_km / 111.32
    dlon = radio_km / (111.32 * max(0.1, math.cos(math.radians(lat))))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    geom = Point(lon, lat).buffer(max(dlon, dlat), resolution=32)
    return Zona(nombre=nombre, bbox=bbox, geometria=geom)


def desde_geojson(ruta: str | Path, nombre: str = "") -> Zona:
    """--geojson archivo.geojson  (acepta FeatureCollection, Feature o geometry)."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo GeoJSON: {ruta}")
    datos = json.loads(ruta.read_text(encoding="utf-8"))

    if datos.get("type") == "FeatureCollection":
        geometrias = [shape(f["geometry"]) for f in datos.get("features", []) if f.get("geometry")]
    elif datos.get("type") == "Feature":
        geometrias = [shape(datos["geometry"])]
    else:
        geometrias = [shape(datos)]

    if not geometrias:
        raise ValueError("El GeoJSON no contiene geometrias.")

    geom = unary_union(geometrias)
    minlon, minlat, maxlon, maxlat = geom.bounds
    return Zona(
        nombre=nombre or ruta.stem,
        bbox=(float(minlon), float(minlat), float(maxlon), float(maxlat)),
        geometria=geom,
    )
