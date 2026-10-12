"""Salidas del programa: GeoTIFF, mapa PNG con leyenda, areas por clase y metadatos."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin interfaz grafica, solo archivos
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform


def recortar_a_geometria(mapa: np.ndarray, geometria, transform, crs: str) -> np.ndarray:
    """Pone 255 (sin dato) fuera del poligono indicado en el GeoJSON."""
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geom_utm = shapely_transform(lambda x, y, z=None: tf.transform(x, y), geometria)
    dentro = geometry_mask(
        [mapping(geom_utm)], out_shape=mapa.shape, transform=transform, invert=True
    )
    mapa = mapa.copy()
    mapa[~dentro] = 255
    return mapa


def escribir_geotiff(ruta: Path, mapa: np.ndarray, transform, crs: str, colores: list[str]) -> None:
    """GeoTIFF uint8 con colormap embebido, listo para abrir en QGIS."""
    colormap = {}
    for i, hexa in enumerate(colores):
        r, g, b = (int(hexa[j : j + 2], 16) for j in (1, 3, 5))
        colormap[i] = (r, g, b, 255)
    colormap[255] = (0, 0, 0, 0)  # sin dato transparente

    with rasterio.open(
        ruta,
        "w",
        driver="GTiff",
        height=mapa.shape[0],
        width=mapa.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=255,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(mapa, 1)
        dst.write_colormap(1, colormap)


def calcular_areas(mapa: np.ndarray, transform, clases: list[str]) -> list[dict]:
    """Hectareas y porcentaje por clase (se excluye el 255 = sin dato)."""
    area_pixel_m2 = abs(transform.a * transform.e)
    validos = mapa[mapa != 255]
    total = int(validos.size)
    filas = []
    for idx, nombre in enumerate(clases):
        conteo = int((validos == idx).sum())
        filas.append(
            {
                "clase": nombre,
                "pixeles": conteo,
                "hectareas": round(conteo * area_pixel_m2 / 10_000.0, 3),
                "porcentaje": round(100.0 * conteo / total, 2) if total else 0.0,
            }
        )
    sin_dato = int((mapa == 255).sum())
    filas.append(
        {
            "clase": "sin dato (nubes/limites)",
            "pixeles": sin_dato,
            "hectareas": round(sin_dato * area_pixel_m2 / 10_000.0, 3),
            "porcentaje": round(100.0 * sin_dato / mapa.size, 2),
        }
    )
    return filas


def escribir_areas(ruta_csv: Path, ruta_json: Path, filas: list[dict]) -> None:
    with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)
    ruta_json.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")


def escribir_mapa_png(
    ruta: Path,
    rgb: np.ndarray,
    mapa: np.ndarray,
    clases: list[str],
    colores: list[str],
    titulo: str,
) -> None:
    """Comparacion lado a lado: color verdadero Sentinel-2 vs mapa de cobertura."""
    cmap = ListedColormap(colores)
    cmap.set_bad(alpha=0.0)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 7))
    ejes[0].imshow(rgb)
    ejes[0].set_title("Sentinel-2 (color verdadero)")
    ejes[0].axis("off")

    salida = np.ma.masked_where(mapa == 255, mapa)
    im = ejes[1].imshow(salida, cmap=cmap, vmin=0, vmax=len(clases) - 1, interpolation="nearest")
    ejes[1].set_title("Cobertura terrestre (modelo de IA)")
    ejes[1].axis("off")

    leyenda = [Patch(facecolor=colores[i], label=clases[i]) for i in range(len(clases))]
    leyenda.append(Patch(facecolor="#000000", alpha=0.25, label="sin dato"))
    fig.legend(handles=leyenda, loc="lower center", ncol=min(4, len(clases) + 1), frameon=False)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def escribir_metadatos(ruta: Path, datos: dict) -> None:
    datos = dict(datos)
    datos["fecha_ejecucion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def escribir_todo(
    directorio: Path,
    zona_nombre: str,
    compuesto,
    mapa: np.ndarray,
    info: dict,
    extra: dict,
) -> dict:
    """Escribe todos los archivos de salida y devuelve sus rutas."""
    directorio.mkdir(parents=True, exist_ok=True)
    clases = info["clases"]
    colores = info["colores"]

    rutas = {}
    rutas["geotiff"] = str(directorio / "cobertura.tif")
    escribir_geotiff(Path(rutas["geotiff"]), mapa, compuesto.transform, compuesto.crs, colores)

    rutas["png"] = str(directorio / "mapa.png")
    escribir_mapa_png(
        Path(rutas["png"]),
        compuesto.rgb,
        mapa,
        clases,
        colores,
        titulo=f"{zona_nombre} - {info['nombre']}",
    )

    filas = calcular_areas(mapa, compuesto.transform, clases)
    rutas["areas_csv"] = str(directorio / "areas.csv")
    rutas["areas_json"] = str(directorio / "areas.json")
    escribir_areas(Path(rutas["areas_csv"]), Path(rutas["areas_json"]), filas)

    metadatos = {
        "zona": zona_nombre,
        "crs": compuesto.crs,
        "forma_pixeles": list(mapa.shape),
        "resolucion_m": abs(compuesto.transform.a),
        "fraccion_observada": round(compuesto.fraccion_valida, 4),
        "escenas": compuesto.escenas,
        "modelo": info["nombre"],
        "modelo_id": info["id"],
        "clases": clases,
        **extra,
    }
    rutas["metadata"] = str(directorio / "metadata.json")
    escribir_metadatos(Path(rutas["metadata"]), metadatos)
    return rutas
