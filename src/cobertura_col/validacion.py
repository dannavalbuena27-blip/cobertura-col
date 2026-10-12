"""Validacion opcional del mapa contra ESA WorldCover 10 m (producto oficial de la ESA).

El producto es gratuito y esta en un bucket S3 publico. Los archivos se nombran
por teselas de 3x3 grados: ESA_WorldCover_10m_2021_v200_N04W076_Map.tif

Ambos mapas se agregan a un esquema comun de 6 clases para poder compararlos
(agua, bosque, cultivos, pastizal, urbano, desnudo).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .colores import COLORES_COMUN, ESQUEMA_COMUN, WORLDCOVER_A_COMUN, info_modelo
from .descarga import _leer_reproyectar, avisar
from rasterio.warp import Resampling

S3_WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{rejilla}_Map.tif"
)
CLASES_REPORTE = ESQUEMA_COMUN[1:]  # sin_dato no se reporta


def _nombre_rejilla(lat: float, lon: float) -> str:
    """Ej: (-4.0, -76.5) -> 'S04W076' (esquina superior izquierda de la tesela)."""
    ns = "N" if lat >= 0 else "S"
    we = "E" if lon >= 0 else "W"
    return f"{ns}{abs(int(lat)):02d}{we}{abs(int(lon)):03d}"


def teselas_para_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """Las teselas se nombran por su esquina inferior izquierda (N06W075 = lat 6..9, lon -75..-72)."""
    minlon, minlat, maxlon, maxlat = bbox
    longsitudes = []
    lon = 3 * math.floor(minlon / 3)
    while lon < maxlon:
        longsitudes.append(lon)
        lon += 3
    lats = []
    lat = 3 * math.floor(minlat / 3)
    while lat < maxlat:
        lats.append(lat)
        lat += 3
    return [_nombre_rejilla(lat, lon) for lat in lats for lon in longsitudes]


def leer_worldcover(bbox, transform, crs, forma) -> np.ndarray:
    """Descarga por rangos HTTP las teselas necesarias y las acomoda en nuestra grilla."""
    dst = np.zeros(forma, dtype="float32")
    for rejilla in teselas_para_bbox(bbox):
        url = S3_WORLDCOVER.format(rejilla=rejilla)
        try:
            capa = _leer_reproyectar(
                url, transform, crs, forma, Resampling.nearest, nodata_src=0.0, nodata_dst=0.0
            )
        except Exception as exc:
            avisar(f"No se pudo leer la tesela {rejilla} ({exc}); se omite.")
            continue
        dst = np.where(capa > 0, capa, dst)
    return np.rint(dst).astype(np.uint8)


def _a_comun_modelo(mapa: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mapa de clases del modelo -> esquema comun + mascara de clases excluidas."""
    lut = np.zeros(len(info["clases"]) + 1, dtype=np.uint8)  # el indice len = excluida
    excluida = len(info["clases"])
    for i, nombre in enumerate(info["a_comun"]):
        lut[i] = ESQUEMA_COMUN.index(nombre) if nombre else excluida
    salida = np.full(mapa.shape, excluida, dtype=np.uint8)
    valido = mapa != 255
    salida[valido] = lut[mapa[valido]]
    return salida, salida == excluida


def wc_a_comun(wc: np.ndarray) -> np.ndarray:
    """Codigos de WorldCover (10, 20, ... 100) -> esquema comun (1..6)."""
    lut = np.zeros(101, dtype=np.uint8)
    for codigo, nombre in WORLDCOVER_A_COMUN.items():
        lut[codigo] = ESQUEMA_COMUN.index(nombre)
    salida = np.zeros(wc.shape, dtype=np.uint8)
    en_rango = (wc > 0) & (wc < len(lut))
    salida[en_rango] = lut[wc[en_rango]]
    return salida


def comparar(mapa: np.ndarray, info: dict, wc_comun: np.ndarray) -> dict:
    """Metricas de un mapa de clases del modelo contra WorldCover ya en esquema comun."""
    from sklearn.metrics import confusion_matrix

    modelo_comun, excluida = _a_comun_modelo(mapa, info)
    evaluable = (~excluida) & (wc_comun > 0)

    n_eval = int(evaluable.sum())
    if n_eval == 0:
        raise RuntimeError("No hay pixeles comparables entre el modelo y WorldCover.")

    # Ambos vectores pasan al rango 0..5 (sin_dato = indice 0 del esquema)
    y_verdadero = wc_comun[evaluable].astype(np.int16) - 1
    y_predicho = modelo_comun[evaluable].astype(np.int16) - 1

    cm = confusion_matrix(y_verdadero, y_predicho, labels=list(range(len(CLASES_REPORTE))))

    total = cm.sum()
    exactitud = float(np.trace(cm) / total)
    filas = []
    ious = []
    for i, clase in enumerate(CLASES_REPORTE):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not math.isnan(iou):
            ious.append(iou)
        filas.append(
            {
                "clase": clase,
                "iou": round(iou, 4) if not math.isnan(iou) else "",
                "precision": round(precision, 4) if not math.isnan(precision) else "",
                "recall": round(recall, 4) if not math.isnan(recall) else "",
                "pixeles_worldcover": int(cm[i, :].sum()),
                "pixeles_modelo": int(cm[:, i].sum()),
            }
        )

    return {
        "exactitud": exactitud,
        "iou_promedio": float(np.mean(ious)) if ious else None,
        "n_eval": n_eval,
        "cobertura_pct": 100.0 * n_eval / int(mapa.size),
        "cm": cm,
        "por_clase": filas,
    }


def validar(
    mapa: np.ndarray,
    compuesto,
    info: dict,
    directorio: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> dict:
    """Compara el mapa del modelo con ESA WorldCover y escribe metricas."""
    directorio.mkdir(parents=True, exist_ok=True)
    avisar("Descargando ESA WorldCover 10 m para la misma zona ...")

    wc = leer_worldcover(bbox_wgs84, compuesto.transform, compuesto.crs, mapa.shape)
    res = comparar(mapa, info, wc_a_comun(wc))
    cm, exactitud, filas, n_eval = res["cm"], res["exactitud"], res["por_clase"], res["n_eval"]
    ious = [f["iou"] for f in filas if f["iou"] != ""]

    cobertura = res["cobertura_pct"]
    resumen = {
        "exactitud_global": round(exactitud, 4),
        "iou_promedio": round(float(np.mean(ious)), 4) if ious else None,
        "pixeles_evaluados": n_eval,
        "cobertura_evaluada_pct": round(cobertura, 2),
        "clases_excluidas_modelo": [
            c for c, m in zip(info["clases"], info["a_comun"]) if m is None
        ],
        "referencia": "ESA WorldCover 10 m v200 (2021)",
        "modelo": info["nombre"],
        "nota": (
            "WorldCover es de 2021 y la imagen Sentinel-2 es reciente: parte de la "
            "diferencia se debe al cambio real del territorio en ese lapso."
        ),
    }

    # CSV de metricas por clase
    ruta_csv = directorio / "validacion.csv"
    with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    (directorio / "validacion.json").write_text(
        json.dumps({"resumen": resumen, "por_clase": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Matriz de confusion
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASES_REPORTE)), CLASES_REPORTE, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASES_REPORTE)), CLASES_REPORTE)
    ax.set_xlabel("Predicho (modelo)")
    ax.set_ylabel("Verdadero (WorldCover)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = int(cm[i, j])
            if valor:
                ax.text(j, i, f"{valor}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Matriz de confusion (exactitud {exactitud:.1%})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(directorio / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    avisar(f"Exactitud global vs WorldCover: {exactitud:.1%}  (IoU promedio: {resumen['iou_promedio']})")
    return {"resumen": resumen, "por_clase": filas}
