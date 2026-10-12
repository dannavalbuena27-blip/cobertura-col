"""Diagnostico: comparar distribucion de clases WorldCover vs modelo en una zona cacheada."""

from __future__ import annotations

import numpy as np
import rasterio

from cobertura_col import descarga, inferencia, modelos, validacion
from cobertura_col.backends import crear_backend
from cobertura_col.colores import info_modelo

NOMBRE = "amazonia_bosque"
GAMMA = 0.5


def main() -> int:
    datos = np.load(f"cache_experimentos/{NOMBRE}.npz")
    compuesto = descarga.Compuesto(
        rgb=datos["rgb"],
        mascara=datos["mascara"].astype(bool),
        transform=rasterio.Affine(*datos["transform"]),
        crs=str(datos["crs"]),
        escenas=[],
        fraccion_valida=float(datos["fraccion"]),
    )
    wc = np.load(f"cache_experimentos/{NOMBRE}_wc.npy")
    info = info_modelo("b0")

    wc_comun = validacion.wc_a_comun(wc)
    print("Distribucion WorldCover (esquema comun):")
    for i, clase in enumerate(validacion.CLASES_REPORTE, start=1):
        n = int((wc_comun == i).sum())
        print(f"  {clase:<12} {n:>9} ({100 * n / wc_comun.size:5.1f}%)")

    modelo = modelos.cargar("b0")
    backend, _ = crear_backend("auto", modelo, "b0", 224, "GPU")
    mapa = inferencia.inferir_mapa(
        compuesto.rgb,
        compuesto.mascara,
        backend,
        progreso=lambda *a, **k: None,
        preparar=lambda c: modelos.preparar_lote(c, gamma=GAMMA),
    )
    print("\nDistribucion modelo b0 (gamma 0.5):")
    total = int((mapa != 255).sum())
    for i, clase in enumerate(info["clases"]):
        n = int((mapa == i).sum())
        print(f"  {clase:<18} {n:>9} ({100 * n / max(1, total):5.1f}%)")

    res = validacion.comparar(mapa, info, wc_comun)
    print(f"\nExactitud: {res['exactitud']:.1%}  IoU prom: {res['iou_promedio']:.3f}")
    print(f"{'clase':<12} {'IoU':>7} {'prec':>7} {'rec':>7} {'WC px':>9} {'mod px':>9}")
    for f in res["por_clase"]:
        print(
            f"{f['clase']:<12} {str(f['iou']):>7} {str(f['precision']):>7} "
            f"{str(f['recall']):>7} {f['pixeles_worldcover']:>9} {f['pixeles_modelo']:>9}"
        )
    print("\nMatriz (filas=WorldCover, cols=modelo):")
    print("        " + "".join(f"{c[:6]:>8}" for c in validacion.CLASES_REPORTE))
    for i, c in enumerate(validacion.CLASES_REPORTE):
        print(f"{c[:6]:>8}" + "".join(f"{int(v):>8}" for v in res["cm"][i]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
