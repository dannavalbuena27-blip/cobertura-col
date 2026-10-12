"""Experimento: comparar modelos y variantes de preprocesado contra ESA WorldCover.

Uso:
    uv run python scripts/experimentos.py

Para cada zona de prueba:
  1. construye el compuesto RGB de Sentinel-2 una sola vez
  2. baja WorldCover una sola vez
  3. corre cada variante (modelo + preprocesado) y calcula exactitud / IoU

Asi se elige el mejor combinacion con datos, no a ojo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from cobertura_col import backends, descarga, inferencia, modelos, validacion
from cobertura_col.colores import info_modelo
from cobertura_col.zona import desde_bbox

# Zonas de prueba: una urbana y una de bosque
ZONAS = {
    "bogota_urbana": (-74.13, 4.67, -74.07, 4.72),
    "amazonia_bosque": (-74.60, 1.20, -74.40, 1.35),
}

# (etiqueta, modelo, kwargs de preparar_lote)
VARIANTES = [
    ("b0 g=1.0 (sin gamma)", "b0", {"gamma": 1.0}),
    ("b0 g=0.85", "b0", {"gamma": 0.85}),
    ("b0 g=0.7", "b0", {"gamma": 0.7}),
    ("b0 g=0.6", "b0", {"gamma": 0.6}),
    ("b0 g=0.5", "b0", {"gamma": 0.5}),
    ("b4 g=1.0", "b4", {"gamma": 1.0}),
    ("b4 g=0.7", "b4", {"gamma": 0.7}),
]

CACHE = Path("cache_experimentos")

_modelos_cache: dict[str, object] = {}
_backends_cache: dict[str, object] = {}


def obtener_compuesto(zona, bbox, nombre_zona: str):
    """Compuesto RGB con caché local para no re-descargar en cada ronda."""
    CACHE.mkdir(exist_ok=True)
    ruta = CACHE / f"{nombre_zona}.npz"
    if ruta.exists():
        datos = np.load(ruta, allow_pickle=False)
        compuesto = descarga.Compuesto(
            rgb=datos["rgb"],
            mascara=datos["mascara"].astype(bool),
            transform=__import__("rasterio").Affine(*datos["transform"]),
            crs=str(datos["crs"]),
            escenas=[],
            fraccion_valida=float(datos["fraccion"]),
        )
        print(f"  compuesto desde caché {ruta}")
        return compuesto

    compuesto = descarga.construir_compuesto(zona, max_nubes=40, max_escenas=5, meses=12)
    np.savez_compressed(
        ruta,
        rgb=compuesto.rgb,
        mascara=compuesto.mascara,
        transform=list(compuesto.transform)[:6],
        crs=np.str_(compuesto.crs),
        fraccion=np.float32(compuesto.fraccion_valida),
    )
    return compuesto


def obtener_backend(nombre: str):
    if nombre not in _modelos_cache:
        _modelos_cache[nombre] = modelos.cargar(nombre)
    modelo = _modelos_cache[nombre]
    if nombre not in _backends_cache:
        b, _ = backends.crear_backend("auto", modelo, nombre, 224, "GPU")
        _backends_cache[nombre] = b
    return _backends_cache[nombre]


def obtener_wc(bbox, compuesto, nombre_zona: str) -> np.ndarray:
    """WorldCover con caché local (mismas zonas que los compuestos)."""
    ruta = CACHE / f"{nombre_zona}_wc.npy"
    if ruta.exists():
        return np.load(ruta)
    wc = validacion.leer_worldcover(
        bbox, compuesto.transform, compuesto.crs, compuesto.rgb.shape[:2]
    )
    np.save(ruta, wc)
    return wc


def main() -> int:
    resultados = []
    for nombre_zona, bbox in ZONAS.items():
        print("=" * 72)
        print(f"ZONA {nombre_zona}: {bbox}")
        zona = desde_bbox(",".join(str(v) for v in bbox), nombre_zona)

        t = time.time()
        compuesto = obtener_compuesto(zona, bbox, nombre_zona)
        print(f"  compuesto en {time.time() - t:.1f} s")

        t = time.time()
        wc = obtener_wc(bbox, compuesto, nombre_zona)
        wc_comun = validacion.wc_a_comun(wc)
        print(f"  worldcover en {time.time() - t:.1f} s")

        # mascara sin dato igual que la CLI (nubes + fuera de la zona)
        mascara = compuesto.mascara

        for etiqueta, modelo_nombre, kwargs in VARIANTES:
            info = info_modelo(modelo_nombre)
            try:
                backend = obtener_backend(modelo_nombre)
            except Exception as exc:
                print(f"  [!] {etiqueta}: no se pudo cargar el modelo ({exc})")
                continue

            t = time.time()
            mapa = inferencia.inferir_mapa(
                compuesto.rgb,
                mascara,
                backend,
                ventana=224,
                solape=0.25,
                lote=8,
                progreso=lambda *a, **k: None,
                preparar=lambda corte, kw=kwargs: modelos.preparar_lote(corte, **kw),
            )
            dt = time.time() - t
            try:
                res = validacion.comparar(mapa, info, wc_comun)
            except RuntimeError as exc:
                print(f"  [!] {etiqueta}: {exc}")
                continue

            por_clase = {
                f["clase"]: f["iou"] for f in res["por_clase"] if f["iou"] != ""
            }
            fila = {
                "zona": nombre_zona,
                "variante": etiqueta,
                "exactitud": res["exactitud"],
                "iou_prom": res["iou_promedio"],
                "cobertura": res["cobertura_pct"],
                "seg": dt,
                "iou_urbano": por_clase.get("urbano", ""),
                "iou_bosque": por_clase.get("bosque", ""),
                "iou_pastizal": por_clase.get("pastizal", ""),
                "iou_agua": por_clase.get("agua", ""),
            }
            resultados.append(fila)
            print(
                f"  {etiqueta:<22} exactitud={res['exactitud']:.1%}  "
                f"IoU_prom={res['iou_promedio']:.3f}  cob={res['cobertura_pct']:.1f}%  "
                f"(urb {fila['iou_urbano']}, bosque {fila['iou_bosque']}, "
                f"past {fila['iou_pastizal']})  [{dt:.1f}s]"
            )

    print("=" * 72)
    print(
        f"{'variante':<22} {'zona':<18} {'exact':>7} {'IoUprom':>8} {'cob':>6} "
        f"{'urb':>6} {'bosque':>6} {'past':>6}"
    )
    for r in resultados:
        print(
            f"{r['variante']:<22} {r['zona']:<18} {r['exactitud']:>7.1%} "
            f"{(r['iou_prom'] or 0):>8.3f} {r['cobertura']:>5.1f}% "
            f"{str(r['iou_urbano']):>6} {str(r['iou_bosque']):>6} {str(r['iou_pastizal']):>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
