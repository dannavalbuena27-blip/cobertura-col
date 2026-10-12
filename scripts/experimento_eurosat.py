"""Experimento: clasificador EuroSAT (Swin, entrenado en Sentinel-2) como modelo alternativo.

EuroSAT esta entrenado con Sentinel-2 a 10 m, asi que coincide con nuestra imagen
de entrada en sensor y escala. Devuelve UNA clase por recorte (clasificacion, no
segmentacion), asi que se prueban varios tamanos de recorte.

Uso:
    uv run python scripts/experimento_eurosat.py
"""

from __future__ import annotations

import time

import numpy as np
import rasterio

from cobertura_col import descarga, inferencia, modelos, validacion

MODELO_ID = "brysanicks/swin-tiny-patch4-window7-224-finetuned-eurosat"

# Etiqueta EuroSAT -> esquema comun
A_COMUN = {
    "AnnualCrop": "cultivos",
    "Forest": "bosque",
    "HerbaceousVegetation": "pastizal",
    "Highway": "urbano",
    "Industrial": "urbano",
    "Pasture": "pastizal",
    "PermanentCrop": "cultivos",
    "Residential": "urbano",
    "River": "agua",
    "SeaLake": "agua",
}

ZONAS = ["bogota_urbana", "amazonia_bosque"]
PRUEBAS = [  # (ventana, gamma)
    (224, 1.0),
    (224, 0.7),
    (96, 1.0),
    (96, 0.7),
]


class ClasificadorBackend:
    """Adapta un modelo de clasificacion al contrato de inferencia (ventana completa).

    Usa OpenVINO/GPU si la conversion funciona (rapido) y si no PyTorch/CPU.
    """

    def __init__(self, modelo, ventana: int):
        import torch

        self._torch = torch
        self._modelo = modelo
        self._ventana = ventana
        self._compilado = None
        try:
            import openvino as ov

            from pathlib import Path

            Path("modelos_cache").mkdir(exist_ok=True)
            ruta = Path(f"modelos_cache/eurosat_v{ventana}.xml")
            core = ov.Core()
            if ruta.exists():
                ir = core.read_model(str(ruta))
            else:
                ir = ov.convert_model(modelo, example_input=torch.zeros(1, 3, ventana, ventana))
                ov.save_model(ir, str(ruta), compress_to_fp16=True)
            self._compilado = core.compile_model(
                ir, "GPU", config={"CACHE_DIR": "modelos_cache/ov_cache"}
            )
            self._solicitud = self._compilado.create_infer_request()
        except Exception as exc:
            print(f"    (OpenVINO no disponible para el clasificador: {exc}; CPU con PyTorch)")

    def inferir(self, pixel_values: np.ndarray) -> np.ndarray:
        n = pixel_values.shape[0]
        if self._compilado is not None:
            salida = self._compilado.output(0)
            clase = np.concatenate(
                [
                    np.asarray(self._solicitud.infer({0: pixel_values[i : i + 1]})[salida])
                    .argmax(axis=1)
                    for i in range(n)
                ]
            )
        else:
            with self._torch.no_grad():
                logits = self._modelo(
                    pixel_values=self._torch.from_numpy(pixel_values)
                ).logits.cpu().numpy()
            clase = logits.argmax(axis=1)  # (N,)
        return np.repeat(
            np.repeat(clase[:, None, None], self._ventana, axis=1), self._ventana, axis=2
        )


def cargar_compuesto(nombre: str):
    datos = np.load(f"cache_experimentos/{nombre}.npz")
    return descarga.Compuesto(
        rgb=datos["rgb"],
        mascara=datos["mascara"].astype(bool),
        transform=rasterio.Affine(*datos["transform"]),
        crs=str(datos["crs"]),
        escenas=[],
        fraccion_valida=float(datos["fraccion"]),
    )


def main() -> int:
    from transformers import AutoModelForImageClassification

    print(f"Cargando {MODELO_ID} ...")
    modelo = AutoModelForImageClassification.from_pretrained(MODELO_ID)
    modelo.eval()
    etiquetas = [modelo.config.id2label[i] for i in sorted(modelo.config.id2label)]
    print(f"Clases: {etiquetas}")
    info = {
        "clases": etiquetas,
        "a_comun": [A_COMUN[e] for e in etiquetas],
        "nombre": f"EuroSAT Swin-T ({MODELO_ID.split('/')[0]})",
    }

    resultados = []
    backends_cache = {}
    for nombre_zona in ZONAS:
        compuesto = cargar_compuesto(nombre_zona)
        wc = np.load(f"cache_experimentos/{nombre_zona}_wc.npy")
        wc_comun = validacion.wc_a_comun(wc)
        print("=" * 70)
        print(f"ZONA {nombre_zona}")

        for ventana, gamma in PRUEBAS:
            if ventana not in backends_cache:
                backends_cache[ventana] = ClasificadorBackend(modelo, ventana)
            backend = backends_cache[ventana]
            t = time.time()
            try:
                mapa = inferencia.inferir_mapa(
                    compuesto.rgb,
                    compuesto.mascara,
                    backend,
                    ventana=ventana,
                    solape=0.25,
                    lote=8,
                    progreso=lambda *a, **k: None,
                    preparar=lambda c, g=gamma: modelos.preparar_lote(c, gamma=g),
                )
                res = validacion.comparar(mapa, info, wc_comun)
            except Exception as exc:
                print(f"  ventana={ventana} gamma={gamma}: ERROR {exc}")
                continue
            dt = time.time() - t
            por_clase = {f["clase"]: f["iou"] for f in res["por_clase"] if f["iou"] != ""}
            print(
                f"  vent={ventana:<3} g={gamma:<4} exactitud={res['exactitud']:.1%}  "
                f"IoU_prom={res['iou_promedio']:.3f}  "
                f"(urb {por_clase.get('urbano', '')}, bosque {por_clase.get('bosque', '')}, "
                f"past {por_clase.get('pastizal', '')})  [{dt:.1f}s]"
            )
            resultados.append((nombre_zona, ventana, gamma, res))

    print("=" * 70)
    for zona, vent, gam, res in resultados:
        print(
            f"{zona:<18} vent={vent:<3} g={gam:<4} exact={res['exactitud']:.1%} "
            f"IoU_prom={res['iou_promedio']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
