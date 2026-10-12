"""Carga de los modelos preentrenados de codigo abierto.

Se usan dos modelos de Hugging Face, ambos SegFormer entrenados para segmentacion
semantica de imagenes satelitales:

  * b0 (por defecto): florian-morel22/segformer-b0-deepglobe-land-cover
      6 clases de cobertura (DeepGlobe): urbano, agricultura, pastizal, bosque,
      agua, terreno desnudo + una clase "sin clasificar". Ligero (~3.7M params).
  * b4 (opcional): nave1616/SegFormer-landcover-FT
      9 clases, incluye carreteras y edificaciones. Mas pesado (~64M params).

Ambos reciben recortes RGB de 224x224 y devuelven un mapa de clases del mismo tamano.

Nota: ninguno de los repos incluye preprocessor_config.json, asi que la normalizacion
se aplica a mano con los valores estandar de SegFormer/ImageNet (rescale 1/255 y
media [0.485, 0.456, 0.406] / desviacion [0.229, 0.224, 0.225]), que es con lo que
fueron entrenados estos modelos.
"""

from __future__ import annotations

import numpy as np
from transformers import AutoModelForSemanticSegmentation

from .colores import info_modelo

VENTANA = 224  # tamano de la ventana de inferencia (los dos modelos son fully-convolucional)

_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cargar(nombre: str):
    """Descarga (una sola vez) y carga el modelo preentrenado."""
    info = info_modelo(nombre)
    modelo_id = info["id"]
    print(f"  [i] Cargando modelo {modelo_id} ...")
    modelo = AutoModelForSemanticSegmentation.from_pretrained(modelo_id)
    modelo.eval()

    # El orden de config.id2label debe coincidir con la lista de clases declarada
    etiquetas = [modelo.config.id2label[i] for i in sorted(modelo.config.id2label)]
    if len(etiquetas) != len(info["clases"]):
        raise ValueError(
            f"El modelo {modelo_id} declara {len(etiquetas)} clases "
            f"({etiquetas}) pero la config local tiene {len(info['clases'])}."
        )
    print(f"  [i] Listo: {len(etiquetas)} clases -> {', '.join(info['clases'])}")
    return modelo


def preparar_lote(
    cortes: list[np.ndarray],
    media: np.ndarray | list[float] | None = None,
    desv: np.ndarray | list[float] | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Cortes RGB uint8 (224,224,3) -> tensor float32 (N,3,224,224) normalizado.

    `media`/`desv` permiten probar otras normalizaciones y `gamma` ajusta el brillo
    (gamma < 1 aclara, > 1 oscurece) antes de normalizar.
    """
    x = np.stack(cortes, axis=0).astype(np.float32) / 255.0  # (N,H,W,3)
    if gamma != 1.0:
        x = np.power(np.clip(x, 0.0, 1.0), gamma)
    x = (x - (_MEDIA if media is None else np.asarray(media, dtype=np.float32))) / (
        _DESV if desv is None else np.asarray(desv, dtype=np.float32)
    )
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
