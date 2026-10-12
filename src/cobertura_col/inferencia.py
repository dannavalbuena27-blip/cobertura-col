"""Inferencia por ventanas deslizantes sobre el RGB y construccion del mapa de clases.

El modelo trabaja con recortes de 224x224. Para cubrir la zona completa se usan
ventanas con solape y luego cada pixel se asigna a la ventana cuyo centro le queda
mas cerca (evita costuras entre recortes sin necesidad de acumular probabilidades).
"""

from __future__ import annotations

import numpy as np

from .modelos import preparar_lote


def inicios(total: int, ventana: int, paso: int) -> list[int]:
    """Posiciones de inicio (fila o columna) que cubren toda la dimension."""
    if total <= ventana:
        return [0]
    posiciones = list(range(0, total - ventana + 1, paso))
    if posiciones[-1] != total - ventana:
        posiciones.append(total - ventana)
    return posiciones


def responsabilidades(posiciones: list[int], ventana: int, total: int) -> list[tuple[int, int]]:
    """Para cada ventana, que rango contiguo de pixeles le corresponde (al que le esta mas cerca)."""
    centros = [p + ventana / 2.0 for p in posiciones]
    limites = [(centros[i] + centros[i + 1]) / 2.0 for i in range(len(centros) - 1)]
    propietario = np.searchsorted(limites, np.arange(total))

    rangos: list[tuple[int, int]] = []
    for i in range(len(posiciones)):
        idx = np.where(propietario == i)[0]
        if idx.size == 0:
            rangos.append((0, 0))
        else:
            rangos.append((int(idx[0]), int(idx[-1]) + 1))
    return rangos


def inferir_mapa(
    rgb: np.ndarray,
    mascara: np.ndarray,
    backend,
    ventana: int = 224,
    solape: float = 0.25,
    lote: int = 8,
    progreso=print,
    preparar=None,
) -> np.ndarray:
    """RGB uint8 (H,W,3) + mascara bool -> mapa de clases uint8 (H,W), 255 = sin dato.

    `preparar` es la funcion que convierte una lista de cortes RGB en el tensor de
    entrada (por defecto la normalizacion estandar de modelos.preparar_lote).
    """
    preparar = preparar or preparar_lote
    alto, ancho = rgb.shape[0], rgb.shape[1]
    necesita_pad = alto < ventana or ancho < ventana
    if necesita_pad:
        rgb = np.pad(
            rgb,
            ((0, max(0, ventana - alto)), (0, max(0, ventana - ancho)), (0, 0)),
            mode="edge",
        )
    filas_tot, cols_tot = rgb.shape[0], rgb.shape[1]

    paso = max(1, int(round(ventana * (1.0 - solape))))
    filas = inicios(filas_tot, ventana, paso)
    cols = inicios(cols_tot, ventana, paso)
    rango_filas = responsabilidades(filas, ventana, filas_tot)
    rango_cols = responsabilidades(cols, ventana, cols_tot)

    mapa = np.zeros((filas_tot, cols_tot), dtype=np.uint8)
    recortes_pos = [(f, c) for f in filas for c in cols]
    total = len(recortes_pos)
    progreso(f"  [i] Inferencia: {total} recortes de {ventana}x{ventana} (solape {solape:.0%})")

    # Agrupamos recortes en lotes para no llamar al modelo uno por uno
    contador = 0
    for inicio in range(0, total, lote):
        grupo = recortes_pos[inicio : inicio + lote]
        cortes = [rgb[f : f + ventana, c : c + ventana] for f, c in grupo]
        valores = preparar(cortes)
        preds = backend.inferir(valores)  # (N, ventana, ventana)

        for (f, c), pred in zip(grupo, preds):
            f0, f1 = rango_filas[filas.index(f)]
            c0, c1 = rango_cols[cols.index(c)]
            mapa[f0:f1, c0:c1] = pred[f0 - f : f1 - f, c0 - c : c1 - c]

        contador += len(grupo)
        if total >= 4 and (contador % (lote * 4) < lote or contador == total):
            progreso(f"  ... {contador}/{total} recortes")

    if necesita_pad:
        mapa = mapa[:alto, :ancho]

    mapa[~mascara] = 255  # nubes y zonas sin dato quedan como "sin dato"
    return mapa
