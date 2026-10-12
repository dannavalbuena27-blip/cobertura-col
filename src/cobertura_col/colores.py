"""Paletas de colores, nombres en espanol de las clases y esquema comun de validacion."""

from __future__ import annotations

# --------------------------------------------------------------------------
# Esquema comun de 6 clases, usado para comparar contra ESA WorldCover.
# El orden define los indices internos de validacion (0 = sin dato).
# --------------------------------------------------------------------------
ESQUEMA_COMUN = ["sin_dato", "agua", "bosque", "cultivos", "pastizal", "urbano", "desnudo"]

COLORES_COMUN = {
    "sin_dato": "#000000",
    "agua": "#2C7FB8",
    "bosque": "#1E7B34",
    "cultivos": "#F2C85B",
    "pastizal": "#B0D67A",
    "urbano": "#D7191C",
    "desnudo": "#A6761D",
}

# Codigos de ESA WorldCover v200 (2021) -> esquema comun
# 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up,
# 60 Bare/sparse, 70 Snow/ice, 80 Permanent water, 90 Herbaceous wetland,
# 95 Mangroves, 100 Moss/lichen
WORLDCOVER_A_COMUN = {
    10: "bosque",
    20: "pastizal",
    30: "pastizal",
    40: "cultivos",
    50: "urbano",
    60: "desnudo",
    70: "desnudo",
    80: "agua",
    90: "pastizal",  # humedal herbaceo -> se aproxima a pastizal
    95: "bosque",  # manglar -> bosque
    100: "desnudo",  # musgo/lichen en altura
}

# --------------------------------------------------------------------------
# Modelos. Cada entrada mapea el indice de salida del modelo al esquema comun.
# None = la clase se excluye de la validacion (se reporta como cobertura).
# --------------------------------------------------------------------------
MODELOS = {
    "b0": {
        "id": "florian-morel22/segformer-b0-deepglobe-land-cover",
        "nombre": "SegFormer-B0 (DeepGlobe Land Cover)",
        "descripcion": "Segmentacion RGB en 6 clases de cobertura, ligero y rapido en CPU.",
        # orden = indice de salida del modelo (ver config.json del repositorio)
        "clases": [
            "zona urbana",
            "agricultura",
            "pastizal",
            "bosque",
            "agua",
            "terreno desnudo",
            "sin clasificar",
        ],
        "colores": [
            "#D7191C",  # zona urbana
            "#F2C85B",  # agricultura
            "#B0D67A",  # pastizal
            "#1E7B34",  # bosque
            "#2C7FB8",  # agua
            "#A6761D",  # terreno desnudo
            "#9E9E9E",  # sin clasificar
        ],
        "a_comun": ["urbano", "cultivos", "pastizal", "bosque", "agua", "desnudo", None],
        "ventana_defecto": 224,
    },
    "b4": {
        "id": "nave1616/SegFormer-landcover-FT",
        "nombre": "SegFormer-B4 (landcover 9 clases)",
        "descripcion": "Segmentacion RGB con 9 clases; incluye carreteras y edificaciones. Mas lento.",
        "clases": [
            "fondo",
            "terreno desnudo",
            "pastizal",
            "espacio desarrollado",
            "carretera",
            "arbol",
            "agua",
            "agricultura",
            "edificaciones",
        ],
        "colores": [
            "#9E9E9E",  # fondo
            "#A6761D",  # terreno desnudo
            "#B0D67A",  # pastizal
            "#F4A582",  # espacio desarrollado
            "#4D4D4D",  # carretera
            "#1E7B34",  # arbol
            "#2C7FB8",  # agua
            "#F2C85B",  # agricultura
            "#A50F15",  # edificaciones
        ],
        "a_comun": [
            None,
            "desnudo",
            "pastizal",
            "urbano",
            "urbano",
            "bosque",
            "agua",
            "cultivos",
            "urbano",
        ],
        "ventana_defecto": 224,
    },
}


def info_modelo(nombre: str) -> dict:
    if nombre not in MODELOS:
        raise KeyError(f"Modelo desconocido '{nombre}'. Opciones: {', '.join(MODELOS)}")
    return MODELOS[nombre]
