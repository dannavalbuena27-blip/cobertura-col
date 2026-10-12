# cobertura-col

Programa en Python que **detecta el tipo de cobertura terrestre de una zona de Colombia**
usando modelos de inteligencia artificial geoespicial de código abierto e imágenes
Sentinel-2 gratuitas. Todo funciona sin cuentas ni API keys: los datos salen de un catálogo
STAC público y los modelos de Hugging Face.

```
$ uv run cobertura-col --coords 4.70,-74.08 --radio-km 3 --validar
```

## Instalación

Requisitos: [uv](https://docs.astral.sh/uv/) (gestor de paquetes de Python) e internet.
La primera ejecución descarga Python 3.12, las dependencias y los pesos del modelo.

```bash
uv python install 3.12   # instala el interprete (solo la primera vez)
uv sync                   # crea .venv e instala todo
uv run cobertura-col --help
```

También se puede instalar con pip: `pip install -e .`

## Uso rápido

Hay tres formas de indicar la zona (elige una):

```bash
# 1. Un bounding box (minlon,minlat,maxlon,maxlat en grados WGS84)
uv run cobertura-col --bbox -74.13,4.67,-74.07,4.72 --nombre bogota

# 2. Un punto con radio en km
uv run cobertura-col --coords 4.70,-74.08 --radio-km 3

# 3. Un poligono GeoJSON (se recorta exactamente a la figura)
uv run cobertura-col --geojson datos/zona_prueba.geojson --nombre parque

# Contraer con ESA WorldCover para medir la exactitud
uv run cobertura-col --coords 4.70,-74.08 --radio-km 3 --validar

# Usar el modelo grande (mejor en zonas urbanas) y aclarar la imagen
uv run cobertura-col --bbox -74.13,4.67,-74.07,4.72 --modelo b4 --gamma 0.7
```

Opciones principales (`--help` tiene la lista completa):

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--modelo {b0,b4}` | `b0` | Modelo de IA a usar (ver tabla de modelos) |
| `--backend {auto,openvino,pytorch}` | `auto` | Dónde correr la inferencia |
| `--device {GPU,CPU}` | `GPU` | Dispositivo OpenVINO: la GPU Intel integrada o la CPU |
| `--max-nubes` | `40` | Nubosidad máxima de las escenas seleccionadas |
| `--escenas` | `5` | Cuántas escenas entran al compuesto mediano |
| `--gamma` | `1.0` | Brillo antes de clasificar; `<1` aclara (~0.6 ayuda en bosque) |
| `--ventana` / `--solape` | `224` / `0.25` | Tamaño y solape de los recortes de inferencia |
| `--validar` | apagado | Calcula exactitud y IoU contra ESA WorldCover |

## Salidas

Cada ejecución crea una carpeta en `resultados/<zona>/`:

| Archivo | Contenido |
|---|---|
| `cobertura.tif` | Mapa de clases en GeoTIFF (listo para QGIS, con colormap y nodata) |
| `mapa.png` | Comparación lado a lado: color verdadero Sentinel-2 vs clasificación |
| `areas.csv` / `areas.json` | Hectáreas y porcentaje por clase |
| `metadata.json` | Escenas usadas, backend, tiempos y parámetros de la corrida |
| `validacion.csv/json`, `matriz_confusion.png` | Solo con `--validar` |

## Cómo funciona

```
zona (bbox / punto / GeoJSON)
  │  zona.py: valida los límites de Colombia y calcula el EPSG UTM
  ▼
catálogo STAC público (Earth Search, Element84 / AWS)
  │  descarga.py: busca escenas Sentinel-2 L2A de los últimos 12 meses
  ▼
compuesto mediano RGB
  │  lee B04/B03/B02 + máscara SCL en paralelo por rangos HTTP,
  │  recorta y reproyecta a una grilla UTM de 10 m, promedia por mediana
  │  (lida con nubes) y aplica estiramiento percentil 2-98 a 8 bits
  ▼
modelo SegFormer (recortes 224×224 con 25% de solape)
  │  backends.py: OpenVINO convertido a IR y compilado en la GPU Intel
  │  (fallback automático a CPU si algo falla)
  ▼
mapa de clases → GeoTIFF + PNG + áreas (+ validación opcional)
```

Detalle de la inferencia: el mapa se arma sin costuras porque cada píxel se toma del
recorte cuyo centro le queda más cerca (no se acumulan probabilidades, así que no hace
falta un arreglo pesado en memoria).

## Modelos disponibles

| Clave | Modelo | Clases | Cuándo usarlo |
|---|---|---|---|
| `b0` (default) | [florian-morel22/segformer-b0-deepglobe-land-cover](https://huggingface.co/florian-morel22/segformer-b0-deepglobe-land-cover) | zona urbana, agricultura, pastizal, bosque, agua, terreno desnudo, sin clasificar | General: rápido (3.7 M parámetros) y el mejor en bosque |
| `b4` | [nave1616/SegFormer-landcover-FT](https://huggingface.co/nave1616/SegFormer-landcover-FT) | fondo, terreno desnudo, pastizal, espacio desarrollado, carretera, árbol, agua, agricultura, edificaciones | Zonas urbanas: separa calles y edificaciones (64 M parámetros) |

Modelos probados y descartados (documentado para no repetir el trabajo):

- **EuroSAT Swin-T** (`brysanicks/...-finetuned-eurosat`): entrenado en Sentinel-2, pero
  es un *clasificador* (una clase por recorte, no por píxel) y en la Amazonía predijo
  "Pasture" en toda la zona: exactitud 37.9% vs 47.3% de `b0`.
- **torchgeo**: solo modelos binarios (carretera/árbol), no multiclase de cobertura.
- **Shoriful025/satellite_land_cover_segmenter_v1**: el repositorio solo tiene
  `config.json`, sin pesos.

Ninguno de los repos de los modelos trae `preprocessor_config.json`, así que la
normalización se aplica a mano (`modelos.py`): rescale 1/255 y media/desviación
ImageNet, que es con lo que fueron entrenados.

## Rendimiento (GPU Intel Iris Xe, sin GPU NVIDIA)

Medido con SegFormer-B0, recortes de 224×224, en estado estacionario:

| Backend | Milisegundos por recorte |
|---|---|
| PyTorch CPU | 156 ms |
| OpenVINO CPU | 74 ms |
| **OpenVINO GPU (Iris Xe)** | **30.7 ms** (5× más rápido que PyTorch CPU) |

La primera ejecución en GPU paga ~1 minuto de compilación; después se reutiliza desde
`modelos_cache/` (el IR convertido y el caché de kernels de OpenVINO).

Tiempos extremo a extremo para una zona de 6.7 × 5.6 km (Bogotá occidental):
~40 s en total, de los cuales ~37 s son la descarga del compuesto Sentinel-2 y
<1 s la inferencia en GPU.

## Validación y resultados reales

`--validar` compara el resultado con **ESA WorldCover 10 m v200 (2021)**, el mapa de
cobertura mundial de la ESA, descargado de un bucket S3 público por rangos HTTP.
Ambos mapas se agregan a un esquema común de 6 clases (agua, bosque, cultivos,
pastizal, urbano, desnudo) antes de comparar.

Resultados obtenidos en este equipo:

| Zona | Carácter | Modelo | Exactitud | IoU promedio |
|---|---|---|---|---|
| Bogotá occidental (6.7 × 5.6 km) | urbana | `b0` | 83.1% | 0.18 |
| Bogotá occidental | urbana | `b4` | **85.7%** | **0.28** |
| Parque Simón Bolívar y entorno (2.6 km) | urbana + parque | `b4` | **91.3%** | 0.24 |
| Caquetá/Putumayo (22 × 16 km) | bosque/pastizal | `b0` | 34.1% | 0.09 |
| Caquetá/Putumayo | bosque/pastizal | `b0 --gamma 0.6` | **44.8%** | 0.10 |
| Caquetá/Putumayo | bosque/pastizal | `b4` | 18.4% | 0.05 |

Conclusión práctica: **`b0` para zonas rurales y naturales, `b4` para ciudades**.
En la Amazonía el principal error es confundir pastizal tropical con bosque (el modelo
ve "verde" y dice bosque); `--gamma 0.6` lo reduce algo.

## Limitaciones

- **Desfase de dominio.** `b0` fue entrenado con imagen aérea de ~0.5 m (DeepGlobe) y
  aquí recibe Sentinel-2 a 10 m: las texturas no son las mismas y el modelo se inclina
  hacia la clase mayoritaria. Es la razón principal de los números de la tabla anterior.
- **Leyendas distintas.** WorldCover cuenta las copas de árbol urbanas como "bosque",
  mientras DeepGlobe las llamaría "urbano". Parte de la diferencia es de definición.
- **Fechas distintas.** WorldCover es de 2021 y las imágenes son recientes: los cambios
  reales del territorio (deforestación, construcción) cuentan como error.
- **Solo RGB.** Los modelos no usan infrarrojo, donde vegetación y suelo desnudo se
  separan mucho mejor (sería un buen siguiente paso: NDVI/NDWI como canales extra).
- **Nubes.** El compuesto mediano recién llega a 100% de cobertura en zonas poco
  nubladas; lo que no se observa queda como "sin dato" en el mapa y en las áreas.
- **Tamaño de zona.** El límite práctico está en ~50 km por lado (memoria de los
  arreglos de píxeles).

## Estructura del proyecto

```
opencurse/
├── pyproject.toml          # dependencias y punto de entrada (cobertura-col)
├── README.md
├── datos/zona_prueba.geojson
├── scripts/                # experimentos usados para elegir modelo y gamma
│   ├── experimentos.py         # compara variantes de b0/b4 contra WorldCover
│   ├── experimento_eurosat.py  # prueba del clasificador EuroSAT (descartado)
│   └── diagnostico.py          # histogramas y matriz de confusión por zona
├── src/cobertura_col/
│   ├── cli.py              # argumentos y orquestación
│   ├── zona.py             # bbox / punto / GeoJSON, validación de Colombia, UTM
│   ├── descarga.py         # STAC + compuesto mediano RGB libre de nubes
│   ├── modelos.py          # carga de pesos y normalización de entrada
│   ├── backends.py         # OpenVINO GPU/CPU y PyTorch CPU, con fallback
│   ├── inferencia.py       # ventanas deslizantes y costuras invisibles
│   ├── salida.py           # GeoTIFF, PNG con leyenda, áreas, metadatos
│   ├── validacion.py       # comparación contra ESA WorldCover
│   └── colores.py          # clases, paletas y esquema común de validación
├── resultados/             # salidas de ejemplo de las corridas de prueba
├── modelos_cache/          # IR de OpenVINO (se regenera solo)
└── cache_experimentos/     # compuestos cacheados para los experimentos
```

## Datos y modelos

- **Imágenes:** Copernicus Sentinel-2 L2A vía [Earth Search](https://earth-search.aws.element84.com/v1)
  (Element84 en AWS), catálogo STAC público.
- **Referencia de validación:** [ESA WorldCover 10 m v200](https://esa-worldcover.org/)
  (2021), S3 público `esa-worldcover`.
- **Modelos:** pesos publicados en Hugging Face (DeepGlobe y landcover-FT, ambos
  SegFormer con licencia abierta). El uso de cada repositorio está en su propia model card.
