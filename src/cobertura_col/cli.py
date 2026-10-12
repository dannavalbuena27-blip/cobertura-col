"""Interfaz de linea de comandos de cobertura-col.

Ejemplos:
  cobertura-col --coords 4.70,-74.08 --radio-km 3 --salida resultados/bogota
  cobertura-col --bbox -75.60,5.00,-75.50,5.10 --modelo b4 --backend openvino
  cobertura-col --geojson mi_zona.geojson --validar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import backends as backends_mod
from . import descarga, inferencia, modelos, salida, validacion
from .colores import MODELOS, info_modelo
from .zona import desde_bbox, desde_coords, desde_geojson


_OPCIONES_TEXTO = ("--bbox", "--coords", "--geojson", "--nombre", "--salida")


def _normalizar_argv(argv: list[str]) -> list[str]:
    """argparse confunde valores que empiezan con '-' (ej. --bbox -74.1,4.6,...) con flags.
    Se reescriben como --bbox=valor para que funcione en cualquier terminal."""
    salida: list[str] = []
    i = 0
    while i < len(argv):
        actual = argv[i]
        if actual in _OPCIONES_TEXTO and i + 1 < len(argv) and not argv[i + 1].startswith("="):
            salida.append(f"{actual}={argv[i + 1]}")
            i += 2
        else:
            salida.append(actual)
            i += 1
    return salida


def parsear_args(argv: list[str] | None = None) -> argparse.Namespace:
    argumentos = list(sys.argv[1:] if argv is None else argv)
    argumentos = _normalizar_argv(argumentos)
    parser = argparse.ArgumentParser(
        prog="cobertura-col",
        description=(
            "Detecta el tipo de cobertura terrestre de una zona de Colombia usando "
            "modelos de IA geoespacial de codigo abierto e imagenes Sentinel-2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    zona = parser.add_argument_group("zona de estudio (elige una)")
    zona.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat (grados, WGS84)")
    zona.add_argument("--coords", help="lat,lon del centro de la zona")
    zona.add_argument("--radio-km", type=float, default=3.0, help="radio en km para --coords")
    zona.add_argument("--geojson", help="archivo .geojson con el poligono de la zona")
    zona.add_argument("--nombre", default="", help="nombre de la zona (para nombrar salidas)")

    modelo = parser.add_argument_group("modelo y computo")
    modelo.add_argument("--modelo", choices=list(MODELOS), default="b0",
                        help="modelo de IA a usar")
    modelo.add_argument("--backend", choices=["auto", "openvino", "pytorch"], default="auto",
                        help="donde correr la inferencia")
    modelo.add_argument("--device", choices=["GPU", "CPU"], default="GPU",
                        help="dispositivo OpenVINO (GPU = tarjeta Intel integrada)")
    modelo.add_argument("--ventana", type=int, default=224, help="tamano del recorte de inferencia")
    modelo.add_argument("--solape", type=float, default=0.25, help="solape entre recortes (0-0.9)")
    modelo.add_argument("--lote", type=int, default=8, help="recortes por lote en CPU")
    modelo.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="brillo de la imagen antes de clasificar (<1 aclara; ~0.6 ayuda en zonas boscosas)",
    )

    datos = parser.add_argument_group("imagenes")
    datos.add_argument("--max-nubes", type=float, default=40.0, help="nubosidad maxima de las escenas, en porcentaje")
    datos.add_argument("--escenas", type=int, default=5, help="maximo de escenas en el compuesto")
    datos.add_argument("--meses", type=int, default=12, help="buscar escenas de los ultimos N meses")

    parser.add_argument("--salida", default="", help="carpeta de salida (por defecto: resultados/<zona>)")
    parser.add_argument("--validar", action="store_true", help="comparar contra ESA WorldCover 10 m")

    return parser.parse_args(argumentos)


def _construir_zona(args: argparse.Namespace):
    opciones = [bool(args.bbox), bool(args.coords), bool(args.geojson)]
    if sum(opciones) != 1:
        raise SystemExit("Indica exactamente una zona: --bbox, --coords o --geojson.")
    if args.bbox:
        return desde_bbox(args.bbox, args.nombre)
    if args.coords:
        partes = [p.strip() for p in args.coords.split(",")]
        if len(partes) != 2:
            raise SystemExit("--coords espera: lat,lon")
        return desde_coords(float(partes[0]), float(partes[1]), args.radio_km, args.nombre)
    return desde_geojson(args.geojson, args.nombre)


def main(argv: list[str] | None = None) -> int:
    args = parsear_args(argv)
    t_total = time.time()
    try:
        zona = _construir_zona(args)
        info = info_modelo(args.modelo)
        etiqueta = zona.etiqueta()
        directorio = Path(args.salida) if args.salida else Path("resultados") / etiqueta

        lon, lat = zona.centro  # centro devuelve (lon, lat)
        ancho_km = zona.ancho_grados * 111.32 * abs(__import__("math").cos(__import__("math").radians(lat)))
        alto_km = zona.alto_grados * 111.32
        print("=" * 70)
        print(" cobertura-col: deteccion de cobertura terrestre con IA geoespacial")
        print("=" * 70)
        print(f"  Zona       : {etiqueta}  (centro lat {lat:.4f}, lon {lon:.4f})")
        print(f"  Tamano     : {ancho_km:.2f} x {alto_km:.2f} km")
        print(f"  Proyeccion : EPSG:{zona.epsg_utm()} (UTM)")
        print(f"  Modelo     : {info['nombre']}")
        print(f"  Salida     : {directorio}")
        print("-" * 70)

        # 1. Imagen Sentinel-2 -------------------------------------------------
        t0 = time.time()
        compuesto = descarga.construir_compuesto(
            zona,
            max_nubes=args.max_nubes,
            max_escenas=args.escenas,
            meses=args.meses,
        )
        t_descarga = time.time() - t0
        print(f"  [i] Compuesto RGB listo: {compuesto.shape[1]}x{compuesto.shape[0]} px "
              f"({t_descarga:.1f} s)")

        # 2. Modelo + backend --------------------------------------------------
        modelo = modelos.cargar(args.modelo)
        backend, nombre_backend = backends_mod.crear_backend(
            args.backend, modelo, args.modelo, args.ventana, args.device
        )
        print(f"  [i] Backend: {nombre_backend}")

        # 3. Inferencia --------------------------------------------------------
        t0 = time.time()
        mapa = inferencia.inferir_mapa(
            compuesto.rgb,
            compuesto.mascara,
            backend,
            ventana=args.ventana,
            solape=args.solape,
            lote=args.lote,
            preparar=lambda corte: modelos.preparar_lote(corte, gamma=args.gamma),
        )
        t_inferencia = time.time() - t0
        print(f"  [i] Mapa de clases listo ({t_inferencia:.1f} s)")

        if zona.geometria is not None:
            mapa = salida.recortar_a_geometria(mapa, zona.geometria, compuesto.transform, compuesto.crs)

        # 4. Salidas -----------------------------------------------------------
        rutas = salida.escribir_todo(
            directorio,
            etiqueta,
            compuesto,
            mapa,
            info,
            {
                "backend": nombre_backend,
                "tiempos_s": {
                    "descarga": round(t_descarga, 1),
                    "inferencia": round(t_inferencia, 1),
                },
                "parametros": {
                    "ventana": args.ventana,
                    "solape": args.solape,
                    "gamma": args.gamma,
                    "max_nubes_pct": args.max_nubes,
                    "escenas": args.escenas,
                },
            },
        )

        # 5. Validacion opcional ------------------------------------------------
        if args.validar:
            t0 = time.time()
            try:
                validacion.validar(mapa, compuesto, info, directorio, zona.bbox)
                print(f"  [i] Validacion lista ({time.time() - t0:.1f} s)")
            except Exception as exc:
                print(f"  [!] No fue posible validar contra WorldCover: {exc}")

        # 6. Resumen -------------------------------------------------------------
        print("-" * 70)
        filas = salida.calcular_areas(mapa, compuesto.transform, info["clases"])
        print(f"  {'clase':<28} {'ha':>12} {'%':>7}")
        for f in filas:
            print(f"  {f['clase']:<28} {f['hectareas']:>12.2f} {f['porcentaje']:>6.1f}%")
        print("-" * 70)
        print(f"  Archivos en {directorio}:")
        for clave, ruta in rutas.items():
            print(f"    - {clave:<12} {ruta}")
        print(f"  Tiempo total: {time.time() - t_total:.1f} s")
        print("=" * 70)
        return 0
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
