"""Backends de inferencia: OpenVINO (GPU Intel / CPU) y PyTorch (CPU).

El modelo de Hugging Face se convierte una sola vez a OpenVINO IR y se cachea en
`modelos_cache/`, de modo que las siguientes ejecuciones arrancan rapido.

  * auto     -> usa la GPU Intel (Iris Xe, etc.) si OpenVINO la detecta; si no, CPU con PyTorch.
  * openvino -> fuerza OpenVINO con el dispositivo indicado en `device` (GPU/CPU).
  * pytorch  -> fuerza PyTorch en CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_MODELOS = Path("modelos_cache")


class BackendError(RuntimeError):
    pass


def _logits_a_etiquetas(logits: np.ndarray, salida: tuple[int, int]) -> np.ndarray:
    """(N,C,h,w) -> (N,H,W). Los SegFormer de transformers v5 devuelven logits a 1/4
    de la resolucion, asi que se interpolan bilinealmente antes del argmax."""
    if logits.shape[-2:] != tuple(salida):
        import torch

        logits = (
            torch.nn.functional.interpolate(
                torch.from_numpy(logits), size=salida, mode="bilinear", align_corners=False
            )
            .numpy()
        )
    return logits.argmax(axis=1)


class PyTorchBackend:
    """Inferencia con PyTorch en CPU."""

    nombre = "pytorch-cpu"

    def __init__(self, modelo, lote: int = 4):
        import torch

        self._torch = torch
        self._modelo = modelo
        self._lote = lote
        self._salida: tuple[int, int] | None = None

    def inferir(self, pixel_values: np.ndarray) -> np.ndarray:
        """(N,3,H,W) float32 -> (N,H,W) int64 con la clase de cada pixel."""
        torch = self._torch
        if self._salida is None:
            self._salida = (pixel_values.shape[2], pixel_values.shape[3])
        predicciones = []
        with torch.no_grad():
            for i in range(0, pixel_values.shape[0], self._lote):
                x = torch.from_numpy(pixel_values[i : i + self._lote])
                logits = self._modelo(pixel_values=x).logits.cpu().numpy()
                predicciones.append(_logits_a_etiquetas(logits, self._salida))
        return np.concatenate(predicciones, axis=0)


class OpenVINOBackend:
    """Inferencia con OpenVINO en la GPU Intel integrada (o en CPU)."""

    def __init__(self, modelo, device: str = "GPU", ventana: int = 224, convertible=None):
        import openvino as ov
        import torch

        self._ov = ov
        self._device = device
        CACHE_MODELOS.mkdir(exist_ok=True)
        nombre = (convertible or "modelo")
        ruta_xml = CACHE_MODELOS / f"{nombre}_v{ventana}.xml"

        core = ov.Core()
        if ruta_xml.exists():
            print(f"  [i] Cargando modelo OpenVINO desde {ruta_xml}")
            modelo_ov = core.read_model(str(ruta_xml))
        else:
            print(f"  [i] Convirtiendo modelo a OpenVINO IR (primera vez) ...")
            ejemplo = torch.zeros(1, 3, ventana, ventana)
            try:
                modelo_ov = ov.convert_model(modelo, example_input=ejemplo)
            except Exception as exc:
                raise BackendError(f"No se pudo convertir el modelo a OpenVINO: {exc}") from exc
            ov.save_model(modelo_ov, str(ruta_xml), compress_to_fp16=True)

        dispositivos = core.available_devices
        if device not in dispositivos:
            raise BackendError(
                f"El dispositivo '{device}' no esta disponible. Dispositivos: {dispositivos}"
            )
        # CACHE_DIR acelera los arranques siguientes (la primera compilacion en GPU tarda)
        self._compilado = core.compile_model(
            modelo_ov, device, config={"CACHE_DIR": str(CACHE_MODELOS / "ov_cache")}
        )
        self._salida = self._compilado.output(0)
        self._solicitud = self._compilado.create_infer_request()
        self.nombre = f"openvino-{device.lower()}"
        print(f"  [i] Modelo compilado en {device} (dispositivos detectados: {dispositivos})")

    def inferir(self, pixel_values: np.ndarray) -> np.ndarray:
        salida = (pixel_values.shape[2], pixel_values.shape[3])
        preds = []
        for i in range(pixel_values.shape[0]):
            logits = self._solicitud.infer({0: pixel_values[i : i + 1]})[self._salida]
            preds.append(_logits_a_etiquetas(np.asarray(logits), salida))
        return np.concatenate(preds, axis=0)


def _openvino_disponible_gpu() -> bool:
    try:
        import openvino as ov

        return "GPU" in ov.Core().available_devices
    except Exception:
        return False


def crear_backend(
    eleccion: str,
    modelo,
    modelo_nombre: str,
    ventana: int,
    device: str = "GPU",
):
    """Devuelve (backend, nombre_usado). En modo auto cae automaticamente a CPU si algo falla."""
    eleccion = eleccion.lower()

    if eleccion == "pytorch":
        return PyTorchBackend(modelo), "pytorch-cpu"

    if eleccion == "openvino":
        return OpenVINOBackend(modelo, device=device, ventana=ventana, convertible=modelo_nombre), (
            f"openvino-{device.lower()}"
        )

    # auto
    if _openvino_disponible_gpu():
        try:
            b = OpenVINOBackend(modelo, device="GPU", ventana=ventana, convertible=modelo_nombre)
            return b, b.nombre
        except Exception as exc:
            print(f"  [!] No fue posible usar la GPU con OpenVINO ({exc}); usando PyTorch en CPU.")
    return PyTorchBackend(modelo), "pytorch-cpu"
