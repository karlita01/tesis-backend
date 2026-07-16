"""
Módulo de detección YOLOv8 para EP-003.

Responsabilidades:
- Detectar personas en frames individuales (webcam o video).
- Descartar detecciones dentro de zonas de exclusión (RF-3.2).
- Clasificar el nivel de concentración con umbrales configurables (RF-3.3).
- Mantener estado por sesión: ventana deslizante para alerta sostenida (RF-3.4).
- Registrar tiempo hasta primera detección media+ (RF-3.5).
"""

import json
import time
from collections import deque
from threading import Lock

import torch
import cv2
import numpy as np

# ── Modelo YOLO (carga perezosa, una sola instancia) ──────────────────────────

_model = None
_model_lock = Lock()
# Serializa TODAS las llamadas de inferencia (no solo la carga): el modelo es una
# única instancia compartida entre hilos (webcam, video previa, cámara IP), y
# ejecutar forward() concurrentemente sobre la misma instancia/GPU desde hilos
# distintos no es seguro.
_inference_lock = Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NIVEL_ORDEN = ["sin_aglomeracion", "bajo", "medio", "alto"]


def _parchear_export_formats() -> None:
    """
    Evita un bug de compatibilidad binaria pandas/numpy observado en despliegues
    en la nube (pandas 2.2.2 + numpy 1.26 revientan al construir un DataFrame,
    con "TypeError: Cannot convert numpy.ndarray to numpy.ndarray"). Ese DataFrame
    lo arma internamente AutoBackend._model_type() solo para leer la columna
    "Suffix" y así detectar el tipo de archivo del modelo (.pt, .onnx, etc.).
    En este proyecto siempre se carga un .pt plano, así que reemplazamos esa
    función por una versión sin pandas, con exactamente los mismos datos.
    """
    import ultralytics.engine.exporter as exporter_module
    import ultralytics.nn.autobackend as autobackend_module

    class _TablaFormatos:
        def __init__(self, filas: list[list]) -> None:
            self.Suffix = [f[2] for f in filas]
            self.Argument = [f[1] for f in filas]
            self.CPU = [f[3] for f in filas]
            self.GPU = [f[4] for f in filas]

    _FILAS = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]

    _reemplazo = lambda: _TablaFormatos(_FILAS)  # noqa: E731
    exporter_module.export_formats = _reemplazo
    autobackend_module.export_formats = _reemplazo


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _parchear_export_formats()
                from ultralytics import YOLO
                # .to(DEVICE) resetea internamente model.predictor a None (ver
                # Model._apply en ultralytics). Debe terminar ANTES de publicar
                # el modelo en _model: si otro hilo (otra cámara) ve _model ya
                # asignado mientras .to() todavía corre, puede empezar a predecir
                # justo cuando .to() le resetea el predictor a mitad de camino,
                # provocando "TypeError: 'NoneType' object is not callable".
                modelo = YOLO("yolov8s.pt")
                modelo.to(DEVICE)
                _model = modelo
    return _model


def warmup() -> None:
    """
    Carga el modelo y corre una inferencia de prueba. Se llama una vez al
    arrancar la app (lifespan) para que el costo de carga + primera
    inferencia (compilación/autotune de kernels CUDA) se pague al iniciar
    el servidor, no en la primera cámara que abra un usuario.
    """
    model = _get_model()
    frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with _inference_lock:
        # Envuelto en lista: un ndarray "pelado" puede terminar iterado por su
        # primer eje (cada fila de píxeles) en vez de tratado como una sola
        # imagen, rompiendo el preprocesamiento interno de ultralytics.
        # Sin device=: el modelo ya fue movido a DEVICE con .to() en _get_model();
        # re-especificarlo aquí puede disparar una reinicialización interna del
        # backend en modo CPU que rompe el preprocesamiento (cv2.resize).
        model([frame_dummy], classes=[0], conf=0.40, imgsz=416, verbose=False)


# ── Lógica de detección ───────────────────────────────────────────────────────

def _en_zona_exclusion(
    x1: float, y1: float, x2: float, y2: float, zonas: list[dict]
) -> bool:
    """
    Excluye una detección si su centro cae dentro de una zona O si al menos
    el 25 % del área del bounding box intersecta con la zona.
    Usar el bbox completo (no solo el punto inferior) evita falsos positivos
    con objetos altos como maniquíes cuya base puede salirse de la zona.
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bbox_area = max((x2 - x1) * (y2 - y1), 1e-9)

    for z in zonas:
        zx1, zy1 = z["x"], z["y"]
        zx2, zy2 = z["x"] + z["width"], z["y"] + z["height"]

        # Criterio 1: centro del bbox dentro de la zona
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True

        # Criterio 2: área de intersección ≥ 25 % del bbox
        ix1, iy1 = max(x1, zx1), max(y1, zy1)
        ix2, iy2 = min(x2, zx2), min(y2, zy2)
        if ix2 > ix1 and iy2 > iy1:
            if (ix2 - ix1) * (iy2 - iy1) / bbox_area >= 0.25:
                return True

    return False


def _agrupar_bfs(puntos: list[tuple[float, float]], umbral_dist: float) -> int:
    """
    BFS sobre los puntos (coordenadas normalizadas).
    Retorna el tamaño del grupo más grande de personas cercanas entre sí.
    Dos personas pertenecen al mismo grupo si la distancia entre sus
    bottom-centers es ≤ umbral_dist.
    """
    n = len(puntos)
    if n == 0:
        return 0
    visitado = [False] * n
    max_grupo = 0
    for i in range(n):
        if visitado[i]:
            continue
        cola = [i]
        visitado[i] = True
        tam = 0
        while cola:
            curr = cola.pop()
            tam += 1
            px, py = puntos[curr]
            for j in range(n):
                if not visitado[j]:
                    dx = px - puntos[j][0]
                    dy = py - puntos[j][1]
                    if (dx * dx + dy * dy) ** 0.5 <= umbral_dist:
                        visitado[j] = True
                        cola.append(j)
        max_grupo = max(max_grupo, tam)
    return max_grupo


def clasificar_nivel(personas: int, umbral_medio: int, umbral_alto: int) -> str:
    if personas == 0:
        return "sin_aglomeracion"
    if personas < umbral_medio:
        return "bajo"
    if personas < umbral_alto:
        return "medio"
    return "alto"


# Distancia máxima entre bottom-centers (coordenadas normalizadas) para
# considerar a dos personas parte del mismo grupo. ~0.20 equivale a una
# persona de ancho en un encuadre típico de pasillo.
DISTANCIA_GRUPO_DEFAULT = 0.20


def procesar_frame(
    frame_bytes: bytes,
    zonas_exclusion: list[dict],
    umbral_medio: int,
    umbral_alto: int,
    conf_min: float = 0.40,
    distancia_grupo: float = DISTANCIA_GRUPO_DEFAULT,
) -> dict:
    """
    Procesa un frame JPEG y devuelve:
      personas       — tamaño del grupo más grande detectado (BFS, RF-3.3)
      total_personas — total de personas fuera de zonas de exclusión
      nivel          — clasificación según umbrales
      detecciones    — lista de bboxes normalizados con flag excluida
    """
    arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"personas": 0, "total_personas": 0, "nivel": "sin_aglomeracion", "detecciones": []}

    h, w = frame.shape[:2]
    model = _get_model()
    with _inference_lock:
        # imgsz reducido (default YOLO es 640) para sostener más fps (RNF-01)
        # Sin device=: el modelo ya está en DEVICE por .to() en _get_model().
        results = model(frame, classes=[0], conf=conf_min, imgsz=416, verbose=False)[0]

    bottom_centers: list[tuple[float, float]] = []
    detecciones = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        # Coordenadas normalizadas del bbox completo
        nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h

        excluida = _en_zona_exclusion(nx1, ny1, nx2, ny2, zonas_exclusion)
        if not excluida:
            cx = (nx1 + nx2) / 2
            cy = ny2           # bottom-center para BFS (posición en suelo)
            bottom_centers.append((cx, cy))

        detecciones.append({
            "x1": round(nx1, 4),
            "y1": round(ny1, 4),
            "x2": round(nx2, 4),
            "y2": round(ny2, 4),
            "conf": round(conf, 3),
            "excluida": excluida,
        })

    total_personas = len(bottom_centers)
    # BFS: detectar el grupo conectado más grande (personas realmente aglomeradas)
    grupo_max = _agrupar_bfs(bottom_centers, distancia_grupo)

    nivel = clasificar_nivel(grupo_max, umbral_medio, umbral_alto)
    return {
        "personas": grupo_max,
        "total_personas": total_personas,
        "nivel": nivel,
        "detecciones": detecciones,
    }


# ── Estado de sesión (ventana deslizante, RF-3.4 / RF-3.5) ──────────────────

class SesionAnalisisState:
    """Estado acumulado de una sesión de análisis en tiempo real."""

    def __init__(
        self,
        zona_config: dict | None,
        umbral_medio: int = 4,
        umbral_alto: int = 6,
        ventana_segundos: float = 2.0,
        cooldown_segundos: int = 10,
    ):
        # Parámetros (toma valores de zona_config si existe)
        if zona_config:
            self.umbral_medio = zona_config.get("umbral_medio", umbral_medio)
            self.umbral_alto = zona_config.get("umbral_alto", umbral_alto)
            self.ventana_segundos = zona_config.get("ventana_segundos", ventana_segundos)
            self.cooldown_segundos = zona_config.get("cooldown_segundos", cooldown_segundos)
            self.zonas_exclusion: list[dict] = zona_config.get("zonas", [])
        else:
            self.umbral_medio = umbral_medio
            self.umbral_alto = umbral_alto
            self.ventana_segundos = ventana_segundos
            self.cooldown_segundos = cooldown_segundos
            self.zonas_exclusion = []

        self.inicio = time.time()
        self.tiempo_primera_media: float | None = None
        self.personas_maximas = 0
        self.nivel_maximo = "sin_aglomeracion"
        self.alerta_activada = False
        self.ultima_alerta: float | None = None
        self.frames_ventana: deque = deque()   # (timestamp, nivel)
        self.frames_procesados = 0
        self.frame_evidencia_bytes: bytes | None = None  # RF-5.2

    def actualizar(self, personas: int, nivel: str) -> bool:
        """
        Registra el resultado de un frame.
        Devuelve True si se activa una nueva alerta en este frame.
        """
        ahora = time.time()
        self.frames_procesados += 1

        # RF-3.5: tiempo hasta primera detección media+
        if nivel in ("medio", "alto") and self.tiempo_primera_media is None:
            self.tiempo_primera_media = round(ahora - self.inicio, 2)

        # Actualizar máximos históricos
        if personas > self.personas_maximas:
            self.personas_maximas = personas
        if NIVEL_ORDEN.index(nivel) > NIVEL_ORDEN.index(self.nivel_maximo):
            self.nivel_maximo = nivel

        # RF-3.4: ventana deslizante de tiempo
        self.frames_ventana.append((ahora, nivel))
        while self.frames_ventana and ahora - self.frames_ventana[0][0] > self.ventana_segundos:
            self.frames_ventana.popleft()

        # Alerta si ≥70 % de frames en la ventana son "alto" y hay al menos 3
        total = len(self.frames_ventana)
        altos = sum(1 for _, n in self.frames_ventana if n == "alto")

        nueva_alerta = False
        if total >= 3 and altos / total >= 0.70:
            cooldown_ok = (
                self.ultima_alerta is None
                or ahora - self.ultima_alerta > self.cooldown_segundos
            )
            if cooldown_ok:
                self.ultima_alerta = ahora
                self.alerta_activada = True
                nueva_alerta = True

        return nueva_alerta

    def resumen(self) -> dict:
        return {
            "personas_maximas": self.personas_maximas,
            "nivel_maximo": self.nivel_maximo,
            "tiempo_primera_media_seg": self.tiempo_primera_media,
            "alerta_activada": self.alerta_activada,
            "frames_procesados": self.frames_procesados,
        }


# ── Registro global de estados por sesión ────────────────────────────────────

_estados: dict[int, SesionAnalisisState] = {}
_estados_lock = Lock()


def obtener_estado(sesion_id: int) -> SesionAnalisisState | None:
    return _estados.get(sesion_id)


def crear_estado(sesion_id: int, zona_config: dict | None = None) -> SesionAnalisisState:
    estado = SesionAnalisisState(zona_config=zona_config)
    with _estados_lock:
        _estados[sesion_id] = estado
    return estado


def eliminar_estado(sesion_id: int) -> SesionAnalisisState | None:
    with _estados_lock:
        return _estados.pop(sesion_id, None)


# ── Procesamiento de video completo (para grabacion_previa) ──────────────────

def procesar_video_sync(
    ruta_video: str,
    zona_config: dict | None,
    estado: SesionAnalisisState,
    callback,           # callable(event_dict) → None
    cancelado_fn=None,  # callable() → bool, para cancelación
):
    """
    Procesa un archivo de video frame a frame.
    Llama a callback() con cada evento (dict) para que el caller lo transmita vía SSE.
    """
    cap = cv2.VideoCapture(ruta_video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        callback({"tipo": "error", "mensaje": "No se pudo abrir el archivo de video."})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = 0

    zonas_exc = estado.zonas_exclusion
    umbral_m = estado.umbral_medio
    umbral_a = estado.umbral_alto

    # Procesar ~25 fps máximo (RNF-01) para mantener latencia razonable.
    # En GPU el modelo sostiene esto sin problema; en CPU, bajar este valor.
    saltar = max(1, int(fps / 25))

    try:
        while True:
            if cancelado_fn and cancelado_fn():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % saltar != 0:
                continue

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buf.tobytes()
            resultado = procesar_frame(frame_bytes, zonas_exc, umbral_m, umbral_a)

            ts_video = round(frame_num / fps, 2)
            es_nuevo_maximo = resultado["personas"] > estado.personas_maximas
            alerta = estado.actualizar(resultado["personas"], resultado["nivel"])

            # RF-5.2: guardar frame con mayor concentración detectada
            if es_nuevo_maximo and resultado["personas"] > 0:
                _, ev_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                estado.frame_evidencia_bytes = ev_buf.tobytes()

            callback({
                "tipo": "frame",
                "frame_num": frame_num,
                "total_frames": total_frames,
                "progreso": round(frame_num / max(total_frames, 1) * 100, 1),
                "timestamp_video": ts_video,
                "personas": resultado["personas"],
                "nivel": resultado["nivel"],
                "alerta": alerta,
                "detecciones": resultado["detecciones"],
            })
    finally:
        cap.release()

    callback({"tipo": "fin", **estado.resumen()})


# ── Dibujo OpenCV para streams MJPEG ─────────────────────────────────────────

def _rect_dashed(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 color: tuple, dash: int = 8, gap: int = 4, thickness: int = 2):
    """Dibuja un rectángulo con borde discontinuo."""
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (p1, p2) in edges:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(abs(dx), abs(dy), 1)
        step = dash + gap
        for i in range(0, length, step):
            t0 = i / length
            t1 = min(i + dash, length) / length
            s = (int(p1[0] + t0 * dx), int(p1[1] + t0 * dy))
            e = (int(p1[0] + t1 * dx), int(p1[1] + t1 * dy))
            cv2.line(img, s, e, color, thickness)


def _dibujar_frame_cv2(
    frame: np.ndarray,
    detecciones: list,
    zonas_exclusion: list,
) -> np.ndarray:
    """Dibuja zonas de exclusión y bboxes activos sobre el frame. Retorna el frame modificado."""
    h, w = frame.shape[:2]

    # Zonas de exclusión — violeta punteado (BGR: 247, 85, 168)
    violet = (247, 85, 168)
    for zona in zonas_exclusion:
        zx1 = int(zona["x"] * w)
        zy1 = int(zona["y"] * h)
        zx2 = int((zona["x"] + zona["width"]) * w)
        zy2 = int((zona["y"] + zona["height"]) * h)
        _rect_dashed(frame, zx1, zy1, zx2, zy2, violet)

    # Detecciones activas — cyan (BGR: 238, 211, 34)
    cyan = (238, 211, 34)
    black = (0, 0, 0)
    for d in detecciones:
        if d.get("excluida"):
            continue
        bx1 = int(d["x1"] * w)
        by1 = int(d["y1"] * h)
        bx2 = int(d["x2"] * w)
        by2 = int(d["y2"] * h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), cyan, 2)
        label = f"{d['conf']:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(by1 - 4, lh + 2)
        cv2.rectangle(frame, (bx1, label_y - lh - 2), (bx1 + lw + 4, label_y + 2), cyan, -1)
        cv2.putText(frame, label, (bx1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, black, 1)

    return frame


# ── Procesamiento continuo de stream RTSP ────────────────────────────────────

def procesar_rtsp_mjpeg(
    rtsp_url: str,
    zona_config: dict | None,
    estado: SesionAnalisisState,
    cancelado_fn,   # callable() → bool
    on_frame,       # callable(jpeg_bytes: bytes, resultado: dict, alerta: bool) → None
):
    """
    Abre un stream RTSP continuo, procesa cada frame con YOLO+BFS, dibuja
    detecciones y llama on_frame con el JPEG anotado + stats.
    Se detiene cuando cancelado_fn() devuelve True o tras 3 fallos seguidos.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        on_frame(None, {"tipo": "error", "mensaje": "No se pudo conectar al stream RTSP."}, False)
        return

    # Control de ritmo por tiempo real (no por conteo de frames): cap.get(FPS)
    # puede ser poco confiable o el stream puede entregar frames a ráfagas, y
    # decidir "saltar" solo por conteo puede dejar el bucle sin freno real,
    # saturando la CPU (JPEG encode/decode) y matando de hambre al event loop
    # de uvicorn (GIL) aunque nada esté técnicamente en deadlock.
    INTERVALO_MIN = 1.0 / 25  # ~25 fps máximo, en GPU. En CPU, subir este valor.
    ultimo_procesado = 0.0
    fallos = 0

    zonas_exc = estado.zonas_exclusion
    umbral_m = estado.umbral_medio
    umbral_a = estado.umbral_alto

    try:
        while not (cancelado_fn and cancelado_fn()):
            ret, frame = cap.read()
            if not ret:
                fallos += 1
                if fallos >= 3:
                    break
                time.sleep(0.1)
                continue
            fallos = 0

            ahora = time.time()
            if ahora - ultimo_procesado < INTERVALO_MIN:
                continue
            ultimo_procesado = ahora

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            resultado = procesar_frame(buf.tobytes(), zonas_exc, umbral_m, umbral_a)

            es_nuevo_maximo = resultado["personas"] > estado.personas_maximas
            alerta = estado.actualizar(resultado["personas"], resultado["nivel"])

            if es_nuevo_maximo and resultado["personas"] > 0:
                _, ev_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                estado.frame_evidencia_bytes = ev_buf.tobytes()

            # Dibujar anotaciones y emitir JPEG
            frame_anot = frame.copy()
            _dibujar_frame_cv2(frame_anot, resultado["detecciones"], zonas_exc)
            _, out_buf = cv2.imencode(".jpg", frame_anot, [cv2.IMWRITE_JPEG_QUALITY, 75])
            on_frame(out_buf.tobytes(), resultado, alerta)
    finally:
        cap.release()
