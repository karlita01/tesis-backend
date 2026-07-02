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

import cv2
import numpy as np

# ── Modelo YOLO (carga perezosa, una sola instancia) ──────────────────────────

_model = None
_model_lock = Lock()

NIVEL_ORDEN = ["sin_aglomeracion", "bajo", "medio", "alto"]


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO("yolov8n.pt")
    return _model


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
    results = model(frame, classes=[0], conf=conf_min, verbose=False)[0]

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
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        callback({"tipo": "error", "mensaje": "No se pudo abrir el archivo de video."})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = 0

    zonas_exc = estado.zonas_exclusion
    umbral_m = estado.umbral_medio
    umbral_a = estado.umbral_alto

    # Procesar ~8 fps máximo para mantener latencia razonable
    saltar = max(1, int(fps / 8))

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
