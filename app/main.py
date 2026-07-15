import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core import singleton_lock
from app.core.rtsp_manager import cancel_all_sessions
from app.database import close_pool, init_pool
from app.routers import alertas, analisis, auth, camaras, fuentes_video, grabaciones, monitoreo, zonas_exclusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Código que se ejecuta al ARRANCAR y APAGAR la app.
    - Al iniciar: crea carpetas y abre el pool de BD.
    - Al cerrar: libera las conexiones del pool.
    """
    # Startup
    # Primero que nada: si ya hay otra instancia corriendo (ej. un proceso
    # huérfano de un cierre anterior que no murió del todo), esto falla
    # ruidosamente acá en vez de dejar que dos procesos compitan en silencio
    # por la GPU/CPU sin que nadie se entere hasta que la laptop se sienta lenta.
    singleton_lock.acquire()

    os.makedirs(settings.grabaciones_folder, exist_ok=True)
    os.makedirs(settings.zonas_frames_folder, exist_ok=True)
    os.makedirs("uploads/evidencias", exist_ok=True)
    init_pool()

    # Un proceso recién iniciado no puede tener hilos de sesión vivos: cualquier
    # sesión 'activo' en la BD a esta altura quedó así por un cierre anterior
    # que no pasó por /detener (crash, Ctrl+C, kill). Se limpian para que no
    # se sigan acumulando sesiones fantasma en cada reinicio.
    from app.repositories import monitoreo_repo
    limpiadas = monitoreo_repo.detener_todas_las_sesiones_activas()
    if limpiadas:
        logger.info("Limpiadas %d sesión(es) activa(s) huérfanas de un cierre anterior", limpiadas)

    # Cargar el modelo YOLO y correr una inferencia de prueba ahora, no en la
    # primera cámara que abra un usuario — evita ese retraso extra (carga a
    # GPU + autotune de kernels CUDA) la primera vez que alguien conecta.
    import time as _time
    from detector.yolo_detector import DEVICE as _device, warmup as warmup_modelo
    t0 = _time.time()
    try:
        warmup_modelo()
        logger.info("Modelo YOLO precargado en %s (%.2fs)", _device, _time.time() - t0)
    except Exception:
        # El warmup es solo una optimización (evita el retraso en la primera
        # detección real). Si falla, no debe tumbar el arranque de toda la app.
        logger.exception("No se pudo precargar el modelo YOLO en el warmup")

    logger.info("App lista ✓")

    yield  # ← la app corre aquí

    # Shutdown
    # Avisar a los hilos de RTSP/video-previa que paren ANTES de cerrar el pool
    # de BD: si siguen corriendo, pueden quedar usando una conexión justo
    # cuando closeall() la cierra, dejando el apagado colgado indefinidamente.
    cancel_all_sessions()
    close_pool()
    singleton_lock.release()
    logger.info("App apagada ✓")


app = FastAPI(
    title="API de Detección de Aglomeraciones",
    version="2.0.0",
    description=(
        "Detecta y clasifica aglomeraciones en videos usando YOLOv8. "
        "Autenticación JWT requerida en la mayoría de endpoints."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(camaras.router)
app.include_router(fuentes_video.router)
app.include_router(grabaciones.router)
app.include_router(monitoreo.router)
app.include_router(zonas_exclusion.router)
app.include_router(analisis.router)
app.include_router(alertas.router)


os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/", tags=["Estado"])
def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "docs": "/docs",
    }
