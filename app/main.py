import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
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
    os.makedirs(settings.grabaciones_folder, exist_ok=True)
    os.makedirs(settings.zonas_frames_folder, exist_ok=True)
    os.makedirs("uploads/evidencias", exist_ok=True)
    init_pool()
    logger.info("App lista ✓")

    yield  # ← la app corre aquí

    # Shutdown
    close_pool()
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
