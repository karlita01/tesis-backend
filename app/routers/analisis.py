"""
EP-003 — Endpoints de análisis de aglomeraciones.

POST /api/analisis/frame           → procesa un frame de webcam (tiempo real)
GET  /api/analisis/video/{id}/stream → SSE de análisis de grabación previa
GET  /api/analisis/historial        → historial de resultados
GET  /api/analisis/resultado/{id}   → resultado de una sesión específica
"""

import asyncio
import datetime
import json
import logging
import os
import threading

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.security import require_auth
from app.core.sse_manager import alerta_manager
from app.models.schemas import (
    FrameAnalisisResult,
    HistorialAnalisisOut,
    ResultadoAnalisisOut,
    ZonasCriticasOut,
)
from app.repositories import alerta_repo, analisis_repo, grabacion_repo, monitoreo_repo, zona_exclusion_repo
from detector.yolo_detector import (
    crear_estado,
    eliminar_estado,
    obtener_estado,
    procesar_frame,
    procesar_video_sync,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analisis", tags=["Análisis EP-003"])


def _zona_config_dict(zona_id: int | None) -> dict | None:
    """Convierte una fila de zona_exclusion en el dict que espera el detector."""
    if zona_id is None:
        return None
    row = zona_exclusion_repo.get_zona(zona_id)
    if row is None or not row[9]:   # row[9] = activa
        return None
    return {
        "zonas": row[3],            # JSONB → list[dict]
        "umbral_medio": row[4],
        "umbral_alto": row[5],
        "ventana_segundos": float(row[6]),
        "cooldown_segundos": row[7],
    }


def _tipo_dia(dt) -> str | None:
    if dt is None:
        return None
    dow = dt.weekday()  # 0=lunes … 6=domingo
    return "Fin de semana" if dow >= 5 else "Laborable"


def _save_evidencia(sesion_id: int, frame_bytes: bytes) -> str:
    """Guarda el frame de evidencia en disco y devuelve la ruta relativa."""
    os.makedirs("uploads/evidencias", exist_ok=True)
    ruta = f"uploads/evidencias/sesion_{sesion_id}.jpg"
    with open(ruta, "wb") as f:
        f.write(frame_bytes)
    return ruta


def _row_resultado(r: tuple) -> dict:
    # 0=id, 1=sesion_id, 2=zona_config_id, 3=personas_maximas, 4=nivel_maximo,
    # 5=tiempo_primera_media_seg, 6=alerta_activada, 7=frames_procesados,
    # 8=inicio_analisis, 9=fin_analisis, 10=fecha_registro, 11=frame_evidencia,
    # 12=zona_nombre  (presente en queries con JOIN)
    return {
        "id": r[0],
        "sesion_id": r[1],
        "zona_config_id": r[2],
        "zona_nombre": r[12] if len(r) > 12 else None,
        "personas_maximas": r[3],
        "nivel_maximo": r[4],
        "tiempo_primera_media_seg": r[5],
        "alerta_activada": r[6],
        "frames_procesados": r[7],
        "inicio_analisis": r[8].isoformat() if r[8] else None,
        "fin_analisis": r[9].isoformat() if r[9] else None,
        "fecha_registro": r[10].isoformat() if r[10] else None,
        "frame_evidencia": r[11] if len(r) > 11 else None,
        "tipo_dia": _tipo_dia(r[8]),
    }


# ── POST /frame — análisis de webcam ─────────────────────────────────────────

@router.post("/frame", response_model=FrameAnalisisResult)
async def analizar_frame(
    sesion_id: int = Form(...),
    zona_config_id: int | None = Form(None),
    frame: UploadFile = File(...),
    payload: dict = Depends(require_auth),
):
    """
    RF-3.1–RF-3.4: Procesa un frame JPEG enviado por el navegador (webcam).
    Mantiene estado acumulado por sesión en memoria.
    """
    sesion = monitoreo_repo.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if sesion[6] != "activo":   # sesion[6] = estado
        raise HTTPException(status_code=409, detail="La sesión está detenida.")
    if int(payload["sub"]) != sesion[1] and payload.get("rol") != "administrador":
        raise HTTPException(status_code=403, detail="No tienes acceso a esta sesión.")

    # Obtener o crear estado de sesión
    estado = obtener_estado(sesion_id)
    if estado is None:
        # Primera vez: cargar config de zona
        zona_id = zona_config_id or sesion[5]  # preferir parámetro, luego el de la sesión
        config = _zona_config_dict(zona_id)
        estado = crear_estado(sesion_id, zona_config=config)

    frame_bytes = await frame.read()
    resultado = procesar_frame(
        frame_bytes,
        estado.zonas_exclusion,
        estado.umbral_medio,
        estado.umbral_alto,
    )

    # RF-5.2: guardar frame si supera el máximo histórico de esta sesión
    es_nuevo_maximo = resultado["personas"] > estado.personas_maximas
    alerta = estado.actualizar(resultado["personas"], resultado["nivel"])
    if es_nuevo_maximo and resultado["personas"] > 0:
        estado.frame_evidencia_bytes = frame_bytes
    resumen = estado.resumen()

    # RF-4.3 / RF-4.1: guardar alerta en BD y notificar al cliente SSE
    if alerta:
        zona_id_activo = zona_config_id or sesion[5]
        try:
            db_alerta = alerta_repo.crear_alerta(
                sesion_id=sesion_id,
                usuario_id=int(payload["sub"]),
                zona_config_id=zona_id_activo,
                nivel="alto",
                personas=resultado["personas"],
            )
            await alerta_manager.publish(int(payload["sub"]), {
                "tipo": "alerta",
                "id": db_alerta[0],
                "sesion_id": sesion_id,
                "nivel": "alto",
                "personas": resultado["personas"],
                "fecha_alerta": db_alerta[7].isoformat() if db_alerta[7] else None,
            })
        except Exception:
            logger.exception("Error al guardar/publicar alerta (sesion_id=%s)", sesion_id)

    return {
        "sesion_id": sesion_id,
        "personas": resultado["personas"],
        "nivel": resultado["nivel"],
        "alerta": alerta,
        "detecciones": resultado["detecciones"],
        "personas_maximas": resumen["personas_maximas"],
        "nivel_maximo": resumen["nivel_maximo"],
        "tiempo_primera_media_seg": resumen["tiempo_primera_media_seg"],
        "alerta_activada": resumen["alerta_activada"],
    }


# ── GET /video/{sesion_id}/stream — SSE de grabación previa ──────────────────

@router.get("/video/{sesion_id}/stream")
async def stream_analisis_video(
    sesion_id: int,
    payload: dict = Depends(require_auth),
):
    """
    RF-3.1–RF-3.5: Procesa un video previo y emite eventos SSE con el progreso.
    El cliente debe leer el stream con fetch + ReadableStream (no EventSource nativo,
    ya que necesita el header Authorization).
    """
    sesion = monitoreo_repo.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if sesion[2] != "grabacion_previa":
        raise HTTPException(status_code=422, detail="Esta sesión no es de grabación previa.")
    if int(payload["sub"]) != sesion[1] and payload.get("rol") != "administrador":
        raise HTTPException(status_code=403, detail="No tienes acceso a esta sesión.")

    grabacion_id = sesion[4]   # sesion[4] = grabacion_id
    if grabacion_id is None:
        raise HTTPException(status_code=422, detail="La sesión no tiene una grabación asociada.")

    grabacion = grabacion_repo.get_grabacion(grabacion_id)
    if grabacion is None:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")

    ruta_video = grabacion[2]   # grabacion[2] = ruta_archivo

    zona_id = sesion[5]         # sesion[5] = zona_exclusion_id
    usuario_id = sesion[1]      # sesion[1] = usuario_id
    config = _zona_config_dict(zona_id)

    # Eliminar estado previo si existe (re-análisis)
    eliminar_estado(sesion_id)
    estado = crear_estado(sesion_id, zona_config=config)
    inicio = datetime.datetime.now()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        def on_event(ev: dict):
            asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result(timeout=30)

            # RF-4.3 / RF-4.1: guardar alerta en BD y notificar SSE desde hilo
            if ev.get("alerta"):
                try:
                    db_alerta = alerta_repo.crear_alerta(
                        sesion_id=sesion_id,
                        usuario_id=usuario_id,
                        zona_config_id=zona_id,
                        nivel="alto",
                        personas=ev.get("personas", 0),
                    )
                    alerta_manager.publish_from_thread(usuario_id, {
                        "tipo": "alerta",
                        "id": db_alerta[0],
                        "sesion_id": sesion_id,
                        "nivel": "alto",
                        "personas": ev.get("personas", 0),
                        "fecha_alerta": db_alerta[7].isoformat() if db_alerta[7] else None,
                    }, loop)
                except Exception:
                    logger.exception(
                        "Error al guardar/publicar alerta en stream (sesion_id=%s)", sesion_id
                    )

        try:
            procesar_video_sync(ruta_video, config, estado, callback=on_event)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put({"tipo": "error", "mensaje": str(exc)}), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        try:
            while True:
                ev = await asyncio.wait_for(queue.get(), timeout=300)
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            # Guardar resultado en BD
            if estado.frames_procesados > 0:
                fin_dt = datetime.datetime.now()
                # RF-5.2: persistir frame de evidencia si existe
                evidencia_path = None
                if estado.frame_evidencia_bytes:
                    try:
                        evidencia_path = _save_evidencia(sesion_id, estado.frame_evidencia_bytes)
                    except Exception:
                        logger.exception("Error al guardar frame de evidencia (sesion_id=%s)", sesion_id)
                analisis_repo.save_resultado(
                    sesion_id=sesion_id,
                    zona_config_id=zona_id,
                    personas_maximas=estado.personas_maximas,
                    nivel_maximo=estado.nivel_maximo,
                    tiempo_primera_media_seg=estado.tiempo_primera_media,
                    alerta_activada=estado.alerta_activada,
                    frames_procesados=estado.frames_procesados,
                    inicio_analisis=inicio.isoformat(),
                    fin_analisis=fin_dt.isoformat(),
                    frame_evidencia=evidencia_path,
                )
            eliminar_estado(sesion_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /historial ────────────────────────────────────────────────────────────

@router.get("/historial", response_model=HistorialAnalisisOut)
def obtener_historial(payload: dict = Depends(require_auth)):
    """Devuelve el historial de análisis. Admin ve todo; vigilante ve los suyos."""
    es_admin = payload.get("rol") == "administrador"
    usuario_id = int(payload["sub"])
    rows = analisis_repo.list_resultados(usuario_id=None if es_admin else usuario_id)
    return {"resultados": [_row_resultado(r) for r in rows]}


# ── GET /resultado/{sesion_id} ────────────────────────────────────────────────

@router.get("/resultado/{sesion_id}", response_model=ResultadoAnalisisOut)
def obtener_resultado(sesion_id: int, payload: dict = Depends(require_auth)):
    """Devuelve el resultado de análisis de una sesión."""
    sesion = monitoreo_repo.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if int(payload["sub"]) != sesion[1] and payload.get("rol") != "administrador":
        raise HTTPException(status_code=403, detail="No tienes acceso a esta sesión.")

    row = analisis_repo.get_resultado_by_sesion(sesion_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No hay resultado de análisis para esta sesión.")
    return _row_resultado(row)


# ── GET /zonas-criticas ───────────────────────────────────────────────────────

@router.get("/zonas-criticas", response_model=ZonasCriticasOut)
def obtener_zonas_criticas(_: dict = Depends(require_auth)):
    """
    RF-5.5: Resumen agregado por sector (zona de exclusión).
    Devuelve total de sesiones, sesiones con alerta, personas máximas
    y promedio de personas por zona activa.
    """
    rows = analisis_repo.get_zonas_criticas()
    return {
        "zonas": [
            {
                "zona_id": r[0],
                "zona_nombre": r[1],
                "total_sesiones": r[2],
                "sesiones_con_alerta": r[3],
                "max_personas": r[4],
                "promedio_personas": r[5],
            }
            for r in rows
        ]
    }
