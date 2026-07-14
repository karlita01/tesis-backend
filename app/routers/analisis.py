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
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.core.security import require_admin, require_auth
from app.core.sse_manager import alerta_manager
from app.models.schemas import (
    FrameAnalisisResult,
    HistorialAnalisisOut,
    ResultadoAnalisisOut,
    ZonasCriticasOut,
)
from app.core.rtsp_manager import cancel_session as rtsp_cancel, get_session as rtsp_get, set_session as rtsp_set
from app.core.security import decode_token
from app.repositories import alerta_repo, analisis_repo, camara_repo, grabacion_repo, monitoreo_repo, zona_exclusion_repo
from detector.yolo_detector import (
    crear_estado,
    eliminar_estado,
    obtener_estado,
    procesar_frame,
    procesar_rtsp_mjpeg,
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

    # Permite que POST /api/monitoreo/{id}/detener cancele de verdad el hilo
    # en background — sin esto, procesar_video_sync sigue leyendo el archivo
    # aunque el cliente se desconecte, y el .mp4 queda bloqueado en Windows
    # (ej. al intentar borrar la grabación justo después de analizarla).
    cancel_event = threading.Event()
    rtsp_set(sesion_id, {"cancelado": cancel_event})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        def on_event(ev: dict):
            # Grabación previa: no se crean alertas "en vivo" (RF-4.1/4.3) —
            # el video ya ocurrió, no hay nada que atender en tiempo real.
            # El resumen (alerta_activada) igual queda en resultados_analisis
            # vía estado.resumen(), calculado independientemente de esto.
            asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result(timeout=30)

        try:
            procesar_video_sync(ruta_video, config, estado, callback=on_event, cancelado_fn=cancel_event.is_set)
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
            rtsp_cancel(sesion_id)

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


@router.delete("/resultado/{resultado_id}", status_code=204)
def eliminar_resultado(
    resultado_id: int,
    _: dict = Depends(require_admin),
):
    """Elimina un resultado del historial (y su captura de evidencia, si existe). Solo administrador."""
    row = analisis_repo.delete_resultado(resultado_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Resultado no encontrado.")
    evidencia_path = row[0]
    if evidencia_path and os.path.exists(evidencia_path):
        try:
            os.remove(evidencia_path)
        except OSError:
            logger.exception("Error al borrar frame de evidencia (resultado_id=%s)", resultado_id)


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


# ── Helpers de sesiones RTSP ─────────────────────────────────────────────────

def _rtsp_session_start(
    sesion_id: int,
    rtsp_url: str,
    config: dict | None,
    usuario_id: int,
    zona_id: int | None,
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """Lanza el hilo RTSP y registra el estado compartido para la sesión."""
    cancelado = threading.Event()
    estado = crear_estado(sesion_id, zona_config=config)
    inicio = datetime.datetime.now()

    session_data: dict = {
        "frame": None,
        "resultado": None,
        "alerta": False,
        "lock": threading.Lock(),
        "cancelado": cancelado,
        "estado": estado,
        "inicio": inicio,
    }

    rtsp_set(sesion_id, session_data)

    def on_frame(jpeg_bytes, resultado, alerta):
        with session_data["lock"]:
            session_data["frame"] = jpeg_bytes
            session_data["resultado"] = resultado
            session_data["alerta"] = alerta

        if alerta:
            try:
                db_alerta = alerta_repo.crear_alerta(
                    sesion_id=sesion_id,
                    usuario_id=usuario_id,
                    zona_config_id=zona_id,
                    nivel="alto",
                    personas=resultado.get("personas", 0),
                )
                alerta_manager.publish_from_thread(usuario_id, {
                    "tipo": "alerta",
                    "id": db_alerta[0],
                    "sesion_id": sesion_id,
                    "nivel": "alto",
                    "personas": resultado.get("personas", 0),
                    "fecha_alerta": db_alerta[7].isoformat() if db_alerta[7] else None,
                }, loop)
            except Exception:
                logger.exception("Error al guardar alerta RTSP (sesion_id=%s)", sesion_id)

    def worker():
        procesar_rtsp_mjpeg(
            rtsp_url=rtsp_url,
            zona_config=config,
            estado=estado,
            cancelado_fn=cancelado.is_set,
            on_frame=on_frame,
        )
        rtsp_cancel(sesion_id)  # limpia la entrada del manager cuando el hilo termina

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    session_data["thread"] = t
    return session_data


def _build_rtsp_url(camara: tuple) -> str:
    # índices de camara_repo: 2=direccion_ip, 7=rtsp_usuario, 8=rtsp_password,
    #                         9=rtsp_puerto, 10=rtsp_canal, 11=rtsp_subtipo
    ip = camara[2]
    usuario = quote_plus(camara[7] or "admin")
    password = quote_plus(camara[8] or "")
    puerto = camara[9] or 554
    canal = camara[10] or 1
    subtipo = camara[11] if camara[11] is not None else 1
    cred = f"{usuario}:{password}@" if password else f"{usuario}@"
    return f"rtsp://{cred}{ip}:{puerto}/cam/realmonitor?channel={canal}&subtype={subtipo}"


# ── GET /camara/{sesion_id}/mjpeg — stream MJPEG ──────────────────────────────

@router.get("/camara/{sesion_id}/mjpeg")
async def stream_camara_mjpeg(
    sesion_id: int,
    token: str = Query(..., description="JWT token (requerido porque <img> no envía headers)"),
):
    """
    Stream MJPEG con detecciones dibujadas encima.
    Acepta el JWT como ?token= porque el elemento <img> no puede enviar headers.
    """
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")

    sesion = monitoreo_repo.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if sesion[2] != "camara_ip":
        raise HTTPException(status_code=422, detail="Esta sesión no es de cámara IP.")
    if sesion[6] != "activo":
        raise HTTPException(status_code=409, detail="La sesión no está activa.")
    if int(payload["sub"]) != sesion[1] and payload.get("rol") != "administrador":
        raise HTTPException(status_code=403, detail="No tienes acceso a esta sesión.")

    camara_id = sesion[3]  # sesion[3] = camara_id
    if camara_id is None:
        raise HTTPException(status_code=422, detail="La sesión no tiene cámara asociada.")

    camara = camara_repo.get_camara(camara_id)
    if camara is None:
        raise HTTPException(status_code=404, detail="Cámara no encontrada.")

    zona_id = sesion[5]
    usuario_id = sesion[1]
    config = _zona_config_dict(zona_id)
    loop = asyncio.get_running_loop()

    # Reutilizar hilo si ya existe, sino iniciarlo
    session_data = rtsp_get(sesion_id)
    if session_data is None or session_data["cancelado"].is_set():
        rtsp_url = _build_rtsp_url(camara)
        session_data = _rtsp_session_start(sesion_id, rtsp_url, config, usuario_id, zona_id, loop)

    BOUNDARY = b"frame"

    async def mjpeg_generate():
        try:
            while True:
                await asyncio.sleep(0.12)  # ~8 fps máximo
                with session_data["lock"]:
                    frame_bytes = session_data["frame"]
                    cancelado = session_data["cancelado"].is_set()
                if cancelado and frame_bytes is None:
                    break
                if frame_bytes is None:
                    continue
                yield (
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        finally:
            # Detener el hilo si el cliente se desconecta
            session_data["cancelado"].set()

    return StreamingResponse(
        mjpeg_generate(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── GET /camara/{sesion_id}/stream — SSE de stats ────────────────────────────

@router.get("/camara/{sesion_id}/stream")
async def stream_camara_stats(
    sesion_id: int,
    payload: dict = Depends(require_auth),
):
    """
    SSE de estadísticas en tiempo real de una sesión de cámara IP.
    Emite nivel, personas y alerta por cada frame procesado.
    """
    sesion = monitoreo_repo.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if sesion[2] != "camara_ip":
        raise HTTPException(status_code=422, detail="Esta sesión no es de cámara IP.")
    if int(payload["sub"]) != sesion[1] and payload.get("rol") != "administrador":
        raise HTTPException(status_code=403, detail="No tienes acceso a esta sesión.")

    camara_id = sesion[3]
    if camara_id is None:
        raise HTTPException(status_code=422, detail="La sesión no tiene cámara asociada.")

    camara = camara_repo.get_camara(camara_id)
    if camara is None:
        raise HTTPException(status_code=404, detail="Cámara no encontrada.")

    zona_id = sesion[5]
    usuario_id = sesion[1]
    config = _zona_config_dict(zona_id)
    loop = asyncio.get_running_loop()

    session_data = rtsp_get(sesion_id)
    if session_data is None or session_data["cancelado"].is_set():
        rtsp_url = _build_rtsp_url(camara)
        session_data = _rtsp_session_start(sesion_id, rtsp_url, config, usuario_id, zona_id, loop)

    ultimo_resultado = None
    keepalive_counter = 0

    async def sse_generate():
        nonlocal ultimo_resultado, keepalive_counter
        try:
            while True:
                await asyncio.sleep(0.15)
                keepalive_counter += 1
                if keepalive_counter >= 100:  # ~15s keepalive
                    keepalive_counter = 0
                    yield ": keepalive\n\n"
                    continue

                with session_data["lock"]:
                    resultado = session_data["resultado"]
                    alerta = session_data["alerta"]
                    cancelado = session_data["cancelado"].is_set()

                if cancelado:
                    break
                if resultado is None or resultado is ultimo_resultado:
                    continue

                ultimo_resultado = resultado
                ev = {
                    "tipo": "frame",
                    "personas": resultado.get("personas", 0),
                    "nivel": resultado.get("nivel", "sin_aglomeracion"),
                    "alerta": alerta,
                }
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            pass  # el hilo RTSP lo gestiona el endpoint MJPEG

    return StreamingResponse(
        sse_generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
