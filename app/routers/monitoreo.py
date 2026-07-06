from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_auth
from app.models.schemas import MonitoreoIniciarRequest, MonitoreoOut
from app.repositories import camara_repo, grabacion_repo, monitoreo_repo, zona_exclusion_repo

router = APIRouter(prefix="/api/monitoreo", tags=["Monitoreo"])

_MENSAJES = {
    "webcam": "Monitoreo iniciado. El video será capturado por el navegador.",
    "grabacion_previa": "Monitoreo iniciado. Usa el endpoint de análisis para procesar el video.",
    "camara_ip": "Sesión de cámara IP iniciada. Conecta el stream RTSP.",
}


def _row(s: tuple) -> dict:
    # índices: 0=id, 1=usuario_id, 2=tipo_fuente, 3=camara_id,
    #          4=grabacion_id, 5=zona_exclusion_id, 6=estado, 7=fecha_inicio, 8=fecha_fin
    return {
        "id": s[0],
        "estado": s[6],
        "tipo_fuente": s[2],
        "zona_exclusion_id": s[5],
        "mensaje": "",
    }


@router.post("/iniciar", response_model=MonitoreoOut, status_code=201)
def iniciar_monitoreo(
    data: MonitoreoIniciarRequest,
    payload: dict = Depends(require_auth),
):
    if data.tipo_fuente == "camara_ip":
        if not data.camara_id:
            raise HTTPException(status_code=422, detail="Debes indicar camara_id para tipo camara_ip.")
        if not camara_repo.get_camara(data.camara_id):
            raise HTTPException(status_code=404, detail="Cámara no encontrada.")

    if data.tipo_fuente == "grabacion_previa" and data.grabacion_id:
        if not grabacion_repo.get_grabacion(data.grabacion_id):
            raise HTTPException(status_code=404, detail="Grabación no encontrada.")

    if data.zona_exclusion_id:
        zona = zona_exclusion_repo.get_zona(data.zona_exclusion_id)
        if zona is None or not zona[9]:   # zona[9] = activa
            raise HTTPException(status_code=404, detail="Zona de exclusión no encontrada.")

    sesion = monitoreo_repo.create_sesion(
        usuario_id=int(payload["sub"]),
        tipo_fuente=data.tipo_fuente,
        camara_id=data.camara_id,
        grabacion_id=data.grabacion_id,
        zona_exclusion_id=data.zona_exclusion_id,
    )

    result = _row(sesion)
    result["mensaje"] = _MENSAJES[data.tipo_fuente]
    return result


@router.post("/{sesion_id}/detener", response_model=MonitoreoOut)
def detener_monitoreo(
    sesion_id: int,
    payload: dict = Depends(require_auth),
):
    from detector.yolo_detector import eliminar_estado
    from app.repositories import analisis_repo
    import datetime

    sesion = monitoreo_repo.get_sesion(sesion_id)
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión de monitoreo no encontrada.")
    if sesion[6] != "activo":   # sesion[6] = estado
        raise HTTPException(status_code=409, detail="La sesión ya está detenida.")

    # Cancelar hilo RTSP si es sesión de cámara IP
    if sesion[2] == "camara_ip":
        from app.core.rtsp_manager import cancel_session as rtsp_cancel
        rtsp_cancel(sesion_id)

    # Guardar resultado de análisis en BD si había estado activo (webcam o camara_ip)
    estado = eliminar_estado(sesion_id)
    if estado and estado.frames_procesados > 0:
        import os
        import logging
        zona_id = sesion[5]  # sesion[5] = zona_exclusion_id

        # RF-5.2: persistir frame con mayor concentración si existe
        evidencia_path = None
        if estado.frame_evidencia_bytes:
            try:
                os.makedirs("uploads/evidencias", exist_ok=True)
                evidencia_path = f"uploads/evidencias/sesion_{sesion_id}.jpg"
                with open(evidencia_path, "wb") as f:
                    f.write(estado.frame_evidencia_bytes)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Error al guardar frame de evidencia (sesion_id=%s)", sesion_id
                )

        analisis_repo.save_resultado(
            sesion_id=sesion_id,
            zona_config_id=zona_id,
            personas_maximas=estado.personas_maximas,
            nivel_maximo=estado.nivel_maximo,
            tiempo_primera_media_seg=estado.tiempo_primera_media,
            alerta_activada=estado.alerta_activada,
            frames_procesados=estado.frames_procesados,
            inicio_analisis=datetime.datetime.fromtimestamp(estado.inicio).isoformat(),
            fin_analisis=datetime.datetime.now().isoformat(),
            frame_evidencia=evidencia_path,
        )

        if zona_id is not None:
            from app.repositories import heatmap_repo
            from detector.yolo_detector import HEATMAP_GRID_ALTO, HEATMAP_GRID_ANCHO
            try:
                heatmap_repo.acumular_heatmap(
                    zona_id, estado.heatmap_grid, HEATMAP_GRID_ANCHO, HEATMAP_GRID_ALTO
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Error al acumular heatmap (sesion_id=%s, zona_id=%s)", sesion_id, zona_id
                )

    sesion = monitoreo_repo.detener_sesion(sesion_id)
    result = _row(sesion)
    result["mensaje"] = "Monitoreo detenido correctamente."
    return result
