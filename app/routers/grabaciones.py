import os
import re
import shutil
import time
import uuid
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.config import Settings, get_settings
from app.core.security import decode_token, require_auth
from app.models.schemas import GrabacionOut, GrabacionesListOut
from app.repositories import grabacion_repo

router = APIRouter(prefix="/api/grabaciones", tags=["Grabaciones"])

_EXTENSIONES = {".mp4", ".avi", ".mov", ".mkv"}
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _row(g: tuple) -> dict:
    # índices: 0=id, 1=nombre_archivo, 2=ruta_archivo, 3=tipo_contenido,
    #          4=tamanio_bytes, 5=usuario_id, 6=fecha_carga, 7=fecha_grabacion
    return {
        "id": g[0],
        "nombre_archivo": g[1],
        "ruta_archivo": g[2],
        "tipo_contenido": g[3],
        "tamanio_bytes": g[4],
        "usuario_id": g[5],
        "fecha_carga": g[6].isoformat() if g[6] else None,
        "fecha_grabacion": g[7].isoformat() if g[7] else None,
    }


@router.post("", response_model=GrabacionOut, status_code=201)
async def cargar_grabacion(
    file: UploadFile = File(...),
    fecha_grabacion: str | None = Form(None, description="ISO 8601: 2025-06-15T14:30"),
    payload: dict = Depends(require_auth),
    settings: Settings = Depends(get_settings),
):
    """
    Sube una grabación de video al servidor.
    El campo fecha_grabacion (opcional) indica cuándo fue filmado el video.
    Extensiones: .mp4, .avi, .mov, .mkv
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _EXTENSIONES:
        raise HTTPException(
            status_code=422,
            detail=f"Extensión no permitida. Usa: {', '.join(sorted(_EXTENSIONES))}",
        )

    nombre_unico = f"{uuid.uuid4()}{ext}"
    ruta = os.path.join(settings.grabaciones_folder, nombre_unico)

    with open(ruta, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    grabacion = grabacion_repo.create_grabacion(
        nombre_archivo=file.filename or nombre_unico,
        ruta_archivo=ruta,
        tipo_contenido=file.content_type,
        tamanio_bytes=os.path.getsize(ruta),
        usuario_id=int(payload["sub"]),
        fecha_grabacion=fecha_grabacion or None,
    )
    return _row(grabacion)


@router.get("/{grabacion_id}/file")
def servir_grabacion(
    grabacion_id: int,
    request: Request,
    token: str = Query(..., description="JWT Bearer token como query param"),
):
    """
    Sirve el archivo de video para reproducción en el navegador.
    Acepta el token como query param porque <video> no soporta headers personalizados.
    Soporta HTTP Range requests (206 Partial Content) para permitir streaming
    progresivo y seek sin descargar el archivo completo primero — Starlette's
    FileResponse no implementa Range en la versión instalada.
    """
    try:
        payload = decode_token(token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Token inválido.")

    grabacion = grabacion_repo.get_grabacion(grabacion_id)
    if grabacion is None:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")

    usuario_id = int(payload["sub"])
    rol = payload.get("rol", "")
    if grabacion[5] != usuario_id and rol != "administrador":
        raise HTTPException(status_code=403, detail="Sin acceso a esta grabación.")

    ruta = grabacion[2]
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="Archivo no encontrado en disco.")

    nombre = grabacion[1]
    media_type = grabacion[3] or "video/mp4"
    file_size = os.path.getsize(ruta)

    range_header = request.headers.get("range")
    if range_header is None:
        return FileResponse(
            ruta,
            media_type=media_type,
            filename=nombre,
            content_disposition_type="inline",
            headers={"accept-ranges": "bytes"},
        )

    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match or (match.group(1) == "" and match.group(2) == ""):
        raise HTTPException(status_code=416, detail="Encabezado Range inválido.")

    start_str, end_str = match.group(1), match.group(2)
    if start_str == "":
        start = max(file_size - int(end_str), 0)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str != "" else file_size - 1
    end = min(end, file_size - 1)

    if start > end or start >= file_size:
        raise HTTPException(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    def iterfile():
        with open(ruta, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iterfile(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Disposition": f"inline; filename*=utf-8''{quote(nombre)}",
        },
    )


@router.delete("/{grabacion_id}", status_code=204)
def eliminar_grabacion(
    grabacion_id: int,
    payload: dict = Depends(require_auth),
):
    """
    Elimina el registro de una grabación (y su archivo en disco, si existe).
    Solo el dueño o un administrador. Útil para limpiar registros cuyo
    archivo quedó en otra máquina (ej. tras un git clone en otra PC).
    """
    grabacion = grabacion_repo.get_grabacion(grabacion_id)
    if grabacion is None:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")

    usuario_id = int(payload["sub"])
    rol = payload.get("rol", "")
    if grabacion[5] != usuario_id and rol != "administrador":
        raise HTTPException(status_code=403, detail="Sin acceso a esta grabación.")

    ruta = grabacion[2]
    # Si el análisis de esta grabación acaba de detenerse, el hilo en
    # background puede tardar un instante en soltar el archivo (Windows
    # bloquea el .mp4 mientras cv2.VideoCapture lo tiene abierto — incluso
    # os.path.exists() puede lanzar PermissionError, no solo os.remove()).
    # Reintentamos brevemente antes de fallar.
    ultimo_error: PermissionError | None = None
    for intento in range(8):
        try:
            if os.path.exists(ruta):
                os.remove(ruta)
            ultimo_error = None
            break
        except PermissionError as exc:
            ultimo_error = exc
            time.sleep(0.3)
    if ultimo_error is not None:
        raise HTTPException(
            status_code=409,
            detail="El archivo de video sigue en uso (análisis reciente). Intenta de nuevo en unos segundos.",
        )

    grabacion_repo.delete_grabacion(grabacion_id)


@router.get("", response_model=GrabacionesListOut)
def listar_grabaciones(payload: dict = Depends(require_auth)):
    """
    Lista grabaciones.
    - Administrador: ve todas.
    - Vigilante: ve solo las propias.
    """
    es_admin = payload.get("rol") == "administrador"
    usuario_id = int(payload["sub"])
    grabaciones = grabacion_repo.list_grabaciones(
        usuario_id=None if es_admin else usuario_id
    )
    return {"grabaciones": [_row(g) for g in grabaciones]}
