import json
import os
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.security import require_admin
from app.models.schemas import ZonaExclusionOut, ZonaRect, ZonasExclusionListOut
from app.repositories import zona_exclusion_repo

router = APIRouter(prefix="/api/zonas-exclusion", tags=["Zonas de exclusión"])

_EXTENSIONES_FRAME = {".jpg", ".jpeg", ".png", ".webp"}


def _row(r: tuple) -> dict:
    # índices: 0=id, 1=nombre, 2=frame_referencia, 3=zonas,
    #          4=umbral_medio, 5=umbral_alto, 6=ventana_segundos, 7=cooldown_segundos,
    #          8=creado_por, 9=activa, 10=fecha_creacion, 11=fecha_actualizacion
    return {
        "id": r[0],
        "nombre": r[1],
        "frame_referencia": r[2],
        "zonas": r[3],
        "umbral_medio": r[4],
        "umbral_alto": r[5],
        "ventana_segundos": float(r[6]),
        "cooldown_segundos": r[7],
        "creado_por": r[8],
        "activa": r[9],
        "fecha_creacion": r[10].isoformat() if r[10] else None,
        "fecha_actualizacion": r[11].isoformat() if r[11] else None,
    }


def _parse_zonas(zonas_json: str) -> list[dict]:
    try:
        data = json.loads(zonas_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="El campo 'zonas' no es JSON válido.")
    if not isinstance(data, list):
        raise HTTPException(status_code=422, detail="El campo 'zonas' debe ser una lista.")
    if len(data) == 0:
        raise HTTPException(status_code=422, detail="El campo 'zonas' no puede estar vacío.")
    resultado = []
    for i, rect in enumerate(data):
        try:
            resultado.append(ZonaRect.model_validate(rect).model_dump())
        except ValidationError as exc:
            msg = exc.errors()[0]["msg"]
            raise HTTPException(status_code=422, detail=f"Rectángulo #{i}: {msg}")
    return resultado


def _guardar_frame(file: UploadFile, folder: str) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _EXTENSIONES_FRAME:
        raise HTTPException(
            status_code=422,
            detail=f"Extensión no permitida para frame. Usa: {', '.join(sorted(_EXTENSIONES_FRAME))}",
        )
    nombre_unico = f"{uuid.uuid4()}{ext}"
    ruta = os.path.join(folder, nombre_unico)
    with open(ruta, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return ruta


@router.post("", response_model=ZonaExclusionOut, status_code=201)
async def crear_zona(
    nombre: str = Form(..., min_length=1, max_length=120),
    frame: UploadFile = File(...),
    zonas: str = Form(..., description="Lista JSON de rectángulos normalizados"),
    umbral_medio: int = Form(4, ge=1, description="Personas mínimas para nivel Medio"),
    umbral_alto: int = Form(6, ge=1, description="Personas mínimas para nivel Alto"),
    ventana_segundos: float = Form(2.0, ge=0.5, description="Duración ventana deslizante (s)"),
    cooldown_segundos: int = Form(10, ge=1, description="Pausa mínima entre alertas (s)"),
    payload: dict = Depends(require_admin),
    settings: Settings = Depends(get_settings),
):
    if umbral_alto <= umbral_medio:
        raise HTTPException(
            status_code=422,
            detail="umbral_alto debe ser mayor que umbral_medio.",
        )
    zonas_list = _parse_zonas(zonas)
    ruta_frame = _guardar_frame(frame, settings.zonas_frames_folder)
    row = zona_exclusion_repo.create_zona(
        nombre=nombre.strip(),
        frame_referencia=ruta_frame,
        zonas=zonas_list,
        creado_por=int(payload["sub"]),
        umbral_medio=umbral_medio,
        umbral_alto=umbral_alto,
        ventana_segundos=ventana_segundos,
        cooldown_segundos=cooldown_segundos,
    )
    return _row(row)


@router.get("", response_model=ZonasExclusionListOut)
def listar_zonas(payload: dict = Depends(require_admin)):
    rows = zona_exclusion_repo.list_zonas()
    return {"configuraciones": [_row(r) for r in rows]}


@router.get("/{zona_id}", response_model=ZonaExclusionOut)
def obtener_zona(zona_id: int, payload: dict = Depends(require_admin)):
    row = zona_exclusion_repo.get_zona(zona_id)
    if row is None or not row[9]:   # row[9] = activa
        raise HTTPException(status_code=404, detail="Configuración no encontrada.")
    return _row(row)


@router.put("/{zona_id}", response_model=ZonaExclusionOut)
async def actualizar_zona(
    zona_id: int,
    nombre: str | None = Form(None),
    frame: UploadFile | None = File(None),
    zonas: str | None = Form(None),
    umbral_medio: int | None = Form(None, ge=1),
    umbral_alto: int | None = Form(None, ge=1),
    ventana_segundos: float | None = Form(None, ge=0.5),
    cooldown_segundos: int | None = Form(None, ge=1),
    payload: dict = Depends(require_admin),
    settings: Settings = Depends(get_settings),
):
    existing = zona_exclusion_repo.get_zona(zona_id)
    if existing is None or not existing[9]:
        raise HTTPException(status_code=404, detail="Configuración no encontrada.")

    if umbral_medio is not None and umbral_alto is not None and umbral_alto <= umbral_medio:
        raise HTTPException(status_code=422, detail="umbral_alto debe ser mayor que umbral_medio.")

    nombre_val = nombre.strip() if nombre else None
    zonas_val = _parse_zonas(zonas) if zonas else None
    frame_val = None
    if frame and frame.filename:
        frame_val = _guardar_frame(frame, settings.zonas_frames_folder)

    campos = [nombre_val, zonas_val, frame_val, umbral_medio, umbral_alto, ventana_segundos, cooldown_segundos]
    if all(v is None for v in campos):
        raise HTTPException(status_code=422, detail="Envía al menos un campo para actualizar.")

    row = zona_exclusion_repo.update_zona(
        zona_id=zona_id,
        nombre=nombre_val,
        frame_referencia=frame_val,
        zonas=zonas_val,
        umbral_medio=umbral_medio,
        umbral_alto=umbral_alto,
        ventana_segundos=ventana_segundos,
        cooldown_segundos=cooldown_segundos,
    )
    return _row(row)


@router.delete("/{zona_id}")
def eliminar_zona(zona_id: int, payload: dict = Depends(require_admin)):
    existing = zona_exclusion_repo.get_zona(zona_id)
    if existing is None or not existing[9]:
        raise HTTPException(status_code=404, detail="Configuración no encontrada.")
    zona_exclusion_repo.delete_zona(zona_id)
    return {"message": "Configuración de zonas eliminada correctamente."}
