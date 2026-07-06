from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_admin, require_auth
from app.models.schemas import CamaraCreate, CamaraEstadoUpdate, CamaraOut
from app.repositories import camara_repo

router = APIRouter(prefix="/api/camaras", tags=["Cámaras IP"])


def _row(c: tuple) -> dict:
    # 0=id, 1=nombre, 2=direccion_ip, 3=ubicacion, 4=descripcion,
    # 5=activa, 6=fecha_registro, 7=rtsp_usuario, 8=rtsp_password,
    # 9=rtsp_puerto, 10=rtsp_canal, 11=rtsp_subtipo
    return {
        "id": c[0],
        "nombre": c[1],
        "direccion_ip": c[2],
        "ubicacion": c[3],
        "descripcion": c[4],
        "activa": c[5],
        "fecha_registro": c[6].isoformat() if c[6] else None,
        "rtsp_usuario": c[7] or "admin",
        "rtsp_tiene_password": c[8] is not None and c[8] != "",
        "rtsp_puerto": c[9] or 554,
        "rtsp_canal": c[10] or 1,
        "rtsp_subtipo": c[11] if c[11] is not None else 1,
    }


@router.post("", response_model=CamaraOut, status_code=201)
def registrar_camara(
    data: CamaraCreate,
    _: dict = Depends(require_admin),
):
    """Registra una cámara IP con credenciales RTSP."""
    camara = camara_repo.create_camara(
        nombre=data.nombre,
        direccion_ip=data.direccion_ip,
        ubicacion=data.ubicacion,
        descripcion=data.descripcion,
        activa=data.activa,
        rtsp_usuario=data.rtsp_usuario,
        rtsp_password=data.rtsp_password,
        rtsp_puerto=data.rtsp_puerto,
        rtsp_canal=data.rtsp_canal,
        rtsp_subtipo=data.rtsp_subtipo,
    )
    return _row(camara)


@router.get("", response_model=list[CamaraOut])
def listar_camaras(_: dict = Depends(require_auth)):
    """Lista todas las cámaras registradas. Accesible por cualquier usuario autenticado."""
    return [_row(c) for c in camara_repo.list_camaras()]


@router.patch("/{camara_id}/estado", response_model=CamaraOut)
def actualizar_estado(
    camara_id: int,
    data: CamaraEstadoUpdate,
    _: dict = Depends(require_admin),
):
    """Activa o desactiva una cámara. Solo administrador."""
    camara = camara_repo.update_estado(camara_id, data.activa)
    if not camara:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    return _row(camara)


@router.delete("/{camara_id}", status_code=204)
def eliminar_camara(
    camara_id: int,
    _: dict = Depends(require_admin),
):
    """Elimina una cámara registrada. Solo administrador."""
    if not camara_repo.get_camara(camara_id):
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    camara_repo.delete_camara(camara_id)
