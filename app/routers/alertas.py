"""
EP-004 — Endpoints de alertas de aglomeración.

GET   /api/alertas/stream       → SSE en tiempo real para el cliente conectado (RF-4.1)
GET   /api/alertas              → historial de alertas del usuario (RF-4.5)
PATCH /api/alertas/{id}/atender → marcar una alerta como atendida (RF-4.4)

El endpoint /stream mantiene una conexión HTTP persistente abierta.
El frontend lo consume con fetch + ReadableStream (no EventSource nativo,
que no admite cabeceras personalizadas).

Formato de evento de alerta:
  data: {"tipo":"alerta","id":5,"sesion_id":3,"nivel":"alto","personas":8,"fecha_alerta":"..."}

Keepalive cada 15 s (previene cierres por proxies/load balancers):
  : keepalive
"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.security import require_auth
from app.core.sse_manager import alerta_manager
from app.models.schemas import AlertaOut, AlertasListOut
from app.repositories import alerta_repo

router = APIRouter(prefix="/api/alertas", tags=["Alertas EP-004"])

_KEEPALIVE_INTERVAL = 15  # segundos


def _row(r: tuple) -> dict:
    # 0=id, 1=sesion_id, 2=usuario_id, 3=zona_config_id, 4=nivel,
    # 5=personas, 6=atendida, 7=fecha_alerta, 8=fecha_atencion
    return {
        "id": r[0],
        "sesion_id": r[1],
        "usuario_id": r[2],
        "zona_config_id": r[3],
        "nivel": r[4],
        "personas": r[5],
        "atendida": r[6],
        "fecha_alerta": r[7].isoformat() if r[7] else None,
        "fecha_atencion": r[8].isoformat() if r[8] else None,
    }


# ── GET /stream ───────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream_alertas(payload: dict = Depends(require_auth)):
    """
    RF-4.1: Emite alertas en tiempo real vía SSE para el usuario autenticado.
    La conexión se mantiene abierta indefinidamente hasta que el cliente la cierre.
    Se limpia automáticamente de la suscripción al desconectar.
    """
    usuario_id = int(payload["sub"])
    q = alerta_manager.subscribe(usuario_id)

    async def generate():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_INTERVAL)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            alerta_manager.unsubscribe(usuario_id, q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET / ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=AlertasListOut)
def listar_alertas(
    atendida: bool | None = Query(
        None,
        description="Filtrar: true = solo atendidas, false = solo pendientes, omitir = todas",
    ),
    payload: dict = Depends(require_auth),
):
    """RF-4.5: Historial de alertas. Admin ve todas; vigilante solo las propias."""
    es_admin = payload.get("rol") == "administrador"
    usuario_id = int(payload["sub"])
    rows = alerta_repo.list_alertas(
        usuario_id=None if es_admin else usuario_id,
        atendida=atendida,
    )
    return {"alertas": [_row(r) for r in rows]}


# ── PATCH /{alerta_id}/atender ────────────────────────────────────────────────

@router.patch("/{alerta_id}/atender", response_model=AlertaOut)
def atender_alerta(alerta_id: int, payload: dict = Depends(require_auth)):
    """RF-4.4: Marcar una alerta como atendida."""
    row = alerta_repo.get_alerta(alerta_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Alerta no encontrada.")

    es_admin = payload.get("rol") == "administrador"
    if not es_admin and row[2] != int(payload["sub"]):   # row[2] = usuario_id
        raise HTTPException(status_code=403, detail="No tienes acceso a esta alerta.")

    updated = alerta_repo.marcar_atendida(alerta_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="No se pudo actualizar la alerta.")
    return _row(updated)
