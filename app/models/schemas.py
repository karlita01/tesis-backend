from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, model_validator


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    nombre: str = Field(..., min_length=2, max_length=100, examples=["Ana García"])
    email: EmailStr = Field(..., examples=["ana@ejemplo.com"])
    password: str = Field(..., min_length=6, examples=["s3cr3to"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    nombre: str
    email: str
    rol: str


class AuthResponse(BaseModel):
    token: str
    usuario: UserOut


# ── Cámaras IP ───────────────────────────────────────────────────────────────

class CamaraCreate(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=100)
    direccion_ip: str = Field(..., min_length=7, max_length=255)
    ubicacion: str = Field(..., min_length=1, max_length=255)
    descripcion: str | None = None
    activa: bool = True
    rtsp_usuario: str = Field("admin", max_length=100)
    rtsp_password: str | None = Field(None, max_length=100)
    rtsp_puerto: int = Field(554, ge=1, le=65535)
    rtsp_canal: int = Field(1, ge=1)
    rtsp_subtipo: int = Field(1, ge=0)


class CamaraOut(BaseModel):
    id: int
    nombre: str
    direccion_ip: str
    ubicacion: str
    descripcion: str | None
    activa: bool
    fecha_registro: str | None
    rtsp_usuario: str
    rtsp_tiene_password: bool   # no se devuelve la contraseña, solo si existe
    rtsp_puerto: int
    rtsp_canal: int
    rtsp_subtipo: int


class CamaraEstadoUpdate(BaseModel):
    activa: bool


# ── Fuentes de video ──────────────────────────────────────────────────────────

class FuenteInfo(BaseModel):
    tipo: str
    nombre: str
    disponible: bool
    nota: str | None = None


class FuentesVideoOut(BaseModel):
    fuentes_disponibles: list[FuenteInfo]
    camaras_ip: list[CamaraOut]


class SeleccionFuenteRequest(BaseModel):
    tipo: Literal["webcam", "grabacion_previa", "camara_ip"]
    camara_id: int | None = None
    grabacion_id: int | None = None


class SeleccionFuenteOut(BaseModel):
    tipo: str
    mensaje: str
    camara_id: int | None = None
    grabacion_id: int | None = None


# ── Grabaciones ───────────────────────────────────────────────────────────────

class GrabacionOut(BaseModel):
    id: int
    nombre_archivo: str
    ruta_archivo: str
    tipo_contenido: str | None
    tamanio_bytes: int | None
    usuario_id: int | None
    fecha_carga: str | None
    fecha_grabacion: str | None   # hora real en que fue filmada (opcional)


class GrabacionesListOut(BaseModel):
    grabaciones: list[GrabacionOut]


# ── Zonas de exclusión ────────────────────────────────────────────────────────

class ZonaRect(BaseModel):
    """Rectángulo normalizado [0-1]. x+width ≤ 1 e y+height ≤ 1."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(self) -> "ZonaRect":
        if round(self.x + self.width, 10) > 1.0:
            raise ValueError("x + width no puede exceder 1.0")
        if round(self.y + self.height, 10) > 1.0:
            raise ValueError("y + height no puede exceder 1.0")
        return self


class ZonaExclusionOut(BaseModel):
    id: int
    nombre: str
    frame_referencia: str
    zonas: list[ZonaRect]
    umbral_medio: int
    umbral_alto: int
    ventana_segundos: float
    cooldown_segundos: int
    creado_por: int | None
    activa: bool
    fecha_creacion: str | None
    fecha_actualizacion: str | None


class ZonasExclusionListOut(BaseModel):
    configuraciones: list[ZonaExclusionOut]


# ── Sesiones de monitoreo ─────────────────────────────────────────────────────

class MonitoreoIniciarRequest(BaseModel):
    tipo_fuente: Literal["webcam", "grabacion_previa", "camara_ip"]
    camara_id: int | None = None
    grabacion_id: int | None = None
    zona_exclusion_id: int | None = None


class MonitoreoOut(BaseModel):
    id: int
    estado: str
    tipo_fuente: str
    zona_exclusion_id: int | None
    mensaje: str


# ── Análisis — EP-003 ─────────────────────────────────────────────────────────

class DeteccionOut(BaseModel):
    x1: float   # normalizado 0-1
    y1: float
    x2: float
    y2: float
    conf: float
    excluida: bool


class FrameAnalisisResult(BaseModel):
    """Respuesta del endpoint POST /api/analisis/frame (webcam)."""
    sesion_id: int
    personas: int
    nivel: str
    alerta: bool
    detecciones: list[DeteccionOut]
    # Estado acumulado de la sesión
    personas_maximas: int
    nivel_maximo: str
    tiempo_primera_media_seg: float | None
    alerta_activada: bool


class ResultadoAnalisisOut(BaseModel):
    id: int
    sesion_id: int
    zona_config_id: int | None
    zona_nombre: str | None          # nombre del sector (de la zona de exclusión)
    personas_maximas: int
    nivel_maximo: str
    tiempo_primera_media_seg: float | None
    alerta_activada: bool
    frames_procesados: int
    inicio_analisis: str | None
    fin_analisis: str | None
    fecha_registro: str | None
    frame_evidencia: str | None      # ruta relativa al frame de mayor concentración
    tipo_dia: str | None             # "Laborable" | "Fin de semana"


class HistorialAnalisisOut(BaseModel):
    resultados: list[ResultadoAnalisisOut]


class ZonaCriticaOut(BaseModel):
    zona_id: int
    zona_nombre: str
    total_sesiones: int
    sesiones_con_alerta: int
    max_personas: int
    promedio_personas: int


class ZonasCriticasOut(BaseModel):
    zonas: list[ZonaCriticaOut]


# ── Alertas — EP-004 ──────────────────────────────────────────────────────────

class AlertaOut(BaseModel):
    id: int
    sesion_id: int
    usuario_id: int | None
    zona_config_id: int | None
    nivel: str
    personas: int
    atendida: bool
    fecha_alerta: str | None
    fecha_atencion: str | None


class AlertasListOut(BaseModel):
    alertas: list[AlertaOut]
