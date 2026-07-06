from app.database import get_db

_COLS = (
    "id, nombre, direccion_ip, ubicacion, descripcion, activa, fecha_registro, "
    "rtsp_usuario, rtsp_password, rtsp_puerto, rtsp_canal, rtsp_subtipo"
)
# índices: 0=id, 1=nombre, 2=direccion_ip, 3=ubicacion, 4=descripcion,
#          5=activa, 6=fecha_registro, 7=rtsp_usuario, 8=rtsp_password,
#          9=rtsp_puerto, 10=rtsp_canal, 11=rtsp_subtipo


def create_camara(
    nombre: str,
    direccion_ip: str,
    ubicacion: str,
    descripcion: str | None,
    activa: bool,
    rtsp_usuario: str = "admin",
    rtsp_password: str | None = None,
    rtsp_puerto: int = 554,
    rtsp_canal: int = 1,
    rtsp_subtipo: int = 1,
) -> tuple:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO camaras_ip
                    (nombre, direccion_ip, ubicacion, descripcion, activa,
                     rtsp_usuario, rtsp_password, rtsp_puerto, rtsp_canal, rtsp_subtipo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_COLS}
                """,
                (nombre, direccion_ip, ubicacion, descripcion, activa,
                 rtsp_usuario, rtsp_password, rtsp_puerto, rtsp_canal, rtsp_subtipo),
            )
            return cur.fetchone()


def list_camaras(solo_activas: bool = False) -> list[tuple]:
    with get_db() as conn:
        with conn.cursor() as cur:
            query = f"SELECT {_COLS} FROM camaras_ip"
            if solo_activas:
                query += " WHERE activa = TRUE"
            query += " ORDER BY id"
            cur.execute(query)
            return cur.fetchall()


def get_camara(camara_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM camaras_ip WHERE id = %s",
                (camara_id,),
            )
            return cur.fetchone()


def update_estado(camara_id: int, activa: bool) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE camaras_ip SET activa = %s
                WHERE id = %s
                RETURNING {_COLS}
                """,
                (activa, camara_id),
            )
            return cur.fetchone()


def delete_camara(camara_id: int) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM camaras_ip WHERE id = %s", (camara_id,))
