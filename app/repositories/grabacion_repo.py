from app.database import get_db

_COLS = (
    "id, nombre_archivo, ruta_archivo, tipo_contenido, tamanio_bytes, "
    "usuario_id, fecha_carga, fecha_grabacion"
)


def create_grabacion(
    nombre_archivo: str,
    ruta_archivo: str,
    tipo_contenido: str | None,
    tamanio_bytes: int | None,
    usuario_id: int,
    fecha_grabacion: str | None = None,
) -> tuple:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO grabaciones
                    (nombre_archivo, ruta_archivo, tipo_contenido,
                     tamanio_bytes, usuario_id, fecha_grabacion)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING {_COLS}
                """,
                (nombre_archivo, ruta_archivo, tipo_contenido,
                 tamanio_bytes, usuario_id, fecha_grabacion),
            )
            return cur.fetchone()


def list_grabaciones(usuario_id: int | None = None) -> list[tuple]:
    with get_db() as conn:
        with conn.cursor() as cur:
            if usuario_id is None:
                cur.execute(f"SELECT {_COLS} FROM grabaciones ORDER BY fecha_carga DESC")
            else:
                cur.execute(
                    f"SELECT {_COLS} FROM grabaciones WHERE usuario_id = %s ORDER BY fecha_carga DESC",
                    (usuario_id,),
                )
            return cur.fetchall()


def get_grabacion(grabacion_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM grabaciones WHERE id = %s", (grabacion_id,))
            return cur.fetchone()


def delete_grabacion(grabacion_id: int) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM grabaciones WHERE id = %s", (grabacion_id,))
