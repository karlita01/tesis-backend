from app.database import get_db

_COLS = (
    "id, sesion_id, usuario_id, zona_config_id, nivel, personas, "
    "atendida, fecha_alerta, fecha_atencion"
)


def crear_alerta(
    sesion_id: int,
    usuario_id: int,
    zona_config_id: int | None,
    nivel: str,
    personas: int,
) -> tuple:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO alertas
                    (sesion_id, usuario_id, zona_config_id, nivel, personas)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING {_COLS}
                """,
                (sesion_id, usuario_id, zona_config_id, nivel, personas),
            )
            return cur.fetchone()


def get_alerta(alerta_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLS} FROM alertas WHERE id = %s",
                (alerta_id,),
            )
            return cur.fetchone()


def marcar_atendida(alerta_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE alertas
                SET atendida = TRUE,
                    fecha_atencion = NOW()
                WHERE id = %s
                RETURNING {_COLS}
                """,
                (alerta_id,),
            )
            return cur.fetchone()


def list_alertas(
    usuario_id: int | None = None,
    atendida: bool | None = None,
    limit: int = 100,
) -> list[tuple]:
    """Admin ve todas; vigilante solo las propias. Filtrables por estado atendida."""
    conditions: list[str] = []
    params: list = []

    if usuario_id is not None:
        conditions.append("usuario_id = %s")
        params.append(usuario_id)
    if atendida is not None:
        conditions.append("atendida = %s")
        params.append(atendida)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_COLS} FROM alertas
                {where}
                ORDER BY fecha_alerta DESC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()
