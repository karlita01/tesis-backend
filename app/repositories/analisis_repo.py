from app.database import get_db

_COLS = (
    "r.id, r.sesion_id, r.zona_config_id, r.personas_maximas, r.nivel_maximo, "
    "r.tiempo_primera_media_seg, r.alerta_activada, r.frames_procesados, "
    "r.inicio_analisis, r.fin_analisis, r.fecha_registro, r.frame_evidencia, "
    "cze.nombre AS zona_nombre"
)

_COLS_INSERT = (
    "id, sesion_id, zona_config_id, personas_maximas, nivel_maximo, "
    "tiempo_primera_media_seg, alerta_activada, frames_procesados, "
    "inicio_analisis, fin_analisis, fecha_registro, frame_evidencia"
)

_JOIN = (
    "FROM resultados_analisis r "
    "LEFT JOIN configuraciones_zonas_exclusion cze ON cze.id = r.zona_config_id"
)


def save_resultado(
    sesion_id: int,
    zona_config_id: int | None,
    personas_maximas: int,
    nivel_maximo: str,
    tiempo_primera_media_seg: float | None,
    alerta_activada: bool,
    frames_procesados: int,
    inicio_analisis: str | None = None,
    fin_analisis: str | None = None,
    frame_evidencia: str | None = None,
) -> tuple:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO resultados_analisis
                    (sesion_id, zona_config_id, personas_maximas, nivel_maximo,
                     tiempo_primera_media_seg, alerta_activada, frames_procesados,
                     inicio_analisis, fin_analisis, frame_evidencia)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_COLS_INSERT}
                """,
                (sesion_id, zona_config_id, personas_maximas, nivel_maximo,
                 tiempo_primera_media_seg, alerta_activada, frames_procesados,
                 inicio_analisis, fin_analisis, frame_evidencia),
            )
            return cur.fetchone()


def get_resultado_by_sesion(sesion_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_COLS} {_JOIN}
                WHERE r.sesion_id = %s
                ORDER BY r.fecha_registro DESC
                LIMIT 1
                """,
                (sesion_id,),
            )
            return cur.fetchone()


def list_resultados(usuario_id: int | None = None) -> list[tuple]:
    """Admin ve todo; vigilante ve solo sus sesiones."""
    with get_db() as conn:
        with conn.cursor() as cur:
            if usuario_id is None:
                cur.execute(
                    f"""
                    SELECT {_COLS} {_JOIN}
                    ORDER BY r.fecha_registro DESC
                    LIMIT 100
                    """
                )
            else:
                cur.execute(
                    f"""
                    SELECT {_COLS} {_JOIN}
                    JOIN sesiones_monitoreo s ON s.id = r.sesion_id
                    WHERE s.usuario_id = %s
                    ORDER BY r.fecha_registro DESC
                    LIMIT 100
                    """,
                    (usuario_id,),
                )
            return cur.fetchall()


def get_zonas_criticas() -> list[tuple]:
    """
    RF-5.5: Agrega resultados por zona de exclusión.
    Retorna: zona_id, zona_nombre, total_sesiones, sesiones_con_alerta,
             max_personas, promedio_personas.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cze.id                                                          AS zona_id,
                    cze.nombre                                                      AS zona_nombre,
                    COUNT(r.id)                                                     AS total_sesiones,
                    COUNT(r.id) FILTER (WHERE r.alerta_activada = TRUE)             AS sesiones_con_alerta,
                    COALESCE(MAX(r.personas_maximas), 0)                            AS max_personas,
                    COALESCE(ROUND(AVG(r.personas_maximas))::int, 0)                AS promedio_personas
                FROM configuraciones_zonas_exclusion cze
                LEFT JOIN resultados_analisis r ON r.zona_config_id = cze.id
                WHERE cze.activa = TRUE
                GROUP BY cze.id, cze.nombre
                ORDER BY sesiones_con_alerta DESC, max_personas DESC
                """
            )
            return cur.fetchall()
