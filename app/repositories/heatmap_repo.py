import json

from app.database import get_db


def get_heatmap(zona_config_id: int) -> tuple | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT zona_config_id, grid, grid_ancho, grid_alto,
                       total_detecciones, actualizado_en
                FROM heatmap_zonas
                WHERE zona_config_id = %s
                """,
                (zona_config_id,),
            )
            return cur.fetchone()


def acumular_heatmap(
    zona_config_id: int,
    grid_delta: list[list[int]],
    grid_ancho: int,
    grid_alto: int,
) -> None:
    """Suma grid_delta al acumulado histórico de la zona (crea la fila si no existe)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT grid FROM heatmap_zonas WHERE zona_config_id = %s FOR UPDATE",
                (zona_config_id,),
            )
            row = cur.fetchone()

            if row is None:
                total = sum(sum(fila) for fila in grid_delta)
                cur.execute(
                    """
                    INSERT INTO heatmap_zonas
                        (zona_config_id, grid, grid_ancho, grid_alto, total_detecciones)
                    VALUES (%s, %s::jsonb, %s, %s, %s)
                    """,
                    (zona_config_id, json.dumps(grid_delta), grid_ancho, grid_alto, total),
                )
                return

            grid_actual = row[0]
            grid_nuevo = [
                [grid_actual[y][x] + grid_delta[y][x] for x in range(grid_ancho)]
                for y in range(grid_alto)
            ]
            total = sum(sum(fila) for fila in grid_nuevo)
            cur.execute(
                """
                UPDATE heatmap_zonas
                SET grid = %s::jsonb, total_detecciones = %s, actualizado_en = CURRENT_TIMESTAMP
                WHERE zona_config_id = %s
                """,
                (json.dumps(grid_nuevo), total, zona_config_id),
            )
