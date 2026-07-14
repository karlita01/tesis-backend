-- =============================================================
-- Esquema para el backend de detección de aglomeraciones
-- Compatible con Supabase PostgreSQL
-- Ejecutar en: Supabase → SQL Editor → New query
-- =============================================================

-- ── Tabla de usuarios ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS usuarios (
    id              SERIAL PRIMARY KEY,
    nombre          VARCHAR(100)  NOT NULL,
    email           VARCHAR(255)  NOT NULL UNIQUE,
    password_hash   TEXT          NOT NULL,
    rol             VARCHAR(20)   NOT NULL DEFAULT 'vigilante'
                        CHECK (rol IN ('vigilante', 'administrador')),
    activo          BOOLEAN       NOT NULL DEFAULT TRUE,
    fecha_creacion  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_email ON usuarios (email);

-- =============================================================
-- RF-1.1 a RF-1.5 — Cámaras IP, grabaciones y monitoreo
-- =============================================================

-- ── Tabla de cámaras IP ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS camaras_ip (
    id              SERIAL PRIMARY KEY,
    nombre          VARCHAR(100)  NOT NULL,
    direccion_ip    VARCHAR(255)  NOT NULL,
    ubicacion       VARCHAR(255)  NOT NULL,
    descripcion     TEXT,
    activa          BOOLEAN       NOT NULL DEFAULT TRUE,
    fecha_registro  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ── Tabla de grabaciones previas ──────────────────────────────
CREATE TABLE IF NOT EXISTS grabaciones (
    id              SERIAL PRIMARY KEY,
    nombre_archivo  VARCHAR(255)  NOT NULL,
    ruta_archivo    TEXT          NOT NULL,
    tipo_contenido  VARCHAR(100),
    tamanio_bytes   BIGINT,
    usuario_id      INTEGER       REFERENCES usuarios(id) ON DELETE SET NULL,
    fecha_carga     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    fecha_grabacion TIMESTAMP WITH TIME ZONE   -- hora real en que fue filmada (opcional)
);

-- ── Tabla de sesiones de monitoreo ────────────────────────────
CREATE TABLE IF NOT EXISTS sesiones_monitoreo (
    id                  SERIAL PRIMARY KEY,
    usuario_id          INTEGER       REFERENCES usuarios(id) ON DELETE SET NULL,
    tipo_fuente         VARCHAR(20)   NOT NULL
                            CHECK (tipo_fuente IN ('webcam', 'grabacion_previa', 'camara_ip')),
    camara_id           INTEGER       REFERENCES camaras_ip(id) ON DELETE SET NULL,
    grabacion_id        INTEGER       REFERENCES grabaciones(id) ON DELETE SET NULL,
    zona_exclusion_id   INTEGER,      -- FK se añade después de crear la tabla de zonas
    estado              VARCHAR(20)   NOT NULL DEFAULT 'activo'
                            CHECK (estado IN ('activo', 'detenido')),
    fecha_inicio        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    fecha_fin           TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_sesiones_usuario ON sesiones_monitoreo (usuario_id);
CREATE INDEX IF NOT EXISTS idx_sesiones_estado  ON sesiones_monitoreo (estado);

-- =============================================================
-- RF-2.1 a RF-2.5 — Configuraciones de zonas de exclusión
-- =============================================================

CREATE TABLE IF NOT EXISTS configuraciones_zonas_exclusion (
    id                   SERIAL PRIMARY KEY,
    nombre               VARCHAR(120)  NOT NULL,
    frame_referencia     VARCHAR(255)  NOT NULL,
    zonas                JSONB         NOT NULL DEFAULT '[]',
    umbral_medio         INTEGER       NOT NULL DEFAULT 4,   -- personas para nivel Medio
    umbral_alto          INTEGER       NOT NULL DEFAULT 6,   -- personas para nivel Alto
    ventana_segundos     FLOAT         NOT NULL DEFAULT 2.0, -- ventana deslizante alerta
    cooldown_segundos    INTEGER       NOT NULL DEFAULT 10,  -- pausa entre alertas
    creado_por           INTEGER       REFERENCES usuarios(id) ON DELETE SET NULL,
    activa               BOOLEAN       NOT NULL DEFAULT TRUE,
    fecha_creacion       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_zonas_exclusion_activa
    ON configuraciones_zonas_exclusion (activa);
CREATE INDEX IF NOT EXISTS idx_zonas_exclusion_creado_por
    ON configuraciones_zonas_exclusion (creado_por);

-- FK diferida: sesiones_monitoreo → configuraciones_zonas_exclusion
ALTER TABLE sesiones_monitoreo
    ADD CONSTRAINT IF NOT EXISTS fk_sesion_zona
    FOREIGN KEY (zona_exclusion_id)
    REFERENCES configuraciones_zonas_exclusion(id)
    ON DELETE SET NULL;

-- =============================================================
-- EP-003 — Resultados de análisis de aglomeraciones
-- =============================================================

CREATE TABLE IF NOT EXISTS resultados_analisis (
    id                          SERIAL PRIMARY KEY,
    sesion_id                   INTEGER       REFERENCES sesiones_monitoreo(id) ON DELETE CASCADE,
    zona_config_id              INTEGER       REFERENCES configuraciones_zonas_exclusion(id) ON DELETE SET NULL,
    personas_maximas            INTEGER       NOT NULL DEFAULT 0,
    nivel_maximo                VARCHAR(20)   NOT NULL DEFAULT 'sin_aglomeracion'
                                    CHECK (nivel_maximo IN ('sin_aglomeracion','bajo','medio','alto')),
    tiempo_primera_media_seg    FLOAT,        -- RF-3.5: segundos hasta primer nivel medio+
    alerta_activada             BOOLEAN       NOT NULL DEFAULT FALSE,
    frames_procesados           INTEGER       NOT NULL DEFAULT 0,
    inicio_analisis             TIMESTAMP WITH TIME ZONE,
    fin_analisis                TIMESTAMP WITH TIME ZONE,
    fecha_registro              TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_resultados_sesion ON resultados_analisis (sesion_id);

-- =============================================================
-- EP-004 — Alertas de aglomeración
-- =============================================================

CREATE TABLE IF NOT EXISTS alertas (
    id              SERIAL PRIMARY KEY,
    sesion_id       INTEGER      REFERENCES sesiones_monitoreo(id) ON DELETE CASCADE,
    usuario_id      INTEGER      REFERENCES usuarios(id) ON DELETE SET NULL,
    zona_config_id  INTEGER      REFERENCES configuraciones_zonas_exclusion(id) ON DELETE SET NULL,
    nivel           VARCHAR(20)  NOT NULL DEFAULT 'alto'
                        CHECK (nivel IN ('bajo', 'medio', 'alto')),
    personas        INTEGER      NOT NULL DEFAULT 0,
    atendida        BOOLEAN      NOT NULL DEFAULT FALSE,
    fecha_alerta    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    fecha_atencion  TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_alertas_sesion   ON alertas (sesion_id);
CREATE INDEX IF NOT EXISTS idx_alertas_usuario  ON alertas (usuario_id);
CREATE INDEX IF NOT EXISTS idx_alertas_atendida ON alertas (atendida);

-- =============================================================
-- MIGRACIONES — ejecutar solo si la BD ya existe con el esquema anterior
-- =============================================================

-- Agregar columnas nuevas a configuraciones_zonas_exclusion (ignorar si ya existen)
ALTER TABLE configuraciones_zonas_exclusion ADD COLUMN IF NOT EXISTS umbral_medio      INTEGER NOT NULL DEFAULT 4;
ALTER TABLE configuraciones_zonas_exclusion ADD COLUMN IF NOT EXISTS umbral_alto       INTEGER NOT NULL DEFAULT 6;
ALTER TABLE configuraciones_zonas_exclusion ADD COLUMN IF NOT EXISTS ventana_segundos  FLOAT   NOT NULL DEFAULT 2.0;
ALTER TABLE configuraciones_zonas_exclusion ADD COLUMN IF NOT EXISTS cooldown_segundos INTEGER NOT NULL DEFAULT 10;

-- Agregar fecha_grabacion a grabaciones
ALTER TABLE grabaciones ADD COLUMN IF NOT EXISTS fecha_grabacion TIMESTAMP WITH TIME ZONE;

-- Agregar zona_exclusion_id a sesiones_monitoreo
ALTER TABLE sesiones_monitoreo ADD COLUMN IF NOT EXISTS zona_exclusion_id INTEGER;

-- EP-005: frame de evidencia (captura del momento de mayor concentración)
ALTER TABLE resultados_analisis ADD COLUMN IF NOT EXISTS frame_evidencia TEXT;

-- Cámara IP: credenciales RTSP
ALTER TABLE camaras_ip ADD COLUMN IF NOT EXISTS rtsp_usuario  VARCHAR(100) DEFAULT 'admin';
ALTER TABLE camaras_ip ADD COLUMN IF NOT EXISTS rtsp_password VARCHAR(100);
ALTER TABLE camaras_ip ADD COLUMN IF NOT EXISTS rtsp_puerto   INTEGER DEFAULT 554;
ALTER TABLE camaras_ip ADD COLUMN IF NOT EXISTS rtsp_canal    INTEGER DEFAULT 1;
ALTER TABLE camaras_ip ADD COLUMN IF NOT EXISTS rtsp_subtipo  INTEGER DEFAULT 1;
