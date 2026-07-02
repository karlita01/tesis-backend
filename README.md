# Backend — CrowdSense AI

Sistema de detección y clasificación de aglomeraciones en video usando YOLOv8.
API REST construida con FastAPI + PostgreSQL (Supabase) + autenticación JWT.

---

## Stack

| Componente | Versión |
|---|---|
| Python | 3.10.4 |
| FastAPI | 0.115.0 |
| Base de datos | Supabase PostgreSQL (psycopg2-binary) |
| Autenticación | JWT Bearer (python-jose + passlib + bcrypt) |
| Detección | YOLOv8 (ultralytics 8.2.91) + OpenCV |

---

## Configuración inicial

### 1. Crear las tablas en Supabase

1. Abre tu proyecto en Supabase → **SQL Editor → New query**.
2. Pega el contenido de `scripts/create_tables.sql` y haz clic en **Run**.

### 2. Variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y rellena:

| Variable | Dónde encontrarla |
|---|---|
| `DATABASE_URL` | Supabase → Settings → Database → Connection string → URI |
| `JWT_SECRET` | Genera uno con `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Los que elijas para el primer administrador |

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear el usuario administrador

```bash
python scripts/seed_admin.py
```

### 5. Iniciar el servidor

```bash
python main.py
```

API disponible en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

---

## Endpoints principales

### Autenticación — `/auth`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/auth/registro` | Crea usuario con rol `vigilante` |
| `POST` | `/auth/login` | Autentica y devuelve JWT |
| `GET` | `/auth/me` | Datos del usuario autenticado |

### Cámaras IP — `/api/camaras` (admin)

| Método | Ruta |
|---|---|
| `POST` | `/api/camaras` |
| `GET` | `/api/camaras` |
| `PATCH` | `/api/camaras/{id}/estado` |
| `DELETE` | `/api/camaras/{id}` |

### Grabaciones — `/api/grabaciones`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/grabaciones` | Sube video (mp4/avi/mov/mkv) |
| `GET` | `/api/grabaciones` | Lista grabaciones |
| `GET` | `/api/grabaciones/{id}/file` | Descarga el archivo (acepta `?token=` para `<video>`) |

### Monitoreo — `/api/monitoreo`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/monitoreo/iniciar` | Inicia sesión de monitoreo |
| `POST` | `/api/monitoreo/{id}/detener` | Detiene sesión y guarda resultado |

### Zonas de exclusión — `/api/zonas-exclusion` (admin)

CRUD de configuraciones: polígonos normalizados (0–1), umbrales de detección (`umbral_medio`, `umbral_alto`), parámetros de ventana deslizante y cooldown.

### Análisis — `/api/analisis`

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/analisis/frame` | Analiza un frame de webcam (retorna detecciones + stats) |
| `GET` | `/api/analisis/video/{id}/stream` | SSE de análisis de grabación previa |
| `GET` | `/api/analisis/historial` | Historial de sesiones (admin: todo; vigilante: propias) |
| `GET` | `/api/analisis/resultado/{id}` | Resultado de una sesión específica |
| `GET` | `/api/analisis/zonas-criticas` | Resumen agregado por zona de exclusión |

### Alertas — `/api/alertas`

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/alertas/stream` | SSE de alertas en tiempo real |
| `GET` | `/api/alertas` | Historial de alertas |
| `PATCH` | `/api/alertas/{id}/atender` | Marca alerta como atendida |

---

## Roles

| Rol | Cómo se crea | Acceso |
|---|---|---|
| `vigilante` | Registro público | Endpoints autenticados |
| `administrador` | `scripts/seed_admin.py` | Todos los endpoints, incluidos los de admin |

> El registro público nunca permite seleccionar rol. El rol `administrador` solo se crea desde el backend.

---

## Notas

- `uploads/grabaciones/`, `uploads/frames/` y `uploads/evidencias/` se crean automáticamente al iniciar. No se suben al repositorio.
- Los modelos `.pt` (YOLOv8) tampoco se suben al repositorio.
- `.env` nunca debe comitearse.
