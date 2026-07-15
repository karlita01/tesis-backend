FROM python:3.10-slim

WORKDIR /app

# Librerías de sistema necesarias en runtime:
# - libpq5: cliente de PostgreSQL (psycopg2-binary la necesita en el sistema)
# - libgomp1: runtime OpenMP (numpy/opencv/torch la usan por debajo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision: build CPU explícita, evita bajar CUDA (Railway no tiene GPU)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ultralytics arrastra opencv-python (no headless) por debajo. Desinstalar AMBOS
# y reinstalar limpio solo headless evita archivos .so mezclados de los dos paquetes
# conviviendo en la misma carpeta cv2/ (causa errores binarios raros en runtime).
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
# Blindaje final: garantiza esta numpy exacta pase lo que pase con los pasos anteriores
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Diagnóstico: deja constancia en el log del build de qué versiones quedaron
# realmente instaladas, para no seguir adivinando a ciegas si algo vuelve a fallar.
RUN python -c "import numpy, cv2, torch, pandas; print('numpy', numpy.__version__); print('opencv', cv2.__version__); print('torch', torch.__version__); print('pandas', pandas.__version__)"

# Descarga los pesos de YOLOv8 AHORA (en el build, con red garantizada) para que
# queden dentro de la imagen. El .pt está en .gitignore a propósito (no se sube a git),
# así que sin este paso el contenedor lo intentaría descargar en cada arranque.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
