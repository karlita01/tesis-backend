"""
SSE Manager para alertas en tiempo real (EP-004, RF-4.1).

Singleton que mantiene una cola asyncio por cada cliente conectado,
agrupadas por usuario_id (un usuario puede tener varias pestañas abiertas).

Thread-safety:
- subscribe/unsubscribe usan threading.Lock (pueden llamarse desde cualquier contexto).
- publish es una coroutine que corre en el event loop; usa put_nowait() que es
  seguro dentro del loop.
- publish_from_thread usa run_coroutine_threadsafe para cruzar desde el hilo
  de procesamiento de video al event loop de uvicorn.
"""

import asyncio
from collections import defaultdict
from threading import Lock


class _AlertaSSEManager:
    def __init__(self) -> None:
        self._queues: dict[int, list[asyncio.Queue]] = defaultdict(list)
        self._lock = Lock()

    def subscribe(self, usuario_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._lock:
            self._queues[usuario_id].append(q)
        return q

    def unsubscribe(self, usuario_id: int, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._queues[usuario_id].remove(q)
            except ValueError:
                pass

    async def publish(self, usuario_id: int, data: dict) -> None:
        """Publica un evento a todos los clientes SSE conectados del usuario."""
        with self._lock:
            queues = list(self._queues.get(usuario_id, []))
        for q in queues:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass  # cliente lento: se descarta el evento

    def publish_from_thread(
        self,
        usuario_id: int,
        data: dict,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Publica desde un hilo (procesamiento de video). Bloquea hasta completar."""
        asyncio.run_coroutine_threadsafe(
            self.publish(usuario_id, data), loop
        ).result(timeout=5)


alerta_manager = _AlertaSSEManager()
