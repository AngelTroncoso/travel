"""Monitor en vivo de los streams de eventos (Redis Streams).

Herramienta de observabilidad para desarrollo: se conecta a Redis y usa
XREAD (bloqueante, desde '$' = solo eventos nuevos) para escuchar en
paralelo transporte.eventos, alojamiento.eventos y agenda.eventos,
imprimiendo cada evento recibido de forma legible y coloreada.

No usa consumer groups (no hace XREADGROUP/XACK): es un observador
pasivo, no compite por mensajes con los consumidores reales (agenda,
etc.) ni afecta su procesamiento.
"""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime

from redis.asyncio import Redis

REDIS_URL = "redis://localhost:6379/0"
STREAMS = ["transporte.eventos", "alojamiento.eventos", "agenda.eventos"]

# Colores ANSI. Windows Terminal / PowerShell 7 los soportan de forma
# nativa; en consolas legacy pueden no renderizar pero no rompen nada.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

STREAM_COLORS = {
    "transporte.eventos": "\033[94m",   # azul
    "alojamiento.eventos": "\033[92m",  # verde
    "agenda.eventos": "\033[95m",       # magenta
}
DEFAULT_COLOR = "\033[97m"  # blanco


def _color_for(stream: str) -> str:
    return STREAM_COLORS.get(stream, DEFAULT_COLOR)


def _print_evento(stream: str, message_id: str, fields: dict) -> None:
    color = _color_for(stream)
    now = datetime.now().strftime("%H:%M:%S")

    payload_raw = fields.get("payload")
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        payload = payload_raw

    event_type = fields.get("event_type", "?")
    tenant_id = fields.get("tenant_id", "?")

    print(
        f"{DIM}[{now}]{RESET} {color}{BOLD}{stream:<20}{RESET} "
        f"{color}id={message_id}{RESET} "
        f"{BOLD}event_type={event_type}{RESET} tenant_id={tenant_id}"
    )
    print(f"{color}{json.dumps(payload, indent=2, ensure_ascii=False)}{RESET}")
    print(f"{DIM}{'-' * 60}{RESET}")


async def monitor() -> None:
    redis = Redis.from_url(REDIS_URL, decode_responses=True)

    # '$' = solo mensajes que lleguen a partir de ahora, no el historico.
    last_ids: dict[str, str] = {stream: "$" for stream in STREAMS}

    print(f"{BOLD}Monitor de eventos Redis Streams{RESET}")
    print(f"Streams observados: {', '.join(STREAMS)}")
    print(f"Conectado a {REDIS_URL}. Esperando eventos nuevos...\n")

    try:
        while True:
            response = await redis.xread(streams=last_ids, count=10, block=5000)
            if not response:
                continue

            for stream_name, entries in response:
                for message_id, fields in entries:
                    _print_evento(stream_name, message_id, fields)
                    last_ids[stream_name] = message_id
    finally:
        await redis.close()


async def _main() -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)  # type: ignore[attr-defined]
        except (NotImplementedError, AttributeError):
            # Windows no soporta add_signal_handler; KeyboardInterrupt
            # se maneja igual via except mas abajo.
            pass

    monitor_task = asyncio.create_task(monitor())
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nMonitor detenido.")
