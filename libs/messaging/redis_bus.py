"""Capa de abstraccion de mensajeria sobre Redis Streams.

Sustituye a Azure Service Bus en el stack MVP de costo cero. Se usa Redis
Streams (en vez de Pub/Sub simple) porque persiste el log de mensajes y
soporta consumer groups con acknowledgment, lo que da una garantia
at-least-once similar a la de una cola durable.

Los tres agentes (Transporte, Alojamiento, Agenda) importan esta clase
para publicar eventos de dominio y para consumirlos via un consumer group
propio, sin acoplarse directamente al cliente de Redis.
"""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from libs.contracts.events import BaseEvent


class RedisStreamBus:
    """Bus de eventos simple sobre Redis Streams.

    Cada evento de dominio se publica en un stream (equivalente a un
    "topic") y cada agente consume desde ese stream usando su propio
    consumer group, lo que permite que multiples agentes lean el mismo
    stream de forma independiente y con su propio checkpoint de lectura.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._redis: Redis | None = None

    async def connect(self) -> None:
        """Inicializa la conexion async con Redis. Idempotente."""
        if self._redis is None:
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        """Cierra la conexion con Redis."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    @property
    def redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError(
                "RedisStreamBus no esta conectado. Llama a connect() primero."
            )
        return self._redis

    async def publish(self, stream_name: str, event: BaseEvent) -> str:
        """Publica un evento de dominio (modelo Pydantic) en un stream (XADD).

        El evento completo (incluyendo event_id, timestamp y tenant_id
        heredados de BaseEvent) se serializa a JSON y se guarda en el
        campo "payload". event_type y tenant_id tambien se replican como
        campos planos del mensaje para permitir filtrado rapido sin tener
        que deserializar el JSON completo.

        Args:
            stream_name: nombre del stream/canal (ej. "transporte.eventos").
            event: instancia de un modelo que hereda de BaseEvent.

        Returns:
            El id del mensaje asignado por Redis (ej. "1690000000000-0").
        """
        fields = {
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "payload": event.model_dump_json(),
        }
        message_id: str = await self.redis.xadd(stream_name, fields)
        return message_id

    async def ensure_consumer_group(
        self, stream: str, group: str
    ) -> None:
        """Crea el consumer group si no existe (XGROUP CREATE).

        Usa MKSTREAM para crear el stream automaticamente si aun no
        existe (ej. antes de que se publique el primer evento).
        """
        try:
            await self.redis.xgroup_create(
                name=stream, groupname=group, id="0", mkstream=True
            )
        except Exception as exc:  # noqa: BLE001
            # Redis lanza un error "BUSYGROUP" si el grupo ya existe.
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(
        self,
        stream_name: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 5000,
        auto_ack: bool = False,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Lee nuevos mensajes del stream para este consumer group (XREADGROUP).

        Crea el consumer group si aun no existe. Cada mensaje leido queda
        "pendiente" (PEL) hasta que se confirme con XACK; si auto_ack=True
        se confirma automaticamente tras la lectura (util para handlers
        que no necesitan reintentos manuales).

        Args:
            stream_name: nombre del stream a leer.
            group: nombre del consumer group (ej. "agenda-agent").
            consumer: identificador de esta instancia consumidora (util
                para escalar horizontalmente el mismo grupo).
            count: maximo de mensajes a leer por llamada.
            block_ms: milisegundos a esperar por nuevos mensajes antes
                de retornar vacio.
            auto_ack: si es True, confirma (XACK) cada mensaje leido de
                inmediato.

        Returns:
            Lista de tuplas (message_id, fields_dict) con
            event_type/tenant_id/payload ya deserializados.
        """
        await self.ensure_consumer_group(stream_name, group)

        response = await self.redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream_name: ">"},
            count=count,
            block=block_ms,
        )

        messages: list[tuple[str, dict[str, Any]]] = []
        if not response:
            return messages

        for _stream, entries in response:
            for message_id, fields in entries:
                parsed = dict(fields)
                if "payload" in parsed:
                    parsed["payload"] = json.loads(parsed["payload"])
                messages.append((message_id, parsed))
                if auto_ack:
                    await self.ack(stream_name, group, message_id)

        return messages

    async def ack(self, stream_name: str, group: str, message_id: str) -> None:
        """Confirma el procesamiento exitoso de un mensaje (XACK).

        Debe llamarse solo despues de que el mensaje fue procesado con
        exito; si el consumidor falla antes de hacer ack, el mensaje
        queda pendiente y puede ser reclamado/reprocesado.
        """
        await self.redis.xack(stream_name, group, message_id)
