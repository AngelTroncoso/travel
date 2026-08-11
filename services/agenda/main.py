"""Punto de entrada del Agente de Agenda.

El Agente de Agenda es el orquestador central de la saga logistica:
consume eventos de Transporte y Alojamiento via Redis Streams, realiza
llamadas sincronas (REST) a esos agentes para consultas en tiempo real,
y coordina las compensaciones (rollback/escalacion) cuando hay conflictos.

Task 7/8: circuito end-to-end minimo (consumer loop + llamada REST).
Task 10: capa de razonamiento con Semantic Kernel.
- Al recibir un VueloRetrasado, se consulta disponibilidad de hotel
  (REST) y ambos datos (retraso + disponibilidad) se inyectan como
  contexto en un Kernel de Semantic Kernel con un prompt de sistema que
  le da personalidad de orquestador de viajes corporativos.
- El LLM decide REPROGRAMAR o ESCALAR. La decision se loguea y se
  publica de vuelta a Redis como ItinerarioActualizado (REPROGRAMAR) o
  ConflictoRequiereIntervencion (ESCALAR).
- Para mantener el costo en cero: si no hay OPENAI_API_KEY configurada,
  se usa un razonador mock deterministico (sin llamadas de red) que
  sigue el mismo prompt/logica en pseudocodigo, permitiendo demoear el
  flujo completo sin credenciales. Con una API key valida (OpenAI o
  cualquier endpoint OpenAI-compatible via OPENAI_BASE_URL) se activa
  el razonamiento real via Semantic Kernel.

La maquina de estados de la saga con compensaciones explicitas (Task 9)
se profundiza en una task posterior; aqui el razonamiento decide, pero
la ejecucion determinista de la compensacion queda fuera de alcance.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from libs.contracts.events import ConflictoRequiereIntervencion, ItinerarioActualizado
from libs.messaging.redis_bus import RedisStreamBus

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("agenda")

SERVICE_NAME = "agenda"
TRANSPORTE_EVENTS_STREAM = "transporte.eventos"
AGENDA_EVENTS_STREAM = "agenda.eventos"
CONSUMER_GROUP = "agenda-agent"
CONSUMER_NAME = "agenda-agent-1"

ALOJAMIENTO_BASE_URL = "http://127.0.0.1:8002"
ALOJAMIENTO_DISPONIBILIDAD_PATH = "/api/v1/alojamiento/disponibilidad"

# Hotel asociado al itinerario, hardcodeado por ahora: la maquina de
# estados de la saga (Task 9) reemplazara esto por la busqueda real del
# hotel vinculado al itinerario/PNR afectado.
DEFAULT_HOTEL_ID = "HTL-001"

SYSTEM_PROMPT = (
    "Eres un orquestador de viajes corporativos. Recibes un retraso de "
    "vuelo y la disponibilidad de hotel. Debes decidir si REPROGRAMAR o "
    "ESCALAR el conflicto. Responde SIEMPRE empezando con la palabra "
    "REPROGRAMAR o ESCALAR en mayusculas, seguida de dos puntos y una "
    "justificacion breve (1-2 frases)."
)

bus = RedisStreamBus()
http_client: httpx.AsyncClient | None = None
_consumer_task: asyncio.Task | None = None
_shutdown_event = asyncio.Event()


# --- Capa de razonamiento (Semantic Kernel) ---------------------------------


class ConflictReasoner:
    """Interfaz minima de un razonador: decide REPROGRAMAR o ESCALAR."""

    async def decidir(self, retraso_info: dict, disponibilidad_info: dict) -> str:
        raise NotImplementedError


class MockConflictReasoner(ConflictReasoner):
    """Razonador determinista sin costo, usado cuando no hay API key.

    Replica la misma logica que se le pediria al LLM (via el prompt de
    sistema), pero de forma local y gratuita: si el hotel esta
    disponible, reprograma; si no, escala. Permite demoear el circuito
    completo end-to-end sin credenciales de un proveedor de IA.
    """

    async def decidir(self, retraso_info: dict, disponibilidad_info: dict) -> str:
        disponible = bool(disponibilidad_info.get("disponible"))
        pnr = retraso_info.get("pnr", "desconocido")
        minutos = retraso_info.get("minutos_retraso", "?")
        if disponible:
            return (
                f"REPROGRAMAR: el vuelo {pnr} tiene un retraso de {minutos} "
                "minutos, pero el hotel sigue disponible para la nueva fecha; "
                "se ajusta el itinerario sin intervencion humana."
            )
        motivo = disponibilidad_info.get("motivo_no_disponible", "sin disponibilidad")
        return (
            f"ESCALAR: el vuelo {pnr} tiene un retraso de {minutos} minutos y "
            f"el hotel no tiene disponibilidad ({motivo}); se requiere "
            "intervencion humana para reasignar alojamiento."
        )


class SemanticKernelConflictReasoner(ConflictReasoner):
    """Razonador real, respaldado por un Kernel de Semantic Kernel.

    Usa un conector de chat OpenAI-compatible (OpenAI, Azure AI, o
    cualquier servidor local/gratuito que expone la misma API, ej. LM
    Studio u Ollama con proxy OpenAI) configurado via variables de
    entorno, para mantener el costo de infraestructura en cero.
    """

    def __init__(self, api_key: str, base_url: str, model_id: str) -> None:
        self._service_id = "agenda-reasoner"
        self._kernel = Kernel()

        openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._kernel.add_service(
            OpenAIChatCompletion(
                service_id=self._service_id,
                ai_model_id=model_id,
                async_client=openai_client,
            )
        )

        settings = self._kernel.get_prompt_execution_settings_from_service_id(
            self._service_id
        )
        settings.max_tokens = 300
        settings.temperature = 0.2

        self._chat_function = self._kernel.add_function(
            plugin_name="AgendaOrchestrator",
            function_name="EvaluarConflicto",
            prompt="{{$chat_history}}{{$user_input}}",
            template_format="semantic-kernel",
            prompt_execution_settings=settings,
        )

    async def decidir(self, retraso_info: dict, disponibilidad_info: dict) -> str:
        user_input = (
            "Datos del retraso de vuelo (JSON):\n"
            f"{retraso_info}\n\n"
            "Datos de disponibilidad de hotel (JSON):\n"
            f"{disponibilidad_info}\n\n"
            "Decide REPROGRAMAR o ESCALAR."
        )
        chat_history = ChatHistory(system_message=SYSTEM_PROMPT)

        result = await self._kernel.invoke(
            self._chat_function,
            KernelArguments(user_input=user_input, chat_history=chat_history),
        )
        return str(result)


def _build_reasoner() -> ConflictReasoner:
    """Selecciona el razonador segun la configuracion disponible.

    Si OPENAI_API_KEY no esta definida, se usa el razonador mock (costo
    cero, sin red). Si esta definida, se activa Semantic Kernel contra
    el endpoint OpenAI-compatible configurado en OPENAI_BASE_URL.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "OPENAI_API_KEY no configurada: usando MockConflictReasoner (costo cero)."
        )
        return MockConflictReasoner()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_id = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")
    logger.info(
        "OPENAI_API_KEY detectada: usando SemanticKernelConflictReasoner "
        "(base_url=%s, model_id=%s).",
        base_url,
        model_id,
    )
    return SemanticKernelConflictReasoner(
        api_key=api_key, base_url=base_url, model_id=model_id
    )


reasoner: ConflictReasoner = _build_reasoner()


# --- Llamadas sincronas a otros agentes -------------------------------------


async def consultar_disponibilidad_alojamiento(
    hotel_id: str, check_in: date, check_out: date
) -> dict:
    """Llamada sincrona (REST) al Agente de Alojamiento.

    Se usa cuando la Agenda necesita revalidar disponibilidad de hotel en
    reaccion a un evento del Agente de Transporte (ej. VueloRetrasado).
    """
    assert http_client is not None, "http_client no inicializado"
    response = await http_client.get(
        ALOJAMIENTO_DISPONIBILIDAD_PATH,
        params={
            "hotel_id": hotel_id,
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
        },
    )
    response.raise_for_status()
    return response.json()


# --- Reaccion a eventos ------------------------------------------------------


async def _publicar_decision(
    tenant_id: str, pnr: str, decision_text: str
) -> None:
    """Publica el resultado del razonamiento como evento de dominio.

    REPROGRAMAR -> ItinerarioActualizado (la agenda se ajusta sola).
    ESCALAR -> ConflictoRequiereIntervencion (requiere intervencion humana).
    """
    if decision_text.strip().upper().startswith("REPROGRAMAR"):
        evento = ItinerarioActualizado(
            tenant_id=tenant_id,
            itinerary_id=pnr,
            version=1,
            resumen_cambio=decision_text,
        )
    else:
        evento = ConflictoRequiereIntervencion(
            tenant_id=tenant_id,
            itinerary_id=pnr,
            motivo="Conflicto de reprogramacion tras VueloRetrasado",
            detalle=decision_text,
            requiere_humano=True,
        )

    await bus.publish(AGENDA_EVENTS_STREAM, evento)
    logger.info(
        "Evento '%s' publicado en '%s' para itinerary_id=%s",
        evento.event_type,
        AGENDA_EVENTS_STREAM,
        pnr,
    )


async def _handle_vuelo_retrasado(payload: dict) -> None:
    """Reacciona a un evento VueloRetrasado: revalida disponibilidad de
    hotel, inyecta ambos datos en el razonador (Semantic Kernel o mock),
    y publica la decision resultante."""
    tenant_id = payload.get("tenant_id", "desconocido")
    pnr = payload.get("pnr")
    nueva_hora_salida = payload.get("nueva_hora_salida_utc")
    minutos_retraso = payload.get("minutos_retraso")

    logger.info(
        "VueloRetrasado recibido: pnr=%s minutos_retraso=%s nueva_hora_salida_utc=%s",
        pnr,
        minutos_retraso,
        nueva_hora_salida,
    )

    nueva_fecha = date.fromisoformat(nueva_hora_salida[:10])
    check_in = nueva_fecha
    check_out = nueva_fecha + timedelta(days=1)

    try:
        disponibilidad = await consultar_disponibilidad_alojamiento(
            hotel_id=DEFAULT_HOTEL_ID, check_in=check_in, check_out=check_out
        )
        logger.info(
            "Disponibilidad revalidada para pnr=%s hotel_id=%s -> %s",
            pnr,
            DEFAULT_HOTEL_ID,
            disponibilidad,
        )
    except httpx.HTTPError as exc:
        logger.error(
            "Error consultando disponibilidad para pnr=%s hotel_id=%s: %s",
            pnr,
            DEFAULT_HOTEL_ID,
            exc,
        )
        return

    decision_text = await reasoner.decidir(
        retraso_info=payload, disponibilidad_info=disponibilidad
    )
    logger.info("Decision del razonador para pnr=%s: %s", pnr, decision_text)

    await _publicar_decision(tenant_id=tenant_id, pnr=pnr, decision_text=decision_text)


async def _dispatch_event(event_type: str, payload: dict) -> None:
    if event_type == "VueloRetrasado":
        await _handle_vuelo_retrasado(payload)
    else:
        logger.info("Evento ignorado (sin handler): %s", event_type)


async def _consume_transporte_events_loop() -> None:
    """Loop permanente que consume 'transporte.eventos' via consumer group.

    Corre como background task durante todo el ciclo de vida de la app
    (arranca en lifespan). Hace ack de cada mensaje tras procesarlo.
    """
    logger.info(
        "Iniciando consumo de '%s' (group=%s, consumer=%s)",
        TRANSPORTE_EVENTS_STREAM,
        CONSUMER_GROUP,
        CONSUMER_NAME,
    )
    while not _shutdown_event.is_set():
        try:
            messages = await bus.consume(
                stream_name=TRANSPORTE_EVENTS_STREAM,
                group=CONSUMER_GROUP,
                consumer=CONSUMER_NAME,
                count=10,
                block_ms=5000,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Error leyendo eventos de '%s'", TRANSPORTE_EVENTS_STREAM)
            await asyncio.sleep(2)
            continue

        for message_id, fields in messages:
            event_type = fields.get("event_type", "")
            payload = fields.get("payload", {})
            try:
                await _dispatch_event(event_type, payload)
            except Exception:  # noqa: BLE001
                logger.exception("Error procesando evento %s (%s)", event_type, message_id)
            finally:
                await bus.ack(TRANSPORTE_EVENTS_STREAM, CONSUMER_GROUP, message_id)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global http_client, _consumer_task

    await bus.connect()
    http_client = httpx.AsyncClient(base_url=ALOJAMIENTO_BASE_URL, timeout=10.0)
    _shutdown_event.clear()
    _consumer_task = asyncio.create_task(_consume_transporte_events_loop())

    try:
        yield
    finally:
        _shutdown_event.set()
        if _consumer_task is not None:
            _consumer_task.cancel()
            try:
                await _consumer_task
            except asyncio.CancelledError:
                pass
        if http_client is not None:
            await http_client.aclose()
        await bus.close()


app = FastAPI(title="Agente de Agenda", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check basico del servicio."""
    return {"status": "ok", "service": SERVICE_NAME}
