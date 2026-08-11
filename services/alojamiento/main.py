"""Punto de entrada del Agente de Alojamiento.

Nucleo determinista: expone un endpoint sincrono de consulta de
disponibilidad (usado por el Agente de Agenda via REST) y un endpoint
para crear reservas de hotel, publicando el evento de dominio
correspondiente (ReservaHotelConfirmada) en Redis Streams.

Almacenamiento: diccionario en memoria (se migra a SQLite en una task
posterior). El bus de mensajeria se inyecta como dependencia de FastAPI,
igual que en el Agente de Transporte.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from libs.contracts.api import DisponibilidadResponse
from libs.contracts.events import ReservaHotelConfirmada
from libs.messaging.redis_bus import RedisStreamBus

SERVICE_NAME = "alojamiento"
EVENTS_STREAM = "alojamiento.eventos"

# Estado en memoria. Se reemplaza por SQLite en una task posterior.
# hoteles_db: hotel_id -> habitaciones totales disponibles para asignar.
hoteles_db: dict[str, dict] = {
    "HTL-001": {"destino": "Bogota", "habitaciones_totales": 10},
    "HTL-002": {"destino": "Medellin", "habitaciones_totales": 5},
}
# reservas_db: reserva_id -> detalle de la reserva confirmada.
reservas_db: dict[str, dict] = {}

bus = RedisStreamBus()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await bus.connect()
    try:
        yield
    finally:
        await bus.close()


app = FastAPI(title="Agente de Alojamiento", lifespan=lifespan)


def get_bus() -> RedisStreamBus:
    """Dependencia de FastAPI que provee la instancia compartida del bus."""
    return bus


def _habitaciones_reservadas(hotel_id: str, check_in: date, check_out: date) -> int:
    """Suma las habitaciones ya reservadas para un hotel que se solapan
    con el rango [check_in, check_out) solicitado."""
    total = 0
    for reserva in reservas_db.values():
        if reserva["hotel_id"] != hotel_id:
            continue
        solapa = reserva["check_in"] < check_out and check_in < reserva["check_out"]
        if solapa:
            total += reserva["habitaciones"]
    return total


# --- Esquemas de request --------------------------------------------------


class CrearReservaRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    hotel_id: str = Field(..., min_length=1)
    check_in: date
    check_out: date
    habitaciones: int = Field(1, ge=1)


# --- Endpoints -------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check basico del servicio."""
    return {"status": "ok", "service": SERVICE_NAME}


@app.get(
    "/api/v1/alojamiento/disponibilidad",
    response_model=DisponibilidadResponse,
)
async def consultar_disponibilidad(
    hotel_id: str = Query(..., min_length=1),
    check_in: date = Query(...),
    check_out: date = Query(...),
    habitaciones_requeridas: int = Query(1, ge=1),
) -> DisponibilidadResponse:
    """Consulta sincrona de disponibilidad para un hotel y rango de fechas.

    Usado por el Agente de Agenda via REST antes de confirmar/rearmar un
    itinerario (ej. tras un VueloRetrasado que exige revalidar el hotel).
    """
    hotel = hoteles_db.get(hotel_id)
    if hotel is None:
        raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} no encontrado.")

    if check_out <= check_in:
        raise HTTPException(
            status_code=422, detail="check_out debe ser posterior a check_in."
        )

    reservadas = _habitaciones_reservadas(hotel_id, check_in, check_out)
    disponibles = max(hotel["habitaciones_totales"] - reservadas, 0)
    disponible = disponibles >= habitaciones_requeridas

    return DisponibilidadResponse(
        hotel_id=hotel_id,
        check_in=check_in,
        check_out=check_out,
        disponible=disponible,
        habitaciones_disponibles=disponibles,
        motivo_no_disponible=None if disponible else "Sin habitaciones suficientes",
    )


@app.post("/api/v1/alojamiento/reservas", status_code=201)
async def crear_reserva(
    request: CrearReservaRequest,
    stream_bus: RedisStreamBus = Depends(get_bus),
) -> dict:
    """Crea una reserva de hotel y publica el evento ReservaHotelConfirmada."""
    hotel = hoteles_db.get(request.hotel_id)
    if hotel is None:
        raise HTTPException(
            status_code=404, detail=f"Hotel {request.hotel_id} no encontrado."
        )

    if request.check_out <= request.check_in:
        raise HTTPException(
            status_code=422, detail="check_out debe ser posterior a check_in."
        )

    reservadas = _habitaciones_reservadas(
        request.hotel_id, request.check_in, request.check_out
    )
    disponibles = hotel["habitaciones_totales"] - reservadas
    if disponibles < request.habitaciones:
        raise HTTPException(
            status_code=409,
            detail=f"Sin disponibilidad suficiente en {request.hotel_id}.",
        )

    reserva_id = str(uuid4())
    registro = {
        "reserva_id": reserva_id,
        "tenant_id": request.tenant_id,
        "hotel_id": request.hotel_id,
        "check_in": request.check_in,
        "check_out": request.check_out,
        "habitaciones": request.habitaciones,
    }
    reservas_db[reserva_id] = registro

    evento = ReservaHotelConfirmada(
        tenant_id=request.tenant_id,
        reserva_id=reserva_id,
        hotel_id=request.hotel_id,
        check_in=request.check_in,
        check_out=request.check_out,
    )
    await stream_bus.publish(EVENTS_STREAM, evento)

    return registro
