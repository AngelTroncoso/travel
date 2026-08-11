"""Punto de entrada del Agente de Transporte.

Nucleo determinista: expone endpoints REST para crear reservas de vuelo
y registrar retrasos, publicando los eventos de dominio correspondientes
(VueloConfirmado / VueloRetrasado) en Redis Streams via RedisStreamBus.

Almacenamiento: por ahora un diccionario en memoria (se migra a SQLite
en una task posterior). El bus de mensajeria se inyecta como dependencia
de FastAPI y se conecta/cierra segun el ciclo de vida de la app.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from libs.contracts.events import VueloConfirmado, VueloRetrasado
from libs.messaging.redis_bus import RedisStreamBus

SERVICE_NAME = "transporte"
EVENTS_STREAM = "transporte.eventos"

# Almacenamiento en memoria, indexado por PNR. Se reemplaza por SQLite
# en una task posterior del plan.
_vuelos_db: dict[str, dict] = {}

bus = RedisStreamBus()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await bus.connect()
    try:
        yield
    finally:
        await bus.close()


app = FastAPI(title="Agente de Transporte", lifespan=lifespan)


def get_bus() -> RedisStreamBus:
    """Dependencia de FastAPI que provee la instancia compartida del bus."""
    return bus


# --- Esquemas de request -----------------------------------------------


class CrearVueloRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    pnr: str = Field(..., min_length=1)
    vuelo_id: str = Field(..., min_length=1)
    hora_salida_utc: datetime
    hora_llegada_utc: datetime


class RegistrarRetrasoRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    minutos_retraso: int = Field(..., ge=0)
    nueva_hora_salida_utc: datetime


# --- Endpoints -----------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check basico del servicio."""
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/api/v1/vuelos", status_code=201)
async def crear_vuelo(
    request: CrearVueloRequest,
    stream_bus: RedisStreamBus = Depends(get_bus),
) -> dict:
    """Crea un registro de vuelo y publica el evento VueloConfirmado."""
    if request.pnr in _vuelos_db:
        raise HTTPException(
            status_code=409, detail=f"El PNR {request.pnr} ya existe."
        )

    registro = {
        "tenant_id": request.tenant_id,
        "pnr": request.pnr,
        "vuelo_id": request.vuelo_id,
        "hora_salida_utc": request.hora_salida_utc.astimezone(timezone.utc),
        "hora_llegada_utc": request.hora_llegada_utc.astimezone(timezone.utc),
        "minutos_retraso": 0,
    }
    _vuelos_db[request.pnr] = registro

    evento = VueloConfirmado(
        tenant_id=request.tenant_id,
        pnr=request.pnr,
        vuelo_id=request.vuelo_id,
        hora_salida_utc=request.hora_salida_utc,
        hora_llegada_utc=request.hora_llegada_utc,
    )
    await stream_bus.publish(EVENTS_STREAM, evento)

    return registro


@app.put("/api/v1/vuelos/{pnr}/retraso")
async def registrar_retraso(
    pnr: str,
    request: RegistrarRetrasoRequest,
    stream_bus: RedisStreamBus = Depends(get_bus),
) -> dict:
    """Actualiza el retraso de un vuelo y publica el evento VueloRetrasado."""
    registro = _vuelos_db.get(pnr)
    if registro is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr} no encontrado.")

    registro["minutos_retraso"] = request.minutos_retraso
    registro["hora_salida_utc"] = request.nueva_hora_salida_utc.astimezone(
        timezone.utc
    )

    evento = VueloRetrasado(
        tenant_id=request.tenant_id,
        pnr=pnr,
        nueva_hora_salida_utc=request.nueva_hora_salida_utc,
        minutos_retraso=request.minutos_retraso,
    )
    await stream_bus.publish(EVENTS_STREAM, evento)

    return registro
