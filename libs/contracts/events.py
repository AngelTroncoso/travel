"""Contratos de eventos de dominio (Pydantic v2).

Todo evento publicado por los agentes en Redis Streams debe modelarse
aqui, heredando de BaseEvent. Estos modelos son el "schema" que viaja
como payload (serializado a JSON) en RedisStreamBus.publish().
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    """Normaliza un datetime a UTC, asumiendo UTC si no trae tzinfo."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class BaseEvent(BaseModel):
    """Modelo base del que heredan todos los eventos de dominio.

    - event_id: identificador unico del evento (UUID v4).
    - timestamp: momento de creacion del evento, siempre normalizado a UTC.
    - tenant_id: obligatorio, usado para el aislamiento multi-tenant en
      Redis Streams (filtrado por tenant) y en la persistencia (SQLite).
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utc_now)
    tenant_id: str = Field(..., min_length=1)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


# --- Eventos: Agente de Transporte -----------------------------------------


class VueloRetrasado(BaseEvent):
    event_type: Literal["VueloRetrasado"] = "VueloRetrasado"
    pnr: str = Field(..., min_length=1, description="Codigo de reserva del vuelo")
    nueva_hora_salida_utc: datetime
    minutos_retraso: int = Field(..., ge=0)

    @field_validator("nueva_hora_salida_utc")
    @classmethod
    def _validate_nueva_hora(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


class VueloConfirmado(BaseEvent):
    event_type: Literal["VueloConfirmado"] = "VueloConfirmado"
    pnr: str = Field(..., min_length=1)
    vuelo_id: str = Field(..., min_length=1)
    hora_salida_utc: datetime
    hora_llegada_utc: datetime

    @field_validator("hora_salida_utc", "hora_llegada_utc")
    @classmethod
    def _validate_horas(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


# --- Eventos: Agente de Alojamiento -----------------------------------------


class DisponibilidadConfirmada(BaseEvent):
    event_type: Literal["DisponibilidadConfirmada"] = "DisponibilidadConfirmada"
    hotel_id: str = Field(..., min_length=1)
    check_in: date
    check_out: date
    disponible: bool
    habitaciones_disponibles: int = Field(0, ge=0)


class ReservaHotelConfirmada(BaseEvent):
    event_type: Literal["ReservaHotelConfirmada"] = "ReservaHotelConfirmada"
    reserva_id: str = Field(..., min_length=1)
    hotel_id: str = Field(..., min_length=1)
    check_in: date
    check_out: date


# --- Eventos: Agente de Agenda ----------------------------------------------


class ItinerarioActualizado(BaseEvent):
    event_type: Literal["ItinerarioActualizado"] = "ItinerarioActualizado"
    itinerary_id: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    resumen_cambio: str = Field(..., min_length=1)


class ConflictoRequiereIntervencion(BaseEvent):
    event_type: Literal["ConflictoRequiereIntervencion"] = (
        "ConflictoRequiereIntervencion"
    )
    itinerary_id: str = Field(..., min_length=1)
    motivo: str = Field(..., min_length=1)
    detalle: str | None = None
    requiere_humano: bool = True
