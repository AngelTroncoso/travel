"""Esquemas REST (Pydantic v2) para las llamadas sincronas entre agentes.

Cubre la consulta de disponibilidad que el Agente de Agenda hace por
HTTP al Agente de Alojamiento antes de confirmar/rearmar un itinerario.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class DisponibilidadRequest(BaseModel):
    """Request que el Agente de Agenda envia al Agente de Alojamiento."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(..., min_length=1)
    hotel_id: str = Field(..., min_length=1)
    check_in: date
    check_out: date
    habitaciones_requeridas: int = Field(1, ge=1)


class DisponibilidadResponse(BaseModel):
    """Response del Agente de Alojamiento con el resultado de la consulta."""

    model_config = ConfigDict(frozen=True)

    hotel_id: str
    check_in: date
    check_out: date
    disponible: bool
    habitaciones_disponibles: int = Field(0, ge=0)
    motivo_no_disponible: str | None = None
