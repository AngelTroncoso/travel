"""Libreria compartida de contratos (Pydantic v2) entre los tres agentes."""

from .api import DisponibilidadRequest, DisponibilidadResponse
from .events import (
    BaseEvent,
    ConflictoRequiereIntervencion,
    DisponibilidadConfirmada,
    ItinerarioActualizado,
    ReservaHotelConfirmada,
    VueloConfirmado,
    VueloRetrasado,
)

__all__ = [
    "BaseEvent",
    "VueloRetrasado",
    "VueloConfirmado",
    "DisponibilidadConfirmada",
    "ReservaHotelConfirmada",
    "ItinerarioActualizado",
    "ConflictoRequiereIntervencion",
    "DisponibilidadRequest",
    "DisponibilidadResponse",
]
