from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.predictor import predict_booking_from_payload


class BookingPredictionRequest(BaseModel):
    hotel: str = Field(..., description="Tipo de hotel (City Hotel o Resort Hotel)")
    lead_time: int = Field(..., description="Número de días entre la reserva y la fecha de llegada")
    arrival_date_year: int = Field(..., description="Año de llegada")
    arrival_date_month: str = Field(..., description="Mes de llegada")
    # arrival_date_week_number: int = Field(..., description="Número de semana de llegada")
    arrival_date_day_of_month: int = Field(..., description="Día del mes de llegada")
    stays_in_weekend_nights: int = Field(..., description="Número de noches de fin de semana incluidas en la reserva")
    stays_in_week_nights: int = Field(..., description="Número de noches entre semana incluidas en la reserva")
    adults: int = Field(..., description="Número de adultos")
    children: int = Field(..., description="Número de niños (0 si no aplica)")
    babies: int = Field(..., description="Número de bebés (0 si no aplica)")
    meal: str = Field(..., description="Tipo de régimen alimenticio reservado")
    country: str = Field(..., description="País de origen del cliente")
    market_segment: str = Field(..., description="Canal de captación de la reserva")
    distribution_channel: str = Field(..., description="Canal de distribución usado para formalizar la reserva")
    is_repeated_guest: int = Field(..., description="Indica si es un cliente repetido")
    previous_cancellations: int = Field(..., description="Número de cancelaciones anteriores")
    previous_bookings_not_canceled: int = Field(..., description="Número de reservas anteriores no canceladas")
    reserved_room_type: str = Field(..., description="Tipo de habitación reservada")
    assigned_room_type: str = Field(..., description="Tipo de habitación asignada")
    booking_changes: int = Field(..., description="Número de cambios realizados en la reserva")
    deposit_type: str = Field(..., description="Tipo de depósito requerido para la reserva")
    days_in_waiting_list: int = Field(..., description="Número de días que la reserva estuvo en lista de espera")
    customer_type: str = Field(..., description="Tipo de cliente según su comportamiento de reserva")
    adr: float = Field(..., description="Tarifa media diaria de la reserva")
    required_car_parking_spaces: int = Field(..., description="Número de espacios de aparcamiento requeridos")
    total_of_special_requests: int = Field(..., description="Número total de peticiones especiales realizadas por el cliente")
    # reservation_status: str = Field(..., description="Estado final de la reserva (Cancelada o No Cancelada)")
    # reservation_status_date: str = Field(..., description="Fecha en la que se registró el estado final de la reserva (YYYY-MM-DD)")

    # Campos opcionales
    agent: int | None = Field(None, description="Número de identificación del agente de viajes que formalizó la reserva")
    company: str | None = Field(None, description="ID de la empresa asociada a la reserva. Null si es reserva particular")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hotel": "City Hotel",
                    "lead_time": 45,
                    "arrival_date_year": 2024,
                    "arrival_date_month": "July",
                    "arrival_date_day_of_month": 15,
                    "stays_in_weekend_nights": 2,
                    "stays_in_week_nights": 3,
                    "adults": 2,
                    "children": 0,
                    "babies": 0,
                    "meal": "BB",
                    "country": "ESP",
                    "market_segment": "Online TA",
                    "distribution_channel": "TA/TO",
                    "is_repeated_guest": 0,
                    "previous_cancellations": 0,
                    "previous_bookings_not_canceled": 0,
                    "reserved_room_type": "A",
                    "assigned_room_type": "A",
                    "booking_changes": 0,
                    "deposit_type": "No Deposit",
                    "agent": None,
                    "company": None,
                    "days_in_waiting_list": 0,
                    "customer_type": "Transient",
                    "adr": 89.50,
                    "required_car_parking_spaces": 0,
                    "total_of_special_requests": 1,
                }
            ]
        }
    }


class BookingPredictionResponse(BaseModel):
    predicted_label: int = Field(..., description="Predicción binaria: 1 para cancelacion probable, 0 = reserva segura")
    predicted_class_name: str = Field(..., description="Etiqueta legible de la predicción")
    predicted_probability: float = Field(..., description="Probabilidad de cancelación estimada por el modelo (0.0 - 1.0)")
    model_type: str = Field(..., description="Tipo de modelo que realizó la inferencia")


app = FastAPI(
    title="Hotel Cancellation Predictor API",
    description="""
    ## Descripción
    API de predicción de cancelaciones hoteleras basada en un modelo **Random Forest** 
    entrenado sobre datos históricos de reservas.

    ## Uso
    Envía los datos de una reserva al endpoint `/predict` y obtendrás:
    - La predicción binaria (cancelable / no cancelable)
    - La probabilidad de cancelación
    - El tipo de modelo que realizó la inferencia
    """, 
    version="1.0.0",
    contact={
        "name": "Lucía Mateo / Paris Arcos",
    }
    )


# Recibe una reserva nueva y devuelve la predicción del modelo entrenado.
@app.post("/predict",
        response_model=BookingPredictionResponse,
        summary="Predice la probabilidad de cancelación de una reserva hotelera",
        description="""
        Recibe los datos de una reserva hotelera y devuelve la predicción del modelo.

        **Lógica de negocio:**
        - `predicted_probability` > 0.7 → riesgo alto de cancelación
        - `predicted_probability` entre 0.5 y 0.7 → riesgo moderado
        - `predicted_probability` < 0.5 → reserva probablemente segura

        **Notas:**
        - Los campos `agent` y `company` son opcionales (null = reserva directa sin intermediario)
        - El campo `country` acepta códigos ISO 3166-1 alpha-3 (ESP, PRT, GBR, FRA...)
        """,
            responses={
                200: {"description": "Predicción generada correctamente"},
                400: {"description": "Datos de entrada inválidos o mal formateados"},
                404: {"description": "Modelo entrenado no encontrado en el sistema de archivos"},
            },
        )
def predict_endpoint(
    payload: BookingPredictionRequest,
) -> BookingPredictionResponse:
    try:
        prediction_result = predict_booking_from_payload(payload.model_dump())
        return BookingPredictionResponse(**prediction_result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
