from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from src.predictor import predict_booking_from_payload


class BookingPredictionRequest(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: int
    company: str
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str
    reservation_status_date: str


class BookingPredictionResponse(BaseModel):
    predicted_label: int
    predicted_class_name: str
    predicted_probability: float
    model_type: str


app = FastAPI(title="Cancel Predictor")


# Recibe una reserva nueva y devuelve la predicción del modelo entrenado.
@app.post("/predict", response_model=BookingPredictionResponse)
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
