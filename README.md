# Predicción de cancelaciones hoteleras

## Autores

- Paris Arcos
- Lucia Mateo

## Descripción del Proyecto

Sistema automático para predecir cancelaciones de reservas hoteleras usando múltiples modelos de Machine Learning.

**Problema:** Clasificación binaria para predecir si una reserva será cancelada (`is_canceled = 1`) o no (`is_canceled = 0`).

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.10+
- **Manipulación y análisis de datos:** Pandas, NumPy, SciPy
- **Machine Learning clásico:** Scikit-Learn
- **Gradient Boosting:** CatBoost
- **Red neuronal multicapa:** TensorFlow Keras
- **Visualización y reporting:** Matplotlib, Seaborn, Plotly
- **Persistencia de modelos:** Joblib
- **API de inferencia:** FastAPI y Uvicorn
- **Entorno de trabajo:** Jupyter Notebooks para EDA y scripts modulares en `src/`

## Estructura del Proyecto

```text
.
├── data/
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos preprocesados
│
├── docs/                       # Documentación adicional
│
├── models/                     # Modelos entrenados
│   ├── tests/                  # Modelos intermedios de prueba
│   └── best_model.pkl          # El mejor modelo
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carga de datos
│   ├── preprocessor.py         # Preprocesamiento
│   ├── model_trainer.py        # Entrenamiento
│   ├── evaluator.py            # Evaluación
│   ├── predictor.py            # Lógica de inferencia
│   └── api.py                  # API FastAPI
│
├── notebooks/                  # Notebooks exploratorios
│   └── EDA_hotel_bookings.ipynb
│
├── outputs/                    # Resultados y gráficos
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

### 1. Entorno Virtual (Python 3.10)

Crea y activa el entorno según tu sistema operativo:

```bash
# Crear entorno (Asegúrate de usar Python 3.10)
python -m venv venv

# Activar en Linux / macOS
source venv/bin/activate

# Activar en Windows
.\venv\Scripts\Activate
```

### 2. Dependencias y Kernel de Jupyter

Con el entorno activado, ejecuta los siguientes comandos para instalar los paquetes y registrar el proyecto en Jupyter:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Registrar el Kernel en Jupyter
python -m ipykernel install --user --name ml-hotel --display-name "ML Hotel"
```

> **Nota:** Al abrir el notebook, recuerda cambiar el kernel a **"ML Hotel"** desde el menú superior.

## Ejecución

```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python -m src.model_trainer
```

## API

Para levantar la API en local:

```bash
python -m uvicorn src.api:app --reload
```

Endpoint:

```text
http://localhost:8000/predict
```

Contract (`POST /predict`):

```json
{
  "hotel": "Resort Hotel",
  "lead_time": 45,
  "arrival_date_year": 2017,
  "arrival_date_month": "July",
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 5,
  "stays_in_weekend_nights": 2,
  "stays_in_week_nights": 5,
  "adults": 2,
  "children": 0,
  "babies": 0,
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Online TA",
  "distribution_channel": "TA/TO",
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "reserved_room_type": "A",
  "assigned_room_type": "A",
  "booking_changes": 0,
  "deposit_type": "No Deposit",
  "agent": 9,
  "company": "no company",
  "days_in_waiting_list": 0,
  "customer_type": "Transient",
  "adr": 120.0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 1,
  "reservation_status": "Check-Out",
  "reservation_status_date": "2017-07-01"
}
```

## Roles de la Pareja

Lorep ipsum

## Resultados y Conclusiones

- Hemos elegido AUC-ROC como métrica principal porque el dataset no está perfectamente equilibrado: hay más reservas no canceladas que canceladas. Por eso, medir solo el porcentaje de aciertos puede engañar. AUC-ROC nos ayuda a ver mejor si el modelo realmente distingue bien entre reservas que se cancelan y reservas que no.
- Elegimos `CatBoost` como modelo principal dentro de los modelos de boosting porque funciona muy bien con datasets en formato tabla, no requiere un preprocesamiento tan complejo y además su implementación es más sencilla.

## Limitaciones y Mejoras Futuras

Lorep ipsum
