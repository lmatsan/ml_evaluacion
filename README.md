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
│   ├── data_loader.py          # Carga el CSV, aplica la limpieza del EDA.
│   ├── preprocessor.py         # Recibe el dataframe limpio, realiza el split aplica encoding y scaling según el modelo.
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

## Flujo de entrenamiento

Este es el flujo simplificado que sigue el proyecto cuando se ejecuta `python -m src.model_trainer`:

1. Se carga el CSV completo con los datos originales.
   Funciones principales: `prepare_training_datasets()` llama a `load_dataset()` y `validate_dataset_structure()`.
2. Se hace una limpieza base del dataset.
   Función principal: `clean_dataset()`. Aquí se eliminan duplicados, se quitan columnas con leakage y se filtran registros no válidos.
3. Se aplican transformaciones fijas.
   Función principal: `apply_fixed_preprocessing()`. Aquí se convierten tipos con diccionarios, se rellenan nulos y se transforma `arrival_date_month`.
4. Se divide el dataset en entrenamiento y validación.
   Función principal: `split_train_validation()`. Aquí se ejecuta `train_test_split` con `stratify`.
5. Se aprenden reglas solo con el conjunto de entrenamiento.
   Función principal: `fit_preprocessing_rules()`. Aquí se calculan, por ejemplo, el país más frecuente, los países principales y el umbral superior de `adr`.
6. Esas reglas se aplican tanto a entrenamiento como a validación.
   Función principal: `apply_preprocessing_rules()`.
7. Se separan las variables de entrada (`X`) y la variable objetivo (`y`).
   Función principal: `split_features_and_target()`.
8. Se entrenan todos los modelos definidos en el proyecto.
   Funciones principales: `train_all_models()`, `train_classical_model()` y `train_neural_network_model()`.
9. Cada modelo se evalúa con el conjunto de validación.
   Función principal: `evaluate_classification_model()`. Aquí se calculan accuracy, precision, recall, f1 y roc_auc, además de la matriz de confusión y la curva ROC.
10. Se comparan los resultados y se elige el mejor modelo.
    Funciones principales: `build_model_comparison()` y `select_best_model()`.
11. Se guardan el informe, los gráficos y los artefactos finales en `outputs/`.
    Funciones principales: `export_training_report()`, `export_comparative_roc_curve()`, `export_confusion_matrix_figure()`, `export_random_forest_feature_importance()` y `save_best_artifact()`.

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

## Roles

Lorep ipsum

## Resultados y Conclusiones

- Hemos elegido AUC-ROC como métrica principal porque el dataset no está perfectamente equilibrado: hay más reservas no canceladas que canceladas. Por eso, medir solo el porcentaje de aciertos puede engañar. AUC-ROC nos ayuda a ver mejor si el modelo realmente distingue bien entre reservas que se cancelan y reservas que no.
- Elegimos `CatBoost` como modelo principal dentro de los modelos de boosting porque funciona muy bien con datasets en formato tabla, no requiere un preprocesamiento tan complejo y además su implementación es más sencilla.
- Para mejorar la organización y mantenibilidad del código, se extrajo la lógica de preprocesamiento de model_trainer.py a un módulo independiente, preprocessor.py. De esta forma, el entrenamiento del modelo y la preparación de los datos quedan separados en responsabilidades distintas.

## Limitaciones y Mejoras Futuras

Lorep ipsum

## Seguimiento de Experimentos (MLflow)

Este proyecto utiliza **MLflow** para registrar los hiperparámetros, métricas y modelos generados en cada entrenamiento. Esto permite comparar diferentes ejecuciones y mantener un registro del "modelo ganador".

### Cómo ver los experimentos localmente:

1. Asegúrate de tener las dependencias instaladas.
2. Levanta el servidor local de MLflow ejecutando el siguiente comando en la raíz del proyecto:
   ```bash
   mlflow ui
   ```

```
3. Abre tu navegador e ingresa a http://127.0.0.1:5000 para ver el panel de control con las gráficas y comparativas.
> **Nota:** La carpeta mlruns/ está excluida en el .gitignore para evitar subir archivos pesados (como los modelos .pkl) al repositorio. Cada entrenamiento se guarda localmente en tu máquina.
```
