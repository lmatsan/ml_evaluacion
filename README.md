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
- **Entorno de trabajo:** Jupyter Notebooks (para EDA y experimentación) y scripts modulares en `src/`

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos preprocesados
|
├── docs/                       # Documentacion adicional
|
├── models/                     # Modelos entrenados
│   ├── tests/                  # Modelos intermedios de prueba
│   └── best_model.pkl          # El mejor modelo
|
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carga de datos
│   ├── preprocessor.py         # Preprocesamiento
│   ├── model_trainer.py        # Entrenamiento
│   └── evaluator.py            # Evaluación
│   └── predictor.py            # Funciones para hacer predicciones con modelos entrenados
|
├── notebooks/                  # Notebooks exploratorios
│   └── EDA_hotel_bookings.ipynb
|
├── outputs/                    # Resultados y gráficos
|
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

> 💡 **Nota:** Al abrir el notebook, recuerda cambiar el kernel a **"ML Hotel"** desde el menú superior.

## Ejecución

python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python -m src.model_trainer

## Roles de la Pareja

Lorep ipsum

## Resultados y Conclusiones

- Hemos elegido AUC-ROC como métrica principal porque el dataset no está perfectamente equilibrado: hay más reservas no canceladas que canceladas. Por eso, medir solo el porcentaje de aciertos puede engañar. AUC-ROC nos ayuda a ver mejor si el modelo realmente distingue bien entre reservas que se cancelan y reservas que no.
- Elegimos "CatBoost" como modelo principal dentro de los modelos de boosting porque funciona muy bien con datasets en formato tabla, no requiere un preprocesamiento tan complejo y además su implementacion es mas sencilla.

## Limitaciones y Mejoras Futuras

Lorep ipsum
