# Predicción de cancelaciones hoteleras

## Autores

- Paris Arcos
- Lucia Mateo

## Descripción del Proyecto

Sistema automático para predecir cancelaciones de reservas hoteleras usando múltiples modelos de Machine Learning.

**Problema:** Clasificación binaria para predecir si una reserva será cancelada (`is_canceled = 1`) o no (`is_canceled = 0`).

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.10+
- **Librerías Clave:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Entorno:** Jupyter Notebooks (para experimentación) y Scripts modulares en producción

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
│   ├── data_loader.py          # Carga el CSV, aplica la limpieza del EDA.
│   ├── preprocessor.py         # Recibe el dataframe limpio, realiza el split aplica encoding y scaling según el modelo.
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

Lorep ipsum

## Roles de la Pareja

Lorep ipsum

## Resultados y Conclusiones

Lorep ipsum

## Limitaciones y Mejoras Futuras

Lorep ipsum
