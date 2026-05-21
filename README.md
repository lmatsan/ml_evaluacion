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

1. Crear entorno virtual
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate

2. Instalar dependencias (todas incluidas)
   pip install -r requirements.txt

3. Para el kernel de Jupyter (ver notebooks/SETUP.md)
   python -m ipykernel install --user --name ml-hotel --display-name "ML Hotel"

## Ejecución

Lorep ipsum

## Roles de la Pareja

Lorep ipsum

## Resultados y Conclusiones

Lorep ipsum

## Limitaciones y Mejoras Futuras

Lorep ipsum
