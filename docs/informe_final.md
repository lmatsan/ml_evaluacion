# Informe Final del Proyecto

## 1. Participantes

- Paris Arcos
- Lucia Mateo

## 2. Justificación del problema

El sector hotelero se enfrenta constantemente a un problema crítico: la incertidumbre en la ocupación real de sus habitaciones. Las cancelaciones de reservas de última hora desbaratan la planificación de las habitaciones, desajustan los turnos del personal y provocan pérdidas económicas directas que muchas veces son difíciles de recuperar.

Este proyecto nace para dar una solución a este problema mediante el uso de datos. El objetivo es desarrollar un modelo predictivo capaz de anticiparse a estas cancelaciones basándose en el comportamiento histórico de los clientes: cuánto tiempo antes reservan, cuántos días se van a quedar, si viajan con niños, el canal por el que reservaron o si ya han cancelado alguna vez en el pasado.

La variable que queremos predecir es `is_canceled`, que se organiza de forma binaria:

- **1 (Positivo):** Si el cliente termina cancelando la reserva antes de llegar al hotel.
- **0 (Negativo):** Si la reserva sigue adelante y el cliente realiza el ingreso (_check-in_) con normalidad.

### ¿Por qué es importante resolver esto?

Disponer de una predicción fiable aporta ventajas reales tanto para el negocio como para la experiencia del usuario:

- **Para el hotel:** Saber qué reservas tienen un riesgo real de caerse permite actuar con margen. Ayuda a gestionar el inventario de habitaciones de forma inteligente, ajustar la compra de suministros, organizar mejor los turnos del equipo y, si es necesario, aplicar estrategias de _overbooking_ basadas en datos reales y no en intuiciones.
- **Para el cliente:** Si el sistema detecta patrones de cancelación con antelación, las plataformas pueden ofrecer políticas más flexibles o alternativas de reprogramación antes de aplicar penalizaciones agresivas. Esto evita sorpresas financieras para el usuario y mejora su relación con la marca.

---

> ### 📊 En pocas palabras
>
> El propósito de este trabajo es utilizar los datos de las reservas para construir un modelo que identifique qué clientes tienen más probabilidades de cancelar. De este modo, el hotel puede adelantarse al imprevisto, proteger sus ingresos y planificar el día a día con mucha más seguridad.

## 3. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio de este conjunto de datos se diseñó no solo para entender la distribución de las variables, sino para tomar decisiones estratégicas de limpieza y detectar comportamientos atípicos antes de entrenar los modelos. Tras una depuración inicial, el volumen de datos pasó de 119.390 registros originales a 84.555 registros.

### 3.1. Diagnóstico de Calidad, Duplicados y Data Leakage

Uno de los desafíos más interesantes del EDA fue la gestión de los registros duplicados, que representaban el 27% del total. Eliminar más de 31.000 filas idénticas a la ligera habría sido un error de negocio, ya que en el sector hotelero es habitual que una agencia o familia reserve varias habitaciones idénticas de golpe. Sin embargo, mantenerlas en el entrenamiento provocaría _Data Leakage_ (sobreajuste por repetición).

La solución adoptada fue colapsar esos duplicados en registros únicos, pero creando una nueva variable llamada `room_count`. De este modo, protegemos la matemática del modelo sin perder la información del volumen de la reserva.

Por otro lado, se identificaron y subsanaron los siguientes puntos críticos:

- **Fuga de datos (Leakage directo):** Las variables `reservation_status` y `reservation_status_date` predecían el futuro (avisaban de si el cliente ya había hecho el _check-out_ o cancelado). Se eliminaron de inmediato para evitar modelos falsamente perfectos.
- **Valores nulos e inconsistencias:** Los nulos en `children` se asumieron como ausencia de niños (0). Los identificadores de `agent` y `company` vacíos se catalogaron como "reserva directa". En las variables numéricas de precio (`adr`), se eliminaron registros con valores negativos o superiores al percentil 99 por tratarse de errores de registro u _outliers_ extremos.

### 3.2. Hallazgos Clave y Comportamiento del Cliente

El análisis bivariado reveló patrones de comportamiento muy marcados que separan a los clientes que cancelan de los que finalmente se hospedan:

- **El factor tiempo (`lead_time`):** Es el predictor más potente. Los clientes que cancelan reservan con una mediana de 80 días de antelación, frente a los 38 días de los que sí acuden. A mayor tiempo de espera, más margen para imprevistos o cambios de planes.
- **Sensibilidad al precio (`adr`):** Las reservas con tarifas diarias medias más altas muestran una tasa de cancelación superior (mediana de 108€ vs 93€). Esto sugiere un comportamiento de comparación activa: el cliente asegura una habitación cara, pero sigue buscando y cancela si encuentra una alternativa más económica.
- **Variables de compromiso y planificación:** Existe una relación inversa entre el esfuerzo que invierte el cliente en su viaje y la probabilidad de cancelarlo. Por ejemplo, **ninguna de las reservas canceladas había solicitado plaza de parking** (`required_car_parking_spaces`). Asimismo, un mayor número de peticiones especiales (`total_of_special_requests`) correlaciona directamente con reservas más seguras.
- **Segmentación y Canales:** Los hoteles urbanos (_City Hotel_) registran mayor inestabilidad que los vacacionales (_Resort Hotel_). Además, los canales online (`Online TA`) sufren un volumen de cancelaciones drásticamente mayor (35%) en comparación con las reservas corporativas (12%).

### 3.3. Preparación para el Modelado

El desbalanceo moderado detectado en la variable objetivo `is_canceled` (63% de estancias frente a un 37% de cancelaciones) dictaminó que la métrica de acierto general (_Accuracy_) no será un indicador fiable para evaluar los algoritmos. En su lugar, el proyecto priorizará el uso del **Área bajo la curva ROC (AUC-ROC)** y el **F1-Score**.

Finalmente, el comportamiento de las variables obligó a realizar un trabajo de ingeniería de características (_Feature Engineering_) diferenciado según el modelo:

- **Reducción de cardinalidad:** La variable `country` contenía 175 categorías inmanejables; se simplificó agrupando el top de países emisores y englobando el resto en la etiqueta `OTHER`.
- **Simplificación lógica:** Los identificadores ordinales de `agent` y `company` se transformaron en variables binarias (`has_agent` / `has_company`) para medir el impacto de intermediarios.
- **Estrategias de preprocesamiento:** El EDA dejó claro que modelos sensibles a la escala (como la Regresión Logística o las Redes Neuronales) requerirán un escalado estricto (`StandardScaler` / `MinMaxScaler`) y codificación _One-Hot_, mientras que los modelos basados en árboles (Random Forest o CatBoost) trabajarán de forma nativa y óptima con codificaciones tipo _Label Encoding_.

## 4. Diseño del Sistema

Para garantizar que el modelo seleccionado no se quede únicamente en un entorno de experimentación en local, se ha diseñado una arquitectura modular basada en principios de MLOps. El sistema está dividido en dos grandes fases: el **ciclo de entrenamiento y tracking** (desarrollado en batch) y el **ciclo de servicio en producción** (ejecutado en tiempo real a través de una API).

### 4.1. Arquitectura del Sistema y Flujo de Datos

El diseño del sistema sigue un pipeline desacoplado que asegura que cualquier cambio en los datos o en los modelos pueda actualizarse de forma independiente sin romper la aplicación final. El flujo funciona de la siguiente manera:

1. **Ingesta y Preprocesamiento:** Los datos históricos se limpian y transforman aplicando las reglas maduradas en el EDA (control de outliers, binarización y colapso de duplicados con `room_count`).
2. **Entrenamiento y Optimización:** El script principal ejecuta de golpe el entrenamiento de los modelos clásicos y la red neuronal. Si detecta que faltan hiperparámetros en la configuración, activa un proceso de búsqueda inteligente (`GridSearchCV`).
3. **Gobierno del Modelo (Tracking):** Cada entrenamiento, parámetro y métrica de evaluación se registra bajo un árbol jerárquico en **MLflow**, vinculando de forma estricta los sub-runs hijos con el experimento raíz para asegurar la trazabilidad absoluta del proyecto.
4. **Exportación de Artefactos:** Una vez seleccionado el mejor algoritmo, el sistema exporta de manera automatizada tanto el archivo binario del modelo entrenado como el diccionario de reglas de preprocesamiento (`preprocessing_rules.joblib`), además de generar los reportes visuales de rendimiento.

### 4.2. Despliegue y Consumo en Producción (`predictor.py`)

El consumo del modelo en producción se realiza mediante una interfaz de programación de aplicaciones (API) que expone el script maestro `predictor.py`.

Cuando llega una nueva solicitud de reserva desde el software de gestión del hotel, la API realiza las siguientes acciones en milisegundos:

- **Carga Dinámica:** Al arrancar, el predictor lee de forma automática el mejor artefacto guardado en la fase de entrenamiento, garantizando que la API siempre consuma la versión más óptima sin necesidad de recodificar el servicio.
- **Transformación al Vuelo:** El preprocesador recibe los datos crudos de la nueva reserva y les aplica exactamente las mismas transformaciones matemáticas con las que se entrenó el modelo.
- **Inferencia:** El modelo procesa las características y devuelve una probabilidad. La API traduce esta salida en una respuesta binaria simple junto con su porcentaje de confianza, permitiendo al hotel saber en tiempo real si una reserva concreta corre el riesgo de ser cancelada.

### 4.3 Diagrama

```mermaid
graph TD
    %% Estilos Generales
    classDef fase/fases fill:#f9f9f9,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef datos fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef proceso fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef guardado fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    %% --- FASE 1: ENTRENAMIENTO BATCH ---
    subgraph FASE_1 [Ciclo de Entrenamiento y Tracking Batch]
        A[(Dataset Histórico <br> 119k registros)] --> B(Limpieza y EDA <br> Control outliers / nulos / duplicados)
        B --> C{¿Faltan <br> hiperparámetros?}

        C -- Sí --> D(Optimización con <br> GridSearchCV)
        C -- No --> E(Entrenamiento Directo <br> Modelos clásicos + Red Neuronal)
        D --> E

        E --> F[MLflow Tracking <br> Estructura Jerárquica / Tags]
        F --> G[Evaluación Cruzada <br> AUC-ROC / F1-Score]
    end

    %% --- PASO DE ARTEFACTOS ---
    G --> H[Elección del Mejor Modelo <br> Exportación de Artefactos]

    H --> I[preprocessing_rules.joblib]
    H --> J[best_model.joblib / .h5]

    %% --- FASE 2: PRODUCCIÓN ---
    subgraph FASE_2 [Ciclo de Servicio en Tiempo Real]
        K[Nueva Reserva Recibida <br> Software del Hotel] --> L(API de Producción <br> predictor.py)

        I -.->|Carga Dinámica| L
        J -.->|Carga Dinámica| L

        L --> M(Procesado datos reserva)
        M --> N(Inferencia del Modelo)
        N --> O[Respuesta API <br> Predicción Binaria + Confianza %]
    end

    %% Aplicación de estilosa
    class FASE_1,FASE_2 fase/fases;
    class A,K,I,J datos;
    class B,C,D,E,M,N proceso;
    class F,G,H,L,O guardado;
```

## 5. Resultados y Elección Final

El proceso de modelado se ejecutó en tres etapas incrementales, permitiendo evaluar el impacto directo del tratamiento del desbalanceo de clases y la optimización de hiperparámetros en el comportamiento del negocio.

### 5.1. Evolución de las Métricas y Experimentos

#### Etapa 1: Modelo Baseline (Datos Originales)

Utilizando la distribución nativa de los datos (63% estancias / 37% cancelaciones), los algoritmos mostraron un sesgo marcadamente conservador. Aunque las métricas de acierto general (_Accuracy_) y precisión eran aceptables, el _Recall_ promedio fue deficiente en todos los casos.

| Model Name              | Accuracy | Precision | Recall |   F1   | ROC-AUC |
| :---------------------- | :------: | :-------: | :----: | :----: | :-----: |
| **Neural Network**      |  0.8379  |  0.7388   | 0.6279 | 0.6789 | 0.8988  |
| **CatBoost**            |  0.8377  |  0.7474   | 0.6115 | 0.6727 | 0.8998  |
| **Random Forest**       |  0.8142  |  0.7850   | 0.4394 | 0.5634 | 0.8801  |
| **Decision Tree**       |  0.8094  |  0.6667   | 0.6025 | 0.6330 | 0.8624  |
| **Logistic Regression** |  0.7924  |  0.6723   | 0.4659 | 0.5504 | 0.8424  |

- **Interpretación de negocio:** El sistema era incapaz de detectar más de la mitad de las cancelaciones reales (destacando el pobre 43.9% de _Recall_ en Random Forest). Para el hotel, operar con un modelo así anula su valor preventivo, ya que la gran mayoría de las fugas de clientes pasarían desapercibidas.

#### Etapa 2: Estrategia Balanceada (`class_weight='balanced'`)

Para corregir el problema anterior, se introdujeron penalizaciones por peso para obligar a los algoritmos a prestar más atención a la clase minoritaria (las cancelaciones).

| Model Name              | Accuracy | Precision | Recall |   F1   | ROC-AUC |
| :---------------------- | :------: | :-------: | :----: | :----: | :-----: |
| **Neural Network**      |  0.8348  |  0.7660   | 0.5678 | 0.6522 | 0.8960  |
| **CatBoost**            |  0.7971  |  0.5887   | 0.8502 | 0.6957 | 0.8996  |
| **Decision Tree**       |  0.7602  |  0.5405   | 0.8064 | 0.6472 | 0.8596  |
| **Random Forest**       |  0.7531  |  0.5292   | 0.8611 | 0.6555 | 0.8811  |
| **Logistic Regression** |  0.7506  |  0.5287   | 0.7908 | 0.6337 | 0.8426  |

- **Interpretación de negocio:** Los niveles de _Recall_ se dispararon por encima del 80% (Random Forest llegó al 86.1%), pero generaron un efecto adverso de "falsas alarmas", desplomando la precisión general al entorno del 52%-58%. Operar con este modelo implicaría que el hotel daría por canceladas muchas reservas que en realidad eran seguras, provocando un _overbooking_ descontrolado que arruinaría la experiencia de clientes reales.

#### Etapa 3: Optimización con GridSearchCV (Ajuste Fino)

La tercera fase consistió en realizar una búsqueda exhaustiva de hiperparámetros para solucionar el conflicto analítico previo (_Precision-Recall Trade-Off_) y encontrar un punto de equilibrio real.

| Model Name              | Accuracy | Precision | Recall |   F1   | ROC-AUC |
| :---------------------- | :------: | :-------: | :----: | :----: | :-----: |
| **CatBoost**            |  0.8428  |  0.7570   | 0.6242 | 0.6842 | 0.9061  |
| **Random Forest**       |  0.8390  |  0.6927   | 0.7363 | 0.7139 | 0.9077  |
| **Neural Network**      |  0.8387  |  0.7426   | 0.6258 | 0.6792 | 0.8982  |
| **Decision Tree**       |  0.7735  |  0.5569   | 0.8295 | 0.6664 | 0.8728  |
| **Logistic Regression** |  0.7508  |  0.5290   | 0.7900 | 0.6336 | 0.8428  |

- **Interpretación de negocio:** La optimización estabilizó ambas métricas en rangos óptimos y controlados (ambas por encima del 69% en los mejores modelos), incrementando significativamente la robustez global del sistema medida a través de la métrica armónica _F1-Score_ y el área bajo la curva (_AUC-ROC_), que superó la barrera del 90%.

**Conclusiones**

1. **Modelo Baseline:** Utilizando la distribución nativa de los datos, los algoritmos mostraron un sesgo conservador. Aunque la precisión general era aceptable, el _Recall_ promedio fue deficiente (destacando el 43.9% en Random Forest). En términos hoteleros, el sistema era incapaz de detectar más de la mitad de las cancelaciones reales, anulando su valor predictivo.
2. **Estrategia Balanceada:** La introducción de penalizaciones por peso (`class_weight='balanced'`) corrigió el sesgo de detección, elevando los niveles de _Recall_ por encima del 80%. No obstante, generó un efecto adverso de "falsas alarmas", reduciendo la precisión al entorno del 52%-58%. Para el hotel, operar con este modelo implicaría una tasa muy alta de errores de predicción, comprometiendo la confianza de la estrategia de _overbooking_.
3. **Optimización con GridSearchCV:** La sintonización fina de hiperparámetros resolvió el conflicto analítico (_Precision-Recall Trade-Off_). Esta última fase estabilizó ambas métricas en rangos óptimos, incrementando significativamente la robustez global del sistema medida a través del área bajo la curva (_AUC-ROC_).

### 5.2. Justificación del Modelo Elegido

Tras realizar la optimización de toda la batería de algoritmos, se seleccionó **Random Forest** como el modelo definitivo para producción, fundamentado en los siguientes criterios:

- **Puntuación F1 Superior (0.7139):** Al ser la media armónica entre precisión y sensibilidad, un F1-Score dominante demuestra que es el modelo más equilibrado y el que mejor mitiga tanto las falsas alarmas como las cancelaciones imprevistas.
- **Maximización del Recall (0.7363):** En el contexto predictivo de cancelaciones, el coste de omitir una cancelación (Falso Negativo) suele ser más perjudicial para el inventario que lanzar una alerta errónea. Random Forest es capaz de capturar el 73.6% de las cancelaciones reales, superando en más de 11 puntos porcentuales a CatBoost (`0.6242`).
- **Excelente Capacidad de Discriminación (AUC-ROC = 0.9077):** Un valor que roza el 91% en la curva ROC garantiza que el modelo separa las clases con mucha fiabilidad, asegurando la estabilidad de la API en el entorno de producción (`predictor.py`).

## 6. Reflexión Crítica: Limitaciones y Mejoras Futuras

Aunque el sistema desarrollado ofrece un rendimiento robusto (con un AUC-ROC cercano al 91% en el modelo ganador), la puesta en marcha de un proyecto de Machine Learning en un entorno de producción real siempre desvela limitaciones técnicas y operativas que deben analizarse críticamente para garantizar su viabilidad a largo plazo.

### 6.1. Limitaciones Identificadas en el Proyecto

- **Incertidumbre en la gestión de duplicados:** Alrededor del 27% del dataset original consistía en registros idénticos. Aunque se resolvió de forma técnica agrupándolos y creando la variable `room_count` para proteger al modelo del _Data Leakage_, en el mundo real esto genera cierta ceguera de negocio. Al no disponer de un identificador único de usuario (_User ID_), no podemos saber con total certeza si esos duplicados eran errores del sistema de reservas o reservas masivas legítimas de grupos o turoperadores.
- **Falta de contexto macroeconómico y temporal:** Las cancelaciones hoteleras dependen fuertemente de factores externos que no están presentes en las variables originales del dataframe. Cuestiones como la climatología imprevista, huelgas de aerolíneas, crisis de inflación o la fluctuación de tarifas de la competencia directa alteran drásticamente el comportamiento del cliente, y el modelo actual es ajeno a este contexto.
- **Riesgo latente de Degradación del Modelo (_Data Drift_):** El turismo es un sector vivo que evoluciona por modas, cambios demográficos o la aparición de nuevos canales de venta. Un modelo entrenado con datos estáticos del pasado puede quedar deprecado si no se vigila convenientemente.

### 6.2. Líneas de Mejora y Trabajo Futuro

Para evolucionar esta solución analítica hacia una plataforma predictiva de nivel empresarial, se proponen las siguientes mejoras:

- **Estrategia de Reentrenamiento Automatizado (Pipeline de MLOps Continuo):** En lugar de ejecuciones manuales aisladas, el siguiente paso lógico es programar tareas en batch (por ejemplo, con herramientas como Apache Airflow o GitHub Actions) que extraigan los datos del último mes y reentrenen automáticamente el Random Forest. MLflow permitiría comparar las métricas del modelo nuevo con el de producción para sustituirlo de forma segura si el comportamiento del cliente cambia.
- **Enriquecimiento del Dataset (_Feature Enrichment_):** Incorporar fuentes de datos externas mediante APIs públicas. Añadir variables como la previsión meteorológica a medio plazo, el calendario de festivos locales e internacionales o un índice de precios de los competidores directos de la zona aportaría el contexto que hoy le falta al modelo.
- **Análisis de la Incertidumbre en la API (`predictor.py`):** Modificar el servicio de producción para que no devuelva simplemente un "0" o un "1", sino que exponga el porcentaje de probabilidad exacto de la predicción de Random Forest. Esto permitiría al hotel segmentar sus acciones comerciales: por ejemplo, lanzar campañas agresivas de fidelización solo a los clientes con un riesgo de cancelación superior al 80%, y ofrecer alternativas flexibles a los que se encuentren en un rango intermedio de riesgo (50%-70%).
