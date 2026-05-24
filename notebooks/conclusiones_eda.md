# 📋 Conclusiones del EDA

## 1. Dataset

| Parámetro               | Valor                             |
| ----------------------- | --------------------------------- |
| Registros originales    | 119.390                           |
| Registros tras limpieza | ~84.555                           |
| Variables originales    | 32                                |
| Variables eliminadas    | 3                                 |
| Tipo de problema        | Clasificación binaria supervisada |

> _Variables eliminadas_: reservation_status, reservation_status_date y arrival_date_week_number

---

## 2. Variable objetivo

| Clase            | Registros | Proporción |
| ---------------- | --------- | ---------- |
| No cancelada (0) | 75.166    | 63%        |
| Cancelada (1)    | 44.224    | 37%        |

> **Desbalance moderado (63/37).** `accuracy` no es una métrica válida como referencia. Se usarán **AUC-ROC** y **F1-Score** como métricas principales. SMOTE se evaluará como mejora opcional.

---

## 3. Calidad de los datos

**Nulos imputados:**

- `children` → 0 (ausencia = sin niños)
- `agent` → `'no agent'` (reserva directa)
- `company` → `'no company'` (cliente directo)
- `country` → moda (`PRT`, bajo % de nulos)

**Duplicados:** eliminados directamente (error de registro en contexto hotelero).

**Outliers en `adr`:** valores negativos eliminados + > percentil 99 eliminados.

---

## 4. Data Leakage

| Variable                  | Tipo            | Decisión     |
| ------------------------- | --------------- | ------------ |
| `reservation_status`      | Leakage directo | ❌ Eliminada |
| `reservation_status_date` | Leakage directo | ❌ Eliminada |

---

## 5. Hallazgos bivariados

### Variables numéricas

**`lead_time`** — predictor clave:

| Clase        | Media    | Mediana |
| ------------ | -------- | ------- |
| Cancelada    | 106 días | 80 días |
| No cancelada | 70 días  | 38 días |

> A mayor antelación, mayor probabilidad de cancelación. La diferencia en medianas (80 vs 38 días) es el indicador más robusto.

**`adr`** — diferencia moderada pero consistente:

| Clase        | Media  | Mediana |
| ------------ | ------ | ------- |
| Cancelada    | 111,7€ | 108€    |
| No cancelada | 98,0€  | 93€     |

> Las reservas más caras se cancelan más. El cliente busca mejores precios y cancela si encuentra una alternativa más económica.

**`required_car_parking_spaces`** — indicador de compromiso:

| Clase        | Media | Máximo |
| ------------ | ----- | ------ |
| Cancelada    | 0,00  | 0      |
| No cancelada | 0,11  | 8      |

> **Ninguna reserva cancelada solicitó parking.** Pedir parking indica itinerario planificado y compromiso real con el viaje. Es un predictor binario muy potente a pesar de su distribución asimétrica.

**Variables categóricas:**

- `City Hotel` cancela más que `Resort Hotel` (30% vs 22%)
- `Online TA` cancela más (35%); `Corporate` menos (12%)
- `Transient` cancela más (30%); `Group` menos (9%)
- `Undefined` en `market_segment` eliminada (100% cancelación, error de registro)

---

## 6. Decisiones de Feature Engineering

| Variable                   | Decisión                           | Criterio                                      |
| -------------------------- | ---------------------------------- | --------------------------------------------- |
| `arrival_date_week_number` | ❌ Eliminada                       | Correlación perfecta con `arrival_date_month` |
| `company`                  | 🔄 Binarizada → `has_company`      | ID sin significado ordinal                    |
| `agent`                    | 🔄 Binarizada → `has_agent`        | ID sin significado ordinal                    |
| `country`                  | 🔄 Agrupada → top países + `OTHER` | 175 categorías inmanejables                   |
| `arrival_date_month`       | 🔄 Numérica (1–12)                 | Preservar estacionalidad                      |

---

## 7. Variables con mayor potencial predictivo

1. `lead_time` — diferencia clara entre clases
2. `deposit_type` — alta correlación (con riesgo documentado)
3. `total_of_special_requests` — más peticiones = más compromiso
4. `previous_cancellations` — historial como predictor de comportamiento
5. `market_segment` — tasas muy diferenciadas por canal
6. `required_car_parking_spaces` — indica planificación y compromiso

---

## 8. Implicaciones para el modelado

| Modelo              | Scaling          | Encoding |
| ------------------- | ---------------- | -------- |
| Regresión Logística | `StandardScaler` | One-Hot  |
| Árbol de Decisión   | No necesario     | Label    |
| Random Forest       | No necesario     | Label    |
| XGBoost / LightGBM  | No necesario     | Label    |
| Red Neuronal        | `MinMaxScaler`   | One-Hot  |

> ⚠️ El split train/test debe realizarse **antes** de aplicar cualquier transformación que aprenda parámetros del dataset (scaler, encoder) para evitar contaminación del test set.
