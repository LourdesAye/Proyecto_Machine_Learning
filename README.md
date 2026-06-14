# 🏦 Proyecto Integrador: Predicción de Aprobación de Préstamos (Loan Prediction)

## 📋 Descripción del Proyecto
Este proyecto aborda un problema de **Clasificación Binaria** en el sector financiero. El objetivo principal es construir un modelo predictivo de Machine Learning capaz de determinar si una solicitud de préstamo será aprobada o rechazada, basándose en características sociodemográficas y financieras del cliente. 

## 🗂️ Dataset
El conjunto de datos utilizado es el **Loan Prediction Problem Dataset**
* **Origen:** [Kaggle - Loan Prediction](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
* **Dimensiones iniciales:** 614 registros y 13 columnas.
* **Variable Objetivo (Target):** `Loan_Status` (Mapeada a 1 = Aprobado, 0 = Rechazado).
* **Variables Predictoras:** Incluyen género, estado civil, nivel educativo, ingresos del solicitante y co-solicitante, monto solicitado, historial crediticio y zona de la propiedad.

## 🛠️ Tecnologías y Librerías Utilizadas
* **Lenguaje:** Python.
* **Manipulación y Análisis de Datos:** Pandas, NumPy.
* **Visualización:** Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn. 
  * `Pipeline`, `ColumnTransformer` para automatización del flujo.
  * `SimpleImputer`, `StandardScaler`, `OneHotEncoder` para preprocesamiento.
  * `LogisticRegression` para el modelo predictivo.
  * `accuracy_score`, `classification_report`, `ConfusionMatrixDisplay` para evaluación.

## ⚙️ Metodología y Flujo de Trabajo
1. **Análisis Exploratorio de Datos (EDA):** Se analizó la distribución de las variables (detectando un fuerte sesgo positivo en los ingresos), se identificaron valores nulos y atípicos (outliers), y se evaluaron las correlaciones numéricas mediante un mapa de calor.
2. **Feature Engineering y Limpieza:** 
   * Se eliminaron variables sin valor predictivo (ej. `Loan_ID`).
   * Se creó una nueva característica, `Ingreso_Total`, sumando los ingresos del solicitante principal y el co-solicitante, logrando así una mejor representación de la capacidad de pago.
3. **Preprocesamiento Automatizado (Pipelines):** Se diseñó un `ColumnTransformer` para evitar fugas de datos (*data leakage*).
   * *Variables Numéricas:* Imputación de nulos por la media y estandarización de escalas.
   * *Variables Categóricas:* Imputación de nulos por la moda y codificación binaria sin orden jerárquico (*One-Hot Encoding*).
4. **Modelado:** División del dataset (80% entrenamiento, 20% prueba) preservando la distribución de clases mediante `stratify`, y entrenamiento de un modelo *baseline* de **Regresión Logística**.
5. **Evaluación del Modelo:** Medición del desempeño sobre datos no vistos utilizando Accuracy, Precision, Recall, F1-Score y Matriz de Confusión.

## 📊 Resultados Destacados
* **Rendimiento General:** El modelo de Regresión Logística alcanzó un **Accuracy aproximado del 85%**.
* **Hallazgo de Negocio:** La matriz de correlación demostró que el **historial crediticio** (`Credit_History`) es el factor con mayor peso para la aprobación de un préstamo.
* **Desafíos Detectados:** El análisis de métricas (F1-score y Matriz de Confusión) evidenció que el modelo logra un excelente desempeño sobre la clase mayoritaria (préstamos aprobados), pero presenta dificultades en identificar la clase minoritaria (rechazados) debido a un desbalance moderado en los datos originales.

## 👩‍💻 Autora
**Lourdes Ayelén González**
*(Proyecto desarrollado para el trayecto formativo Talento Tech - Machine Learning)*.
