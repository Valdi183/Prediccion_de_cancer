# Predicción de Diagnóstico de Cáncer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Victor Valdivia Calatrava** · Universidad Alfonso X el Sabio — Inteligencia Artificial 2025-2026

Sistema de cribado oncológico basado en aprendizaje automático que compara cuatro clasificadores sobre un dataset clínico sintético de 50 001 pacientes.

---

## 1. Descripción del proyecto

El proyecto aborda un caso clínico ficticio en el que se dispone de registros médicos obtenidos a partir de seis colecciones MongoDB simuladas (bioquímica, clínica, genética, económica, hábitos y sociodemografía). El objetivo es determinar qué modelo — Regresión Logística, Random Forest, XGBoost o red neuronal MLP — generaliza mejor para la detección de cáncer binaria (positivo / negativo) sobre datos no vistos.

El dataset contiene 50 001 pacientes sintéticos con una prevalencia de cáncer del 19,29 % (desbalance 4,18:1). Las variables económicas y de supervivencia se excluyeron por constituir *leakage* causal: son consecuencia del diagnóstico, no predictores de él. La variable `mut_ALK` también se excluyó por ausencia de lift empírico (ratio 0,99×) pese a aparecer en la guía de features del metadata.

---

## 2. Estructura del proyecto

```
Prediccion_de_cancer/
│
├── app.py                          # Aplicación Streamlit de cribado interactivo
├── requirements.txt                # Dependencias del entorno
├── README.md
│
├── .streamlit/
│   └── config.toml                 # Tema visual de la app (dark blue)
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Análisis exploratorio: distribuciones, correlaciones, leakage
│   ├── 02_feature_engineering.ipynb# Preprocesado, split 70/15/15, StandardScaler, guardado de arrays
│   ├── 03_models.ipynb             # Entrenamiento de LR, RF, XGBoost y MLP; optimización de umbral en val
│   └── 04_cancer_diagnosis.ipynb   # Evaluación final sobre test; curvas ROC/PR; matrices de confusión
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # ColumnTransformer: StandardScaler + OrdinalEncoder + passthrough
│   ├── models.py                   # Funciones de entrenamiento de los 4 modelos
│   └── evaluate.py                 # Métricas, optimización de umbral, gráficas ROC/PR/confusión
│
├── outputs/
│   ├── processed/
│   │   ├── X_train.npy / X_val.npy / X_test.npy
│   │   ├── y_train.npy / y_val.npy / y_test.npy
│   │   ├── preprocessor.pkl        # ColumnTransformer ajustado solo sobre train
│   │   └── meta.json               # Nombres de features, tamaños de split, pos_weight
│   ├── models/
│   │   ├── lr_model.pkl
│   │   ├── rf_model.pkl
│   │   ├── xgb_model.pkl
│   │   ├── mlp_weights.pt          # Pesos del MLP (PyTorch state_dict)
│   │   ├── mlp_config.json         # Arquitectura del MLP (input_dim, hidden_dims, dropout)
│   │   └── thresholds.json         # Umbrales optimizados en validación para cada modelo
│   └── figures/
│       ├── test_metrics.csv        # Tabla de métricas finales en test
│       ├── test_roc.png            # Curvas ROC superpuestas
│       ├── test_pr.png             # Curva Precision-Recall con puntos de operación
│       ├── test_metrics_bar.png    # Gráfico de barras comparativo
│       └── test_confusion_best.png # Matriz de confusión del mejor modelo (F1)
│
└── Base de datos-20260512/         # CSVs originales de las 6 colecciones MongoDB
    ├── CASOCANCER_01_BIOQUIMICOS.csv
    ├── CASOCANCER_02_CLINICOS.csv
    ├── CASOCANCER_03_GENETICOS.csv
    ├── CASOCANCER_04_ECONOMICOS.csv
    ├── CASOCANCER_05_GENERALES.csv
    └── CASOCANCER_06_SOCIODEMOGRAFICOS.csv
```

---

## 3. Instalación y entorno

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd Prediccion_de_cancer

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota de compatibilidad crítica**
>
> `torch==2.3.0` fue compilado con la ABI de NumPy 1.x. Instalar NumPy 2.x provoca un error `AttributeError: _ARRAY_API` en tiempo de ejecución. Por eso `requirements.txt` fija `numpy==1.26.4`.
>
> En Windows, el kernel Jupyter de VS Code hereda un entorno de hilos que provoca un deadlock entre MKL y OpenMP al combinar PyTorch con scikit-learn en el mismo proceso. La solución es establecer `OMP_NUM_THREADS=1` y `MKL_NUM_THREADS=1` **antes** de cualquier import de torch, tal como se hace en la primera celda de los notebooks 03 y 04 y en `app.py`.

---

## 4. Orden de ejecución

Ejecutar los notebooks en orden desde VS Code (kernel: Python 3.10+):

1. **`01_EDA.ipynb`** — Análisis exploratorio: distribuciones, correlaciones, identificación de leakage y selección de variables.
2. **`02_feature_engineering.ipynb`** — Fusión de colecciones, split estratificado 70/15/15, ajuste del `ColumnTransformer` y guardado de arrays en `outputs/processed/`.
3. **`03_models.ipynb`** — Entrenamiento de los 4 modelos, optimización de umbral sobre validación (maximiza F1) y guardado de modelos en `outputs/models/`.
4. **`04_cancer_diagnosis.ipynb`** — Evaluación final sobre test con umbrales fijados, generación de curvas ROC/PR, gráfico de barras comparativo y matriz de confusión del mejor modelo.

Para lanzar la aplicación interactiva:

```bash
streamlit run app.py
```

La app carga automáticamente los modelos y umbrales guardados. No requiere re-entrenamiento.

---

## 5. Resultados principales

Métricas evaluadas sobre el conjunto de **test independiente** (n = 7 501 pacientes, nunca visto durante entrenamiento ni optimización de umbral).

| Modelo               | Umbral | Precisión | Recall | F1     | AUC-ROC |
|----------------------|--------|-----------|--------|--------|---------|
| Regresión Logística  | 0.620  | 0.5022    | 0.6254 | 0.5571 | 0.8310  |
| Random Forest        | 0.435  | 0.4894    | 0.6358 | 0.5531 | 0.8265  |
| XGBoost              | 0.555  | 0.4959    | 0.6275 | 0.5540 | 0.8271  |
| Red Neuronal (MLP)   | 0.650  | 0.5208    | 0.5715 | 0.5450 | 0.8330  |

La separación máxima entre AUC-ROC es de 0.006 puntos, lo que indica que los cuatro modelos capturan la misma señal subyacente y que ninguno sobreajusta. La Regresión Logística obtiene el mejor F1 en test con un Recall del 62,5 %, lo que equivale a detectar 905 de los 1 447 casos de cáncer reales presentes en el conjunto de test.

---

## 6. Decisiones técnicas destacadas

- **Exclusión de leakage causal.** Las variables económicas (`coste`, `ingresos`, `dias_hospitalizacion`) y `vive` se excluyen porque son consecuencia del diagnóstico, no predictores. Incluirlas inflaría artificialmente las métricas.
- **Split estratificado 70/15/15 con semilla fija (seed=42).** La estratificación garantiza que la prevalencia del 19,29 % se mantiene en los tres subconjuntos. El test set permanece sellado hasta la evaluación final en el notebook 04.
- **Optimización del umbral en validación, nunca en test.** Para cada modelo se barre el intervalo [0.05, 0.95] con paso 0.005 maximizando F1 sobre `X_val`. El umbral óptimo se guarda en `thresholds.json` y se aplica sin modificación sobre `X_test`.
- **Exclusión de `mut_ALK` por evidencia empírica.** La guía del metadata sugería incluir esta mutación, pero el análisis de lift mostró una diferencia de prevalencia de −0.03 pp entre pacientes con y sin cáncer (ratio 0.99×). La evidencia observada prevaleció sobre la guía teórica.
- **Tratamiento del desbalance 4,18:1.** Se utiliza `class_weight='balanced'` en Regresión Logística y Random Forest, `scale_pos_weight=4.185` en XGBoost y `pos_weight=4.185` en la función de pérdida `BCEWithLogitsLoss` del MLP. Esto evita que los modelos trivialmente predigan la clase mayoritaria.

---

## 7. Limitaciones

- **Dataset sintético con ruido gaussiano (σ = 0.8).** Los datos fueron generados por un modelo probabilístico, no recogidos en entorno clínico real. Las relaciones entre variables reflejan el diseño del generador, no necesariamente la fisiopatología oncológica.
- **Techo de AUC-ROC ~0.83 por diseño del generador.** La señal predictiva está acotada por la cantidad de información causal que el modelo generativo inyectó en los datos. Modelos más complejos no mejorarían significativamente este valor.
- **La app Streamlit es una demo, no un sistema clínico certificado.** No ha sido validada clínicamente, no cumple regulación MDR (EU 2017/745) ni FDA 510(k), y no debe utilizarse para decisiones diagnósticas reales.

---

## Presentación

[Ver diapositivas del proyecto](presentacion.html) — 6 diapositivas en HTML (portada + 5 secciones).

---

## Licencia

MIT License — Universidad Alfonso X el Sabio, 2025-2026.


