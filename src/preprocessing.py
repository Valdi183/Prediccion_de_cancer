"""
src/preprocessing.py

Centraliza la definición de features y el pipeline de preprocesado.
Se importa desde el notebook para evitar duplicar lógica entre celdas.

Decisiones de diseño:
- mut_ALK excluida: sin lift predictivo (4.92 % vs 4.89 % en cancer=1)
- ECONOMICOS descartado: coste/ingresos/dias son consecuencia del diagnóstico (leakage)
- 'vive' descartada: consecuencia del diagnóstico (leakage)
- Sociodemográficas excluidas: ausentes en el modelo generativo, añaden ruido
- 'alcohol' excluida: constante (100 % = 1), varianza cero
- enfermedad_cardiaca / asma / epoc excluidas: lift < 2 pp, fuera del modelo generativo
"""

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

TARGET = "cancer"

# Numéricas continuas → StandardScaler
FEATURES_NUM = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina", "edad",
]

# Ordinal con orden causal explícito → OrdinalEncoder
# Baja=0 · Moderada=1 · Alta=2  (factor protector creciente)
FEATURES_ORD = ["actividad_fisica"]

# Binarias (ya son 0/1) → passthrough sin escalar
FEATURES_BIN = [
    "fumador",       # peso +1.5 en modelo generativo
    "diabetes",      # comorbilidad con lift +7 pp en cancer=1
    "hipertension",  # peso +0.5 en modelo generativo, lift +10 pp
    "obesidad",      # peso +1.1 en modelo generativo, lift +19 pp
    "mut_BRCA1",     # mayor predictor genético: 5 % → 20 % en cancer=1
    "mut_TP53",
    "mut_EGFR",
    "mut_KRAS",
    "mut_PIK3CA",
    "mut_BRAF",
]

ALL_FEATURES = FEATURES_NUM + FEATURES_ORD + FEATURES_BIN  # 19 features total


def build_preprocessor() -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer sin ajustar.
    Uso: preprocessor.fit_transform(X_train) → luego .transform(X_val / X_test).
    Ajustar siempre solo en train para evitar data leakage hacia val/test.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("ord", OrdinalEncoder(categories=[["Baja", "Moderada", "Alta"]]), FEATURES_ORD),
            ("bin", "passthrough", FEATURES_BIN),
        ],
        remainder="drop",
    )
