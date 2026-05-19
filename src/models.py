"""
src/models.py

Definición y entrenamiento de los 4 modelos:
  1. Logistic Regression  — baseline interpretable
  2. Random Forest        — ensemble, importancia de features
  3. XGBoost              — estándar tabular, mejor rendimiento esperado
  4. MLP (PyTorch)        — ≥3 capas ocultas, Dropout, Early Stopping manual,
                            BCEWithLogitsLoss con pos_weight para desbalance

— Se completa en la fase 2 del notebook (celdas 06 en adelante) —
"""
