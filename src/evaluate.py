"""
src/evaluate.py

Funciones de evaluación compartidas por todos los modelos:
  - optimize_threshold : maximiza F1 sobre validación para fijar el umbral
  - eval_model         : calcula Accuracy, Precision, Recall, F1, AUC-ROC
  - plot_roc_pr        : curvas ROC y Precision-Recall comparativas
  - plot_confusion     : matrices de confusión en cuadrícula

— Se completa en la fase 2 del notebook (celdas 06 en adelante) —
"""
