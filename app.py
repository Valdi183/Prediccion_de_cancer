"""
app.py — Sistema de Cribado Oncológico
Universidad Alfonso X el Sabio · IA 2025-2026

Ejecutar: streamlit run app.py
"""
import os, site as _site

# Windows: registra DLLs de torch antes de cualquier import
for _sp in _site.getsitepackages():
    _td = os.path.join(_sp, "torch", "lib")
    if os.path.isdir(_td) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_td)
        break

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pathlib, json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parent
PROC_DIR   = ROOT / "outputs" / "processed"
MODELS_DIR = ROOT / "outputs" / "models"
FIGS_DIR   = ROOT / "outputs" / "figures"

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Cribado Oncológico · UAX",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fondo degradado azul oscuro — replica la imagen de referencia */
.stApp {
    background: radial-gradient(ellipse 120% 65% at 50% -5%,
        #1B47A8 0%, #0A1830 40%, #050810 100%) !important;
    min-height: 100vh !important;
}
/* Sidebar navy oscuro */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background-color: #0D1829 !important;
    border-right: 1px solid #1A2E4A !important;
}
/* Tarjetas de métricas */
.metric-card {
    background: #0F1C35 !important;
    border: 1px solid #1E3A60 !important;
    border-radius: 10px !important;
    padding: 18px 12px !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
.metric-label {
    color: #7B9BBF !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .6px !important;
    margin-bottom: 6px !important;
}
.metric-value {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
/* Tarjetas comparativa de modelos */
.prob-card {
    border-radius: 8px !important;
    padding: 14px 8px !important;
    text-align: center !important;
    background: #0F1C35 !important;
    border: 1px solid #1E3A60 !important;
    border-left: 4px solid #1E3A60 !important;
}
/* Aviso clínico — versión oscura con ámbar */
.disclaimer {
    margin-top: 16px !important;
    padding: 10px 14px !important;
    background: rgba(255, 193, 7, 0.10) !important;
    border-left: 4px solid #FFC107 !important;
    border-radius: 4px !important;
    color: #FFD166 !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Carga de artefactos (cacheada) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos…")
def load_artifacts():
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    with open(PROC_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(PROC_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(MODELS_DIR / "thresholds.json") as f:
        thresholds = json.load(f)

    models = {}
    for key, fname in [("LR", "lr_model.pkl"), ("RF", "rf_model.pkl"), ("XGBoost", "xgb_model.pkl")]:
        with open(MODELS_DIR / fname, "rb") as f:
            models[key] = pickle.load(f)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(1)

    with open(MODELS_DIR / "mlp_config.json") as f:
        cfg = json.load(f)
    mlp = MLP(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    mlp.load_state_dict(
        torch.load(MODELS_DIR / "mlp_weights.pt", map_location="cpu", weights_only=True)
    )
    mlp.eval()
    models["MLP"] = mlp

    return preprocessor, meta, thresholds, models


# ── Predicción ────────────────────────────────────────────────────────────────
def predict_all(raw: dict, preprocessor, thresholds, models) -> dict:
    """Devuelve {nombre: (probabilidad, es_positivo)} para los 4 modelos."""
    df = pd.DataFrame([raw])
    X  = preprocessor.transform(df)   # aplica StandardScaler + OrdinalEncoder + passthrough
    out = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
        else:
            import torch
            with torch.no_grad():
                logit = model(torch.tensor(X, dtype=torch.float32))
                prob  = float(torch.sigmoid(logit).item())
        out[name] = (prob, prob >= thresholds[name])
    return out


# ── Gauge ─────────────────────────────────────────────────────────────────────
def make_gauge(prob: float, threshold: float) -> go.Figure:
    pct   = prob * 100
    t_pct = threshold * 100
    # Zonas fijas: 0-40 verde, 40-70 naranja, 70-100 rojo
    if pct >= 70:
        bar_color = "#DC3545"
    elif pct >= 40:
        bar_color = "#FD7E14"
    else:
        bar_color = "#28A745"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": "#FFFFFF"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "ticksuffix": "%",
                "tickwidth": 1,
                "tickcolor": "#3A5A7A",
                "tickfont": {"color": "#7B9BBF"},
            },
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "#0A1628",
            "bordercolor": "#1E3A60",
            "steps": [
                {"range": [0,   40], "color": "#0A2216"},   # verde oscuro
                {"range": [40,  70], "color": "#221408"},   # naranja oscuro
                {"range": [70, 100], "color": "#220A0E"},   # rojo oscuro
            ],
            "threshold": {
                "line": {"color": "#4F8EF7", "width": 3},
                "thickness": 0.8,
                "value": t_pct,
            },
        },
    ))
    fig.update_layout(
        height=270,
        margin={"l": 20, "r": 20, "t": 40, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial, sans-serif", "color": "#E8EDF5"},
    )
    return fig


# ── Badge de riesgo ───────────────────────────────────────────────────────────
def risk_badge(prob: float) -> str:
    pct = prob * 100
    if pct >= 70:
        bg, text = "#DC3545", "ALTO RIESGO"
    elif pct >= 40:
        bg, text = "#FD7E14", "RIESGO MODERADO"
    else:
        bg, text = "#28A745", "BAJO RIESGO"
    return (
        f'<div style="display:inline-block;background:{bg};color:#FFFFFF;'
        f'padding:10px 28px;border-radius:8px;font-size:17px;font-weight:700;'
        f'letter-spacing:1.5px;margin-top:4px">{text}</div>'
    )


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
preprocessor, meta, thresholds, models = load_artifacts()
MODEL_LABEL = {
    "LR": "Regresión Logística",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "MLP": "Red Neuronal (MLP)",
}

# ── SIDEBAR — Formulario del paciente ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Datos del Paciente")

    st.markdown("**Parámetros bioquímicos**")
    glucosa       = st.slider("Glucosa (mg/dL)",        60,   300,  100)
    colesterol    = st.slider("Colesterol (mg/dL)",    100,   350,  180)
    trigliceridos = st.slider("Triglicéridos (mg/dL)",  50,   500,  150)
    hemoglobina   = st.slider("Hemoglobina (g/dL)",    7.0,  20.0, 13.5, step=0.1)
    leucocitos    = st.slider("Leucocitos (×10³/μL)",  2.0,  20.0,  6.5, step=0.1)
    plaquetas     = st.slider("Plaquetas (×10³/μL)",    80,   500,  250)
    creatinina    = st.slider("Creatinina (mg/dL)",    0.4,   5.0,  1.0, step=0.1)

    st.markdown("**Datos generales**")
    edad             = st.slider("Edad (años)", 18, 90, 50)
    actividad_fisica = st.selectbox("Actividad física", ["Baja", "Moderada", "Alta"], index=1)

    st.markdown("**Comorbilidades y hábitos**")
    fumador      = int(st.checkbox("Fumador"))
    diabetes     = int(st.checkbox("Diabetes"))
    hipertension = int(st.checkbox("Hipertensión"))
    obesidad     = int(st.checkbox("Obesidad"))

    st.markdown("**Mutaciones genéticas**")
    col_a, col_b = st.columns(2)
    with col_a:
        mut_BRCA1  = int(st.checkbox("BRCA1"))
        mut_EGFR   = int(st.checkbox("EGFR"))
        mut_PIK3CA = int(st.checkbox("PIK3CA"))
    with col_b:
        mut_TP53   = int(st.checkbox("TP53"))
        mut_KRAS   = int(st.checkbox("KRAS"))
        mut_BRAF   = int(st.checkbox("BRAF"))

# ── MAIN — Header ─────────────────────────────────────────────────────────────
st.markdown("## 🔬 Sistema de Cribado Oncológico")
st.caption("Universidad Alfonso X el Sabio · Inteligencia Artificial 2025-2026")
st.divider()

# Construir inputs en el orden exacto que espera el preprocessor
raw = {
    "glucosa": glucosa, "colesterol": colesterol, "trigliceridos": trigliceridos,
    "hemoglobina": hemoglobina, "leucocitos": leucocitos, "plaquetas": plaquetas,
    "creatinina": creatinina, "edad": edad,
    "actividad_fisica": actividad_fisica,   # string para OrdinalEncoder
    "fumador": fumador, "diabetes": diabetes, "hipertension": hipertension,
    "obesidad": obesidad,
    "mut_BRCA1": mut_BRCA1, "mut_TP53": mut_TP53, "mut_EGFR": mut_EGFR,
    "mut_KRAS": mut_KRAS, "mut_PIK3CA": mut_PIK3CA, "mut_BRAF": mut_BRAF,
}

results = predict_all(raw, preprocessor, thresholds, models)
primary_prob, primary_pos = results["LR"]

# ── MAIN — Dos columnas ────────────────────────────────────────────────────────
col_pred, col_perf = st.columns([6, 4], gap="large")

# ── Columna izquierda: Evaluación del paciente ────────────────────────────────
with col_pred:
    st.subheader("Evaluación del paciente")
    st.caption("Modelo principal: Regresión Logística — mejor F1 en test")

    st.plotly_chart(
        make_gauge(primary_prob, thresholds["LR"]),
        use_container_width=True,
    )

    st.markdown(risk_badge(primary_prob), unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#7B9BBF;font-size:13px;margin-top:8px'>"
        f"Probabilidad estimada: <b style='color:#E8EDF5'>{primary_prob*100:.1f}%</b>"
        f"&nbsp;·&nbsp;Umbral clínico: <b style='color:#E8EDF5'>{thresholds['LR']*100:.1f}%</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='disclaimer'>"
        "<b>⚠️ Herramienta de apoyo a la decisión clínica.</b> "
        "No sustituye el diagnóstico médico. El umbral puede ser ajustado "
        "por el oncólogo según el balance riesgo-beneficio de cada caso: "
        "reducirlo aumenta la sensibilidad a costa de más falsos positivos."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:28px;color:#E8EDF5;font-weight:600'>Comparativa de modelos</div>", unsafe_allow_html=True)
    prob_cols = st.columns(4)
    for c, (name, (prob, is_pos)) in zip(prob_cols, results.items()):
        accent = "#DC3545" if is_pos else "#28A745"
        label  = "▲ POSITIVO" if is_pos else "▼ NEGATIVO"
        c.markdown(
            f"<div class='prob-card' style='border-left-color:{accent}'>"
            f"<div style='font-size:11px;color:#7B9BBF;margin-bottom:6px'>{MODEL_LABEL[name]}</div>"
            f"<div style='font-size:26px;font-weight:700;color:#FFFFFF'>{prob*100:.1f}%</div>"
            f"<div style='font-size:11px;font-weight:600;color:{accent};margin-top:6px'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Columna derecha: Rendimiento en test ──────────────────────────────────────
with col_perf:
    st.subheader("Rendimiento en test")

    model_sel = st.selectbox(
        "Modelo",
        options=list(MODEL_LABEL.keys()),
        format_func=lambda k: MODEL_LABEL[k],
    )

    metrics_path = FIGS_DIR / "test_metrics.csv"
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path, index_col=0)
        row  = df_m.loc[model_sel]

        r1c1, r1c2 = st.columns(2)
        r1c1.markdown(
            f"<div class='metric-card'><div class='metric-label'>Precisión</div>"
            f"<div class='metric-value'>{row['Precision']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        r1c2.markdown(
            f"<div class='metric-card'><div class='metric-label'>Recall</div>"
            f"<div class='metric-value'>{row['Recall']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        r2c1, r2c2 = st.columns(2)
        r2c1.markdown(
            f"<div class='metric-card'><div class='metric-label'>F1-Score</div>"
            f"<div class='metric-value'>{row['F1']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        r2c2.markdown(
            f"<div class='metric-card'><div class='metric-label'>AUC-ROC</div>"
            f"<div class='metric-value'>{row['AUC-ROC']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Test set · n = 7,501 pacientes · umbral = {row['Umbral']:.3f}"
        )

    roc_path = FIGS_DIR / "test_roc.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="Curvas ROC — Test set (todos los modelos)", use_container_width=True)
    else:
        st.info("Ejecuta el notebook 04 para generar las curvas ROC.")
