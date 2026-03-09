import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import hashlib
import subprocess
import sys

# ============================================================
# AUTO-TRAIN if model doesn't exist
# ============================================================
model_path    = Path("models") / "kenya_sme_credit_model.pkl"
features_path = Path("models") / "feature_columns.pkl"

if not model_path.exists() or not features_path.exists():
    with st.spinner("⚙️ First-time setup: training model on your machine... (takes ~15 seconds)"):
        try:
            subprocess.run([sys.executable, "setup.py"], check=True, capture_output=True)
            st.success("✅ Model trained successfully! Loading app...")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error("❌ Setup failed. Please run `python setup.py` in your terminal.")
            st.code(e.stderr.decode() if e.stderr else "Unknown error")
            st.stop()

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="CreditIQ Kenya | SME Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS — enterprise dark theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #080C14 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D40 !important;
}
[data-testid="stSidebar"] label {
    color: #64748B !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #E2E8F0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00D4AA, #0EA5E9) !important;
    color: #080C14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0,212,170,0.35) !important;
}
div[data-testid="metric-container"] {
    background: #0D1117 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}
div[data-testid="metric-container"] label {
    color: #64748B !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
}
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E2D40 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #00D4AA !important;
    border-bottom: 2px solid #00D4AA !important;
    background: transparent !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
hr { border-color: #1E2D40 !important; }
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00D4AA;
    margin-bottom: 0.25rem;
}
input[type="number"] {
    background: #111827 !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model        = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols

model, feature_cols = load_model()

# ============================================================
# CONSTANTS
# ============================================================
SECTOR_MAP   = {"Retail": 0, "Manufacturing": 1, "Agriculture": 2, "Services": 3, "Technology": 4}
LOCATION_MAP = {"Nairobi": 0, "Mombasa": 1, "Kisumu": 2, "Nakuru": 3, "Eldoret": 4}

TRAINING_RANGES = {
    "business_age":           (1, 30),
    "employees":              (1, 100),
    "monthly_revenue":        (10_000, 500_000),
    "monthly_expenses":       (4_000,  650_000),
    "profit_margin":          (-50, 60),
    "avg_account_balance":    (1_000, 300_000),
    "transaction_frequency":  (1, 60),
    "loan_repayment_history": (0, 10),
    "existing_loans":         (0, 9),
    "collateral_value":       (0, 500_000),
}

# ============================================================
# HELPERS
# ============================================================
def compute_risk_score(inputs):
    score = 0
    flags = []

    rph = inputs["loan_repayment_history"]
    score += (10 - rph) * 3
    if rph <= 2:   flags.append(("CRITICAL", "Loan repayment history critically poor (≤2/10)"))
    elif rph <= 4: flags.append(("HIGH",     "Loan repayment history below average"))

    el = inputs["existing_loans"]
    score += min(el * 2.5, 25)
    if el > 9:   flags.append(("CRITICAL", f"Existing loans ({el}) far exceeds safe threshold (≤9)"))
    elif el > 5: flags.append(("HIGH",     f"Elevated existing loan count ({el})"))

    cf = inputs["monthly_revenue"] - inputs["monthly_expenses"]
    if cf < 0:
        score += 20
        flags.append(("CRITICAL", f"Negative cash flow (KES {cf:,.0f}/month)"))
    elif cf < inputs["monthly_revenue"] * 0.1:
        score += 10
        flags.append(("MEDIUM", "Very thin cash flow margin (<10% of revenue)"))

    pm = inputs["profit_margin"]
    if pm < 0:    score += 15; flags.append(("HIGH",   f"Negative profit margin ({pm:.1f}%)"))
    elif pm < 10: score += 5

    ab = inputs["avg_account_balance"]
    if ab < 2000:    score += 15; flags.append(("HIGH",   f"Critically low account balance (KES {ab:,.0f})"))
    elif ab < 10000: score += 7

    if inputs["business_age"] < 2:
        score += 8
        flags.append(("MEDIUM", f"Early-stage business ({inputs['business_age']} year(s) operating)"))

    loan_exposure = inputs["existing_loans"] * inputs["monthly_revenue"] * 3
    if loan_exposure > 0 and inputs["collateral_value"] < loan_exposure * 0.5:
        score += 10
        flags.append(("MEDIUM", "Collateral insufficient relative to estimated loan exposure"))

    return min(score, 100), flags


def detect_outliers(inputs):
    out = []
    for col, (lo, hi) in TRAINING_RANGES.items():
        v = inputs.get(col)
        if v is not None and (v < lo or v > hi):
            out.append(f"`{col}` = **{v}** (expected {lo}–{hi})")
    return out


def risk_band(score):
    if score >= 75: return "CRITICAL RISK",  "#EF4444"
    if score >= 55: return "HIGH RISK",       "#F97316"
    if score >= 35: return "MODERATE RISK",   "#EAB308"
    if score >= 15: return "LOW RISK",        "#22C55E"
    return              "MINIMAL RISK",   "#00D4AA"


def generate_ref_id(inputs):
    raw = json.dumps(inputs, sort_keys=True) + datetime.now().isoformat()
    return "CIQ-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()


def format_kes(v):
    if v >= 1_000_000: return f"KES {v/1_000_000:.1f}M"
    if v >= 1_000:     return f"KES {v/1_000:.0f}K"
    return f"KES {v:.0f}"



# ============================================================
# PAGE NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0.5rem 1rem 0.5rem; border-bottom:1px solid #1E2D40; margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1rem; color:#F1F5F9;">🏦 CreditIQ</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#334155; margin-top:0.2rem;">SME UNDERWRITING ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Credit Assessment", "Model Performance"],
                    label_visibility="collapsed")
    st.markdown("<hr style='border-color:#1E2D40; margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

# ============================================================
# SHARED HEADER
# ============================================================
page_title    = "Credit Risk Assessment" if page == "Credit Assessment" else "Model Performance Report"
page_subtitle = ("ML-powered underwriting for Kenya's small &amp; medium enterprise sector"
                 if page == "Credit Assessment"
                 else "RandomForestClassifier · 2,000 samples · 5-fold cross-validation")

st.markdown(f"""
<div style="padding:2.5rem 0 1.5rem 0; border-bottom:1px solid #1E2D40; margin-bottom:2rem;
            display:flex; align-items:flex-end; justify-content:space-between;">
    <div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                    text-transform:uppercase; color:#00D4AA; margin-bottom:0.4rem;">
            ◈ CreditIQ Kenya &nbsp;·&nbsp; SME Risk Intelligence Platform
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#F1F5F9; line-height:1.1;">
            {page_title}
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#475569; margin-top:0.4rem;">
            {page_subtitle}
        </div>
    </div>
    <div style="text-align:right; font-family:'DM Mono',monospace; font-size:0.7rem; color:#334155;">
        <div>MODEL v2.1</div>
        <div style="color:#1E2D40;">──────────</div>
        <div>RandomForest · 2000 SMEs</div>
        <div>Kenya Financial Data</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
if page == "Model Performance":
    metrics_path = Path("models") / "model_metrics.json"
    if not metrics_path.exists():
        st.warning("⚠️ Metrics not found. Run `python setup.py` first.")
        st.stop()

    with open(metrics_path) as f:
        m = json.load(f)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Accuracy",          f"{m['accuracy']:.1%}")
    with k2: st.metric("ROC AUC",           f"{m['roc_auc']:.4f}")
    with k3: st.metric("CV F1 Score",       f"{m['cv_f1_mean']:.4f}")
    with k4: st.metric("Default Recall",    f"{m['default_recall']:.1%}")
    with k5: st.metric("Default Precision", f"{m['default_precision']:.1%}")
    with k6: st.metric("Avg Precision",     f"{m['avg_precision']:.4f}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">ROC Curve</div>', unsafe_allow_html=True)
        roc_df = pd.DataFrame({"FPR": m["fpr"], "TPR": m["tpr"]}).set_index("FPR")
        st.line_chart(roc_df, color="#00D4AA")
        st.caption(f"AUC = {m['roc_auc']:.4f} — closer to 1.0 is better. Random = 0.5")
    with col2:
        st.markdown('<div class="section-label">Precision-Recall Curve</div>', unsafe_allow_html=True)
        pr_df = pd.DataFrame({"Recall": m["recall_curve"], "Precision": m["precision_curve"]}).set_index("Recall")
        st.line_chart(pr_df, color="#0EA5E9")
        st.caption(f"Average Precision = {m['avg_precision']:.4f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = m["confusion_matrix"]
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"""
            <div style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);
                        border-radius:10px;padding:1.2rem;text-align:center;margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00D4AA;">{cm[0][0]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#00D4AA;">TRUE NEGATIVE</div>
            </div>
            <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);
                        border-radius:10px;padding:1.2rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#F97316;">{cm[1][0]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#F97316;">FALSE NEGATIVE</div>
            </div>
            """, unsafe_allow_html=True)
        with cb:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                        border-radius:10px;padding:1.2rem;text-align:center;margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#EF4444;">{cm[0][1]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#EF4444;">FALSE POSITIVE</div>
            </div>
            <div style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);
                        border-radius:10px;padding:1.2rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00D4AA;">{cm[1][1]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#00D4AA;">TRUE POSITIVE</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        colors = ["#00D4AA","#0EA5E9","#A855F7","#F97316","#EAB308",
                  "#EF4444","#22C55E","#64748B","#EC4899","#14B8A6","#8B5CF6","#F59E0B"]
        for i, (feat, imp) in enumerate(m["feature_importance"].items()):
            pct = imp * 100
            c   = colors[i % len(colors)]
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#94A3B8;">
                        {feat.replace('_',' ').title()}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{c};">{pct:.1f}%</span>
                </div>
                <div style="background:#1E2D40;border-radius:3px;height:5px;overflow:hidden;">
                    <div style="background:{c};width:{min(pct*5,100):.0f}%;height:100%;border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    perf1, perf2 = st.columns(2)
    with perf1:
        st.markdown('<div class="section-label">No Default Class</div>', unsafe_allow_html=True)
        for label, val in [
            ("Precision", f"{m['no_default_precision']:.1%}"),
            ("Recall",    f"{m['no_default_recall']:.1%}"),
            ("F1 Score",  f"{m['no_default_f1']:.1%}"),
        ]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;">{val}</span></div>
            """, unsafe_allow_html=True)
    with perf2:
        st.markdown('<div class="section-label">Default Class</div>', unsafe_allow_html=True)
        for label, val in [
            ("Precision", f"{m['default_precision']:.1%}"),
            ("Recall",    f"{m['default_recall']:.1%}"),
            ("F1 Score",  f"{m['default_f1']:.1%}"),
        ]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;">{val}</span></div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.info("📄 Full interactive report: `reports/model_evaluation.html` — open in any browser.")
    st.stop()


# ============================================================
# CREDIT ASSESSMENT — SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.markdown('<div class="section-label">▸ Business Profile</div>', unsafe_allow_html=True)
    business_age = st.number_input("Business Age (Years)", min_value=0, max_value=100, value=5)
    employees    = st.number_input("Number of Employees", min_value=1, max_value=500, value=10)
    sector       = st.selectbox("Sector", list(SECTOR_MAP.keys()))
    location     = st.selectbox("Location", list(LOCATION_MAP.keys()))

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">▸ Financial Metrics</div>', unsafe_allow_html=True)
    monthly_revenue       = st.number_input("Monthly Revenue (KES)",  min_value=0, max_value=10_000_000, value=150_000, step=5000)
    monthly_expenses      = st.number_input("Monthly Expenses (KES)", min_value=0, max_value=10_000_000, value=90_000,  step=5000)
    profit_margin         = st.slider("Profit Margin (%)", -50.0, 100.0, 20.0, step=0.5)
    avg_account_balance   = st.number_input("Avg Bank Balance (KES)", min_value=0, max_value=10_000_000, value=50_000,  step=1000)
    transaction_frequency = st.number_input("Monthly Transactions",   min_value=0, max_value=200, value=15)

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">▸ Credit History</div>', unsafe_allow_html=True)
    loan_repayment_history = st.slider("Repayment History (0=Poor · 10=Excellent)", 0, 10, 7)
    existing_loans         = st.number_input("Existing Loan Count", min_value=0, max_value=20, value=1,
                                              help="Values above 9 are flagged as high risk")
    collateral_value       = st.number_input("Collateral Value (KES)", min_value=0, max_value=10_000_000, value=200_000, step=5000)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    run_assessment = st.button("⚡ Run Credit Assessment", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #1E2D40;
                font-family:'DM Mono',monospace; font-size:0.6rem; color:#1E3A5F; line-height:1.8;">
        MODEL: RandomForestClassifier<br>
        TRAINED ON: 2,000 Kenya SMEs<br>
        FEATURES: 12 financial indicators<br>
        ─────────────────────<br>
        For institutional use only.<br>
        Not a substitute for full due diligence.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top:4rem; padding-top:1.5rem; border-top:1px solid #0F1923;
            display:flex; justify-content:space-between;">
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#1E3A5F;">
        CREDITIQ KENYA &nbsp;·&nbsp; SME RISK INTELLIGENCE &nbsp;·&nbsp; FOR INSTITUTIONAL USE ONLY
    </div>
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#1E3A5F;">
        NOT A SUBSTITUTE FOR FULL CREDIT DUE DILIGENCE
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE PAGE (always visible via nav)
# ============================================================
def show_model_performance():
    import json
    metrics_path = Path("models") / "model_metrics.json"
    if not metrics_path.exists():
        st.warning("Run `python setup.py` to generate model metrics.")
        return

    with open(metrics_path) as f:
        m = json.load(f)

    st.markdown("""
    <div style="padding:2.5rem 0 1.5rem 0; border-bottom:1px solid #1E2D40; margin-bottom:2rem;">
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                    text-transform:uppercase; color:#00D4AA; margin-bottom:0.4rem;">
            ◈ CreditIQ Kenya · Model Evaluation
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#F1F5F9;">
            Model Performance Report
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#475569; margin-top:0.4rem;">
            RandomForestClassifier · Kenya SME Credit Scoring · 2,000 samples · 5-fold CV
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Accuracy",         f"{m['accuracy']:.1%}")
    with k2: st.metric("ROC AUC",          f"{m['roc_auc']:.4f}")
    with k3: st.metric("CV F1 Score",      f"{m['cv_f1_mean']:.4f}")
    with k4: st.metric("Default Recall",   f"{m['default_recall']:.1%}")
    with k5: st.metric("Default Precision",f"{m['default_precision']:.1%}")
    with k6: st.metric("Avg Precision",    f"{m['avg_precision']:.4f}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ROC + PR charts using st.line_chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">ROC Curve</div>', unsafe_allow_html=True)
        roc_df = pd.DataFrame({"False Positive Rate": m["fpr"], "True Positive Rate": m["tpr"]})
        st.line_chart(roc_df.set_index("False Positive Rate"), color="#00D4AA")
        st.caption(f"AUC = {m['roc_auc']:.4f} — closer to 1.0 is better")

    with col2:
        st.markdown('<div class="section-label">Precision-Recall Curve</div>', unsafe_allow_html=True)
        pr_df = pd.DataFrame({"Recall": m["recall_curve"], "Precision": m["precision_curve"]})
        st.line_chart(pr_df.set_index("Recall"), color="#0EA5E9")
        st.caption(f"Average Precision = {m['avg_precision']:.4f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = m["confusion_matrix"]
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background:rgba(0,212,170,0.12); border:1px solid rgba(0,212,170,0.3);
                        border-radius:10px; padding:1.2rem; text-align:center; margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#00D4AA;">{cm[0][0]}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#00D4AA; text-transform:uppercase;">True Negative</div>
            </div>
            <div style="background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.2);
                        border-radius:10px; padding:1.2rem; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#F97316;">{cm[1][0]}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#F97316; text-transform:uppercase;">False Negative</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2);
                        border-radius:10px; padding:1.2rem; text-align:center; margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#EF4444;">{cm[0][1]}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#EF4444; text-transform:uppercase;">False Positive</div>
            </div>
            <div style="background:rgba(0,212,170,0.12); border:1px solid rgba(0,212,170,0.3);
                        border-radius:10px; padding:1.2rem; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#00D4AA;">{cm[1][1]}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#00D4AA; text-transform:uppercase;">True Positive</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        colors = ["#00D4AA","#0EA5E9","#A855F7","#F97316","#EAB308","#EF4444",
                  "#22C55E","#64748B","#EC4899","#14B8A6","#8B5CF6","#F59E0B"]
        for i, (feat, imp) in enumerate(m["feature_importance"].items()):
            pct = imp * 100
            c   = colors[i % len(colors)]
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                    <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#94A3B8;">
                        {feat.replace('_',' ').title()}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:{c};">{pct:.1f}%</span>
                </div>
                <div style="background:#1E2D40; border-radius:3px; height:5px; overflow:hidden;">
                    <div style="background:{c}; width:{min(pct*5,100):.0f}%; height:100%; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Link to full HTML report
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.info("📄 Full HTML report available at `reports/model_evaluation.html` — open it in any browser for the complete interactive evaluation.")
