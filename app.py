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
# PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="CreditIQ Kenya | SME Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AUTO-TRAIN if model doesn't exist
# ============================================================
model_path    = Path("models") / "kenya_sme_credit_model.pkl"
features_path = Path("models") / "feature_columns.pkl"

if not model_path.exists() or not features_path.exists():
    with st.spinner("⚙️ First-time setup: training model... (~15 seconds)"):
        try:
            subprocess.run([sys.executable, "setup.py"], check=True, capture_output=True)
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error("❌ Setup failed. Run `python setup.py` in your terminal.")
            st.code(e.stderr.decode() if e.stderr else "Unknown error")
            st.stop()

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] { background: #080C14 !important; color: #E2E8F0 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] > .main { background: #080C14 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #1E2D40 !important; }
[data-testid="stSidebar"] label { color: #64748B !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #111827 !important; border: 1px solid #1E2D40 !important; border-radius: 6px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #E2E8F0 !important; font-family: 'DM Mono', monospace !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] .stButton > button { width: 100% !important; background: linear-gradient(135deg, #00D4AA, #0EA5E9) !important; color: #080C14 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; border: none !important; border-radius: 8px !important; padding: 0.75rem 1.5rem !important; margin-top: 0.5rem !important; }
[data-testid="stSidebar"] .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(0,212,170,0.35) !important; }
div[data-testid="metric-container"] { background: #0D1117 !important; border: 1px solid #1E2D40 !important; border-radius: 12px !important; padding: 1.2rem !important; }
div[data-testid="metric-container"] label { color: #64748B !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1E2D40 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #64748B !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; border: none !important; padding: 0.75rem 1.5rem !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #00D4AA !important; border-bottom: 2px solid #00D4AA !important; background: transparent !important; }
[data-testid="stDataFrame"] { border: 1px solid #1E2D40 !important; border-radius: 10px !important; overflow: hidden !important; }
hr { border-color: #1E2D40 !important; }
.section-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #00D4AA; margin-bottom: 0.25rem; }
input[type="number"] { background: #111827 !important; color: #E2E8F0 !important; border: 1px solid #1E2D40 !important; border-radius: 6px !important; font-family: 'DM Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(model_path), joblib.load(features_path)

model, feature_cols = load_model()

# ============================================================
# CONSTANTS
# ============================================================
SECTOR_MAP   = {"Retail": 0, "Manufacturing": 1, "Agriculture": 2, "Services": 3, "Technology": 4}
LOCATION_MAP = {"Nairobi": 0, "Mombasa": 1, "Kisumu": 2, "Nakuru": 3, "Eldoret": 4}
TRAINING_RANGES = {
    "business_age": (1,30), "employees": (1,100), "monthly_revenue": (10000,500000),
    "monthly_expenses": (4000,650000), "profit_margin": (-50,60),
    "avg_account_balance": (1000,300000), "transaction_frequency": (1,60),
    "loan_repayment_history": (0,10), "existing_loans": (0,9), "collateral_value": (0,500000),
}

# ============================================================
# HELPERS
# ============================================================
def compute_risk_score(inp):
    score, flags = 0, []
    rph = inp["loan_repayment_history"]
    score += (10 - rph) * 3
    if rph <= 2:   flags.append(("CRITICAL", "Loan repayment history critically poor (≤2/10)"))
    elif rph <= 4: flags.append(("HIGH",     "Loan repayment history below average"))
    el = inp["existing_loans"]
    score += min(el * 2.5, 25)
    if el > 9:   flags.append(("CRITICAL", f"Existing loans ({el}) far exceeds safe threshold (≤9)"))
    elif el > 5: flags.append(("HIGH",     f"Elevated existing loan count ({el})"))
    cf = inp["monthly_revenue"] - inp["monthly_expenses"]
    if cf < 0:
        score += 20; flags.append(("CRITICAL", f"Negative cash flow (KES {cf:,.0f}/month)"))
    elif cf < inp["monthly_revenue"] * 0.1:
        score += 10; flags.append(("MEDIUM", "Very thin cash flow margin (<10% of revenue)"))
    pm = inp["profit_margin"]
    if pm < 0:    score += 15; flags.append(("HIGH", f"Negative profit margin ({pm:.1f}%)"))
    elif pm < 10: score += 5
    ab = inp["avg_account_balance"]
    if ab < 2000:    score += 15; flags.append(("HIGH", f"Critically low account balance (KES {ab:,.0f})"))
    elif ab < 10000: score += 7
    if inp["business_age"] < 2:
        score += 8; flags.append(("MEDIUM", f"Early-stage business ({inp['business_age']} year(s) in operation)"))
    loan_exposure = inp["existing_loans"] * inp["monthly_revenue"] * 3
    if loan_exposure > 0 and inp["collateral_value"] < loan_exposure * 0.5:
        score += 10; flags.append(("MEDIUM", "Collateral insufficient relative to estimated loan exposure"))
    return min(score, 100), flags

def detect_outliers(inp):
    return [f"`{c}` = **{inp[c]}** (expected {lo}–{hi})"
            for c, (lo, hi) in TRAINING_RANGES.items()
            if inp.get(c) is not None and (inp[c] < lo or inp[c] > hi)]

def risk_band(score):
    if score >= 75: return "CRITICAL RISK", "#EF4444"
    if score >= 55: return "HIGH RISK",     "#F97316"
    if score >= 35: return "MODERATE RISK", "#EAB308"
    if score >= 15: return "LOW RISK",      "#22C55E"
    return "MINIMAL RISK", "#00D4AA"

def format_kes(v):
    if v >= 1_000_000: return f"KES {v/1_000_000:.1f}M"
    if v >= 1_000:     return f"KES {v/1_000:.0f}K"
    return f"KES {v:.0f}"

def ref_id(inp):
    return "CIQ-" + hashlib.md5((json.dumps(inp,sort_keys=True)+datetime.now().isoformat()).encode()).hexdigest()[:8].upper()

# ============================================================
# SIDEBAR — navigation + inputs
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0.5rem 1rem 0.5rem; border-bottom:1px solid #1E2D40; margin-bottom:1rem;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1rem; color:#F1F5F9;">🏦 CreditIQ</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#334155; margin-top:0.2rem;">SME UNDERWRITING ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["📋 Credit Assessment", "📊 Model Performance"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E2D40; margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    run_assessment = False
    business_age = employees = monthly_revenue = monthly_expenses = 0
    profit_margin = avg_account_balance = transaction_frequency = 0.0
    loan_repayment_history = existing_loans = collateral_value = 0
    sector = list(SECTOR_MAP.keys())[0]
    location = list(LOCATION_MAP.keys())[0]

    if page == "📋 Credit Assessment":
        st.markdown('<div class="section-label">▸ Business Profile</div>', unsafe_allow_html=True)
        business_age = st.number_input("Business Age (Years)", min_value=0, max_value=100, value=5)
        employees    = st.number_input("Number of Employees",  min_value=1, max_value=500, value=10)
        sector       = st.selectbox("Sector",   list(SECTOR_MAP.keys()))
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
        For institutional use only.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SHARED HEADER
# ============================================================
titles = {
    "📋 Credit Assessment": ("Credit Risk Assessment", "ML-powered underwriting for Kenya's small &amp; medium enterprise sector"),
    "📊 Model Performance": ("Model Performance Report", "RandomForestClassifier · 2,000 samples · 5-fold cross-validation"),
}
page_title, page_sub = titles[page]

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
            {page_sub}
        </div>
    </div>
    <div style="text-align:right; font-family:'DM Mono',monospace; font-size:0.7rem; color:#334155; line-height:1.8;">
        MODEL v2.1 &nbsp;·&nbsp; RandomForest<br>
        2,000 Kenya SME records
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
if page == "📊 Model Performance":
    metrics_path = Path("models") / "model_metrics.json"
    if not metrics_path.exists():
        st.warning("⚠️ Run `python setup.py` to generate metrics.")
        st.stop()

    with open(metrics_path) as f:
        m = json.load(f)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Accuracy",          f"{m['accuracy']:.1%}")
    with k2: st.metric("ROC AUC",           f"{m['roc_auc']:.4f}")
    with k3: st.metric("CV F1",             f"{m['cv_f1_mean']:.4f}")
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
            <div style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);border-radius:10px;padding:1.2rem;text-align:center;margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00D4AA;">{cm[0][0]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#00D4AA;">TRUE NEGATIVE</div>
            </div>
            <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);border-radius:10px;padding:1.2rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#F97316;">{cm[1][0]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#F97316;">FALSE NEGATIVE</div>
            </div>""", unsafe_allow_html=True)
        with cb:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:1.2rem;text-align:center;margin-bottom:0.5rem;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#EF4444;">{cm[0][1]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#EF4444;">FALSE POSITIVE</div>
            </div>
            <div style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);border-radius:10px;padding:1.2rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00D4AA;">{cm[1][1]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#00D4AA;">TRUE POSITIVE</div>
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
        colors = ["#00D4AA","#0EA5E9","#A855F7","#F97316","#EAB308","#EF4444","#22C55E","#64748B","#EC4899","#14B8A6","#8B5CF6","#F59E0B"]
        for i, (feat, imp) in enumerate(m["feature_importance"].items()):
            pct = imp * 100
            c   = colors[i % len(colors)]
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#94A3B8;">{feat.replace('_',' ').title()}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{c};">{pct:.1f}%</span>
                </div>
                <div style="background:#1E2D40;border-radius:3px;height:5px;overflow:hidden;">
                    <div style="background:{c};width:{min(pct*5,100):.0f}%;height:100%;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    perf1, perf2 = st.columns(2)
    with perf1:
        st.markdown('<div class="section-label">No Default Class</div>', unsafe_allow_html=True)
        for lbl, val in [("Precision", f"{m['no_default_precision']:.1%}"),("Recall", f"{m['no_default_recall']:.1%}"),("F1", f"{m['no_default_f1']:.1%}"),("CV F1 Mean", f"{m['cv_f1_mean']:.4f}"),("CV F1 Std", f"±{m['cv_f1_std']:.4f}"),("Train Samples", f"{m['train_size']:,}"),("Test Samples", f"{m['test_size']:,}"),]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{lbl}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;">{val}</span></div>""", unsafe_allow_html=True)
    with perf2:
        st.markdown('<div class="section-label">Default Class</div>', unsafe_allow_html=True)
        for lbl, val in [("Precision", f"{m['default_precision']:.1%}"),("Recall", f"{m['default_recall']:.1%}"),("F1", f"{m['default_f1']:.1%}"),("ROC AUC", f"{m['roc_auc']:.4f}"),("Avg Precision", f"{m['avg_precision']:.4f}"),("Features", str(m['n_features'])),("Estimators", str(m['n_estimators'])),]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{lbl}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;">{val}</span></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.info("📄 Full interactive report: `reports/model_evaluation.html` — open in any browser.")
    st.stop()

# ============================================================
# CREDIT ASSESSMENT PAGE
# ============================================================
inputs = {
    "business_age":           business_age,
    "employees":              employees,
    "sector":                 SECTOR_MAP[sector],
    "location":               LOCATION_MAP[location],
    "monthly_revenue":        monthly_revenue,
    "monthly_expenses":       monthly_expenses,
    "profit_margin":          profit_margin,
    "avg_account_balance":    avg_account_balance,
    "transaction_frequency":  transaction_frequency,
    "loan_repayment_history": loan_repayment_history,
    "existing_loans":         existing_loans,
    "collateral_value":       collateral_value,
}

if not run_assessment:
    col1, col2, col3 = st.columns(3)
    for col, label, color, title, desc in [
        (col1, "◈ RandomForest Engine", "#00D4AA", "RandomForest Engine",
         "100-tree ensemble trained on 2,000 Kenya SME records with class-balanced sampling and stratified validation."),
        (col2, "◈ Dual-Layer Scoring",  "#0EA5E9", "Dual-Layer Scoring",
         "ML probability combined with a rule-based financial risk score for explainable, auditable credit decisions."),
        (col3, "◈ 5 Counties · 5 Sectors", "#A855F7", "5 Counties · 5 Sectors",
         "Covers Nairobi, Mombasa, Kisumu, Nakuru and Eldoret across Retail, Agriculture, Manufacturing, Services and Tech."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid #1E2D40;border-radius:12px;padding:1.5rem;">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{color};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">◈ {title}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:3rem;text-align:center;font-family:'DM Mono',monospace;font-size:0.75rem;color:#1E3A5F;letter-spacing:0.1em;">
        ← FILL IN THE SME PROFILE ON THE LEFT AND CLICK › RUN CREDIT ASSESSMENT
    </div>""", unsafe_allow_html=True)

else:
    input_df   = pd.DataFrame([inputs])
    outliers   = detect_outliers(inputs)
    rule_score, flags = compute_risk_score(inputs)
    band, _           = risk_band(rule_score)
    rid        = ref_id(inputs)
    ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    cash_flow  = monthly_revenue - monthly_expenses
    cov_ratio  = collateral_value / max(monthly_revenue * 6, 1)

    ml_pred        = model.predict(input_df)[0]
    ml_proba       = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    ml_def_prob    = float(ml_proba[1]) if ml_proba is not None else 0.5
    hard_override  = any(s == "CRITICAL" for s, _ in flags) and (cash_flow < 0 or existing_loans > 9 or loan_repayment_history <= 2)
    final_default  = 1 if (ml_pred == 1 or hard_override) else 0
    blended        = 0.7 * ml_def_prob + 0.3 * (rule_score / 100)

    if outliers:
        st.warning("⚠️ **Inputs outside training range — predictions may be less reliable.**\n\n" + "\n".join(f"- {o}" for o in outliers))

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#334155;border-bottom:1px solid #1E2D40;padding-bottom:0.75rem;margin-bottom:1.5rem;display:flex;justify-content:space-between;">
        <span>REF: <span style="color:#475569;">{rid}</span></span>
        <span>ASSESSED: <span style="color:#475569;">{ts}</span></span>
        <span>ENGINE: <span style="color:#475569;">CreditIQ v2.1</span></span>
    </div>""", unsafe_allow_html=True)

    if final_default == 1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.04));border:1px solid rgba(239,68,68,0.4);border-left:4px solid #EF4444;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#EF4444;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem;">◈ Credit Decision</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#FCA5A5;">✗ &nbsp;APPLICATION DECLINED</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:#7F1D1D;margin-top:0.3rem;">Risk profile exceeds acceptable underwriting threshold</div>
                {"<div style='font-family:DM Sans,sans-serif;font-size:0.8rem;color:#EF4444;margin-top:0.5rem;'>⚠ Hard override applied due to extreme risk indicators</div>" if hard_override else ""}
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:#EF4444;line-height:1;">{blended:.0%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#7F1D1D;text-transform:uppercase;">Default Probability</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,212,170,0.10),rgba(0,212,170,0.03));border:1px solid rgba(0,212,170,0.35);border-left:4px solid #00D4AA;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#00D4AA;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem;">◈ Credit Decision</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#6EE7B7;">✓ &nbsp;APPLICATION APPROVED</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:#064E3B;margin-top:0.3rem;">Risk profile within acceptable underwriting parameters</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:#00D4AA;line-height:1;">{blended:.0%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#064E3B;text-transform:uppercase;">Default Probability</div>
            </div>
        </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Risk Band",         band)
    with k2: st.metric("Risk Score",        f"{rule_score:.0f} / 100")
    with k3: st.metric("Monthly Cash Flow", format_kes(cash_flow), delta=f"{cash_flow/max(monthly_revenue,1)*100:.1f}% margin")
    with k4: st.metric("Collateral Cover",  f"{cov_ratio:.1f}×")
    with k5: st.metric("Repayment Score",   f"{loan_repayment_history} / 10")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["RISK BREAKDOWN", "FEATURE ANALYSIS", "AUDIT LOG"])

    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-label">Risk Flags</div>', unsafe_allow_html=True)
            if flags:
                sc = {"CRITICAL":"#EF4444","HIGH":"#F97316","MEDIUM":"#EAB308"}
                for sev, msg in sorted(flags, key=lambda x: ["CRITICAL","HIGH","MEDIUM"].index(x[0])):
                    st.markdown(f"""
                    <div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.6rem;background:#0D1117;border:1px solid #1E2D40;border-left:3px solid {sc[sev]};border-radius:8px;padding:0.7rem 1rem;">
                        <span style="font-family:'DM Mono',monospace;font-size:0.6rem;font-weight:600;color:{sc[sev]};text-transform:uppercase;white-space:nowrap;">{sev}</span>
                        <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#94A3B8;">{msg}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:#0D1117;border:1px solid #1E2D40;border-radius:8px;padding:1.5rem;text-align:center;color:#334155;font-family:'DM Mono',monospace;font-size:0.75rem;">NO RISK FLAGS DETECTED</div>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-label">Risk Score Composition</div>', unsafe_allow_html=True)
            cf = monthly_revenue - monthly_expenses
            factors = {
                "Repayment History": max(0,(10-loan_repayment_history)*3),
                "Existing Loans":    min(existing_loans*2.5,25),
                "Cash Flow Health":  20 if cf<0 else (10 if cf<monthly_revenue*0.1 else 0),
                "Profit Margin":     15 if profit_margin<0 else (5 if profit_margin<10 else 0),
                "Account Balance":   15 if avg_account_balance<2000 else (7 if avg_account_balance<10000 else 0),
                "Business Maturity": 8 if business_age<2 else 0,
            }
            for factor, fscore in sorted(factors.items(), key=lambda x: -x[1]):
                pct = fscore/100
                color = "#EF4444" if pct>0.15 else ("#F97316" if pct>0.07 else "#00D4AA")
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                        <span style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#94A3B8;">{factor}</span>
                        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{color};">{fscore:.0f} pts</span>
                    </div>
                    <div style="background:#1E2D40;border-radius:3px;height:5px;overflow:hidden;">
                        <div style="background:{color};width:{min(pct*100*1.5,100):.0f}%;height:100%;border-radius:3px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            if ml_proba is not None:
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">ML Model Confidence</div>', unsafe_allow_html=True)
                for cls, prob in zip(model.classes_, ml_proba):
                    lbl = "Default" if str(cls)=="1" else "No Default"
                    c   = "#EF4444" if str(cls)=="1" else "#00D4AA"
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                            <span style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#94A3B8;">{lbl}</span>
                            <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{c};">{prob:.1%}</span>
                        </div>
                        <div style="background:#1E2D40;border-radius:3px;height:5px;overflow:hidden;">
                            <div style="background:{c};width:{prob*100:.0f}%;height:100%;border-radius:3px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        fa1, fa2 = st.columns(2)
        with fa1:
            st.markdown('<div class="section-label">Financial Health</div>', unsafe_allow_html=True)
            cf2 = monthly_revenue - monthly_expenses
            for lbl, val, sent in [
                ("Monthly Revenue",  format_kes(monthly_revenue),  None),
                ("Monthly Expenses", format_kes(monthly_expenses), None),
                ("Net Cash Flow",    format_kes(cf2),              "pos" if cf2>=0 else "neg"),
                ("Profit Margin",    f"{profit_margin:.1f}%",      "pos" if profit_margin>=15 else "neg"),
                ("Account Balance",  format_kes(avg_account_balance), None),
                ("Collateral Value", format_kes(collateral_value), None),
                ("Collateral Cover", f"{cov_ratio:.2f}×",          "pos" if cov_ratio>=1 else "neg"),
            ]:
                c = "#00D4AA" if sent=="pos" else ("#EF4444" if sent=="neg" else "#94A3B8")
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.6rem 0;border-bottom:1px solid #0F1923;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{lbl}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:{c};font-weight:500;">{val}</span></div>""", unsafe_allow_html=True)
        with fa2:
            st.markdown('<div class="section-label">Business Profile</div>', unsafe_allow_html=True)
            for lbl, val in [("Sector",sector),("Location",location),("Business Age",f"{business_age} yrs"),("Employees",str(employees)),("Transactions",f"{transaction_frequency}/month"),("Repayment History",f"{loan_repayment_history}/10"),("Existing Loans",str(existing_loans)),]:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.6rem 0;border-bottom:1px solid #0F1923;">
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;">{lbl}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;font-weight:500;">{val}</span></div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-label">Decision Audit Trail</div>', unsafe_allow_html=True)
        for k, v in {"reference_id":rid,"timestamp":ts,"model_version":"RandomForestClassifier v2.1","ml_prediction":int(ml_pred),"ml_default_prob":f"{ml_def_prob:.4f}","rule_risk_score":f"{rule_score:.1f}/100","blended_prob":f"{blended:.4f}","hard_override":str(hard_override),"final_decision":"DECLINED" if final_default==1 else "APPROVED","risk_band":band,"flags_count":len(flags),"outliers_detected":len(outliers),}.items():
            st.markdown(f"""<div style="display:flex;gap:2rem;padding:0.45rem 0;border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#334155;min-width:200px;text-transform:uppercase;">{k}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;">{v}</span></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Raw Feature Vector</div>', unsafe_allow_html=True)
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid #0F1923;display:flex;justify-content:space-between;">
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3A5F;">CREDITIQ KENYA &nbsp;·&nbsp; SME RISK INTELLIGENCE &nbsp;·&nbsp; FOR INSTITUTIONAL USE ONLY</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3A5F;">NOT A SUBSTITUTE FOR FULL CREDIT DUE DILIGENCE</div>
</div>""", unsafe_allow_html=True)
