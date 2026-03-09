import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import hashlib

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

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080C14 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D40 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] label {
    color: #64748B !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] .stSelectbox select {
    background: #111827 !important;
    border: 1px solid #1E2D40 !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] input[type="number"]:focus {
    border-color: #00D4AA !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* ── Selectbox ── */
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

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #00D4AA !important;
    border-color: #00D4AA !important;
}
[data-testid="stSlider"] div[data-testid="stTickBar"] { display: none; }

/* ── Button ── */
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
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0,212,170,0.35) !important;
}

/* ── Metric cards ── */
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

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #64748B !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: #1E2D40 !important; }

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E2D40 !important;
    gap: 0 !important;
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

/* ── Section headers ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00D4AA;
    margin-bottom: 0.25rem;
}

/* ── Info boxes ── */
[data-testid="stInfo"] {
    background: rgba(14,165,233,0.08) !important;
    border: 1px solid rgba(14,165,233,0.25) !important;
    border-radius: 8px !important;
    color: #94C6FF !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSuccess"] {
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid rgba(0,212,170,0.25) !important;
    border-radius: 8px !important;
    color: #00D4AA !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00D4AA, #0EA5E9) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: #1E2D40 !important;
    border-radius: 4px !important;
}

/* ── Number input ── */
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
model_path    = Path("models") / "kenya_sme_credit_model.pkl"
features_path = Path("models") / "feature_columns.pkl"

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

RISK_WEIGHTS = {
    "loan_repayment_history": 30,
    "existing_loans":         20,
    "profit_margin":          15,
    "avg_account_balance":    15,
    "collateral_value":       10,
    "business_age":           5,
    "transaction_frequency":  5,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_risk_score(inputs: dict) -> tuple[float, list]:
    """
    Rule-based risk score (0–100) independent of ML model.
    Returns (score, list_of_flags).
    """
    score = 0
    flags = []

    # Repayment history (lower = worse)
    rph = inputs["loan_repayment_history"]
    score += (10 - rph) * 3
    if rph <= 2: flags.append(("CRITICAL", "Loan repayment history critically poor (≤2/10)"))
    elif rph <= 4: flags.append(("HIGH", "Loan repayment history below average"))

    # Existing loans
    el = inputs["existing_loans"]
    score += min(el * 2.5, 25)
    if el > 9:  flags.append(("CRITICAL", f"Existing loans ({el}) far exceeds safe threshold (≤9)"))
    elif el > 5: flags.append(("HIGH", f"Elevated existing loan count ({el})"))

    # Cash flow
    cf = inputs["monthly_revenue"] - inputs["monthly_expenses"]
    if cf < 0:
        score += 20
        flags.append(("CRITICAL", f"Negative cash flow (KES {cf:,.0f}/month)"))
    elif cf < inputs["monthly_revenue"] * 0.1:
        score += 10
        flags.append(("MEDIUM", "Very thin cash flow margin (<10% of revenue)"))

    # Profit margin
    pm = inputs["profit_margin"]
    if pm < 0:   score += 15; flags.append(("HIGH", f"Negative profit margin ({pm:.1f}%)"))
    elif pm < 10: score += 5

    # Account balance
    ab = inputs["avg_account_balance"]
    if ab < 2000:   score += 15; flags.append(("HIGH", f"Critically low account balance (KES {ab:,.0f})"))
    elif ab < 10000: score += 7

    # Collateral coverage ratio
    loan_exposure = inputs["existing_loans"] * inputs["monthly_revenue"] * 3
    if loan_exposure > 0 and inputs["collateral_value"] < loan_exposure * 0.5:
        score += 10
        flags.append(("MEDIUM", "Collateral insufficient relative to estimated loan exposure"))

    # Young business
    if inputs["business_age"] < 2:
        score += 8
        flags.append(("MEDIUM", f"Early-stage business ({inputs['business_age']} year(s) operating)"))

    return min(score, 100), flags


def detect_outliers(inputs: dict) -> list:
    out = []
    for col, (lo, hi) in TRAINING_RANGES.items():
        v = inputs.get(col)
        if v is not None and (v < lo or v > hi):
            out.append(f"`{col}` = **{v}** (expected {lo}–{hi})")
    return out


def risk_band(score: float) -> tuple[str, str, str]:
    """Returns (band_label, color_hex, emoji)"""
    if score >= 75: return "CRITICAL RISK",  "#EF4444", "🔴"
    if score >= 55: return "HIGH RISK",       "#F97316", "🟠"
    if score >= 35: return "MODERATE RISK",   "#EAB308", "🟡"
    if score >= 15: return "LOW RISK",        "#22C55E", "🟢"
    return              "MINIMAL RISK",   "#00D4AA", "✅"


def generate_ref_id(inputs: dict) -> str:
    raw = json.dumps(inputs, sort_keys=True) + datetime.now().isoformat()
    return "CIQ-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()


def format_kes(v: float) -> str:
    if v >= 1_000_000: return f"KES {v/1_000_000:.1f}M"
    if v >= 1_000:     return f"KES {v/1_000:.0f}K"
    return f"KES {v:.0f}"


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1E2D40;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
">
    <div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                    text-transform:uppercase; color:#00D4AA; margin-bottom:0.4rem;">
            ◈ CreditIQ Kenya &nbsp;·&nbsp; SME Risk Intelligence Platform
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
                    color:#F1F5F9; line-height:1.1;">
            Credit Risk Assessment
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#475569; margin-top:0.4rem;">
            ML-powered underwriting for Kenya's small &amp; medium enterprise sector
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
# SIDEBAR — INPUT FORM
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 0.5rem 1rem 0.5rem; border-bottom: 1px solid #1E2D40; margin-bottom: 1rem;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1rem; color:#F1F5F9;">
            🏦 CreditIQ
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#334155; margin-top:0.2rem;">
            SME UNDERWRITING ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">▸ Business Profile</div>', unsafe_allow_html=True)
    business_age = st.number_input("Business Age (Years)", min_value=0, max_value=100, value=5)
    employees    = st.number_input("Number of Employees", min_value=1, max_value=500, value=10)
    sector       = st.selectbox("Sector", list(SECTOR_MAP.keys()))
    location     = st.selectbox("Location", list(LOCATION_MAP.keys()))

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">▸ Financial Metrics</div>', unsafe_allow_html=True)
    monthly_revenue  = st.number_input("Monthly Revenue (KES)",  min_value=0, max_value=10_000_000, value=150_000, step=5000)
    monthly_expenses = st.number_input("Monthly Expenses (KES)", min_value=0, max_value=10_000_000, value=90_000,  step=5000)
    profit_margin    = st.slider("Profit Margin (%)", -50.0, 100.0, 20.0, step=0.5)
    avg_account_balance   = st.number_input("Avg Bank Balance (KES)", min_value=0, max_value=10_000_000, value=50_000, step=1000)
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
        VALIDATION: Stratified k-fold<br>
        ─────────────────────<br>
        For institutional use only.<br>
        Not a substitute for full due diligence.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN PANEL
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
    # ── Welcome / idle state ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#0D1117; border:1px solid #1E2D40; border-radius:12px; padding:1.5rem;">
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#00D4AA;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">
                ◈ ML Engine
            </div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.5rem;">
                RandomForest Classifier
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#475569; line-height:1.6;">
                Ensemble of 100 decision trees trained on Kenya-specific SME financial data with class-balanced sampling.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#0D1117; border:1px solid #1E2D40; border-radius:12px; padding:1.5rem;">
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#0EA5E9;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">
                ◈ Risk Framework
            </div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.5rem;">
                Dual-Layer Scoring
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#475569; line-height:1.6;">
                ML prediction combined with rule-based risk scoring for robust, explainable credit decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#0D1117; border:1px solid #1E2D40; border-radius:12px; padding:1.5rem;">
            <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#A855F7;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">
                ◈ Coverage
            </div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.5rem;">
                5 Counties · 5 Sectors
            </div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#475569; line-height:1.6;">
                Nairobi, Mombasa, Kisumu, Nakuru, Eldoret across Retail, Agri, Manufacturing, Services & Tech.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem; text-align:center; font-family:'DM Mono',monospace;
                font-size:0.75rem; color:#1E3A5F; letter-spacing:0.1em;">
        ← CONFIGURE SME PROFILE IN SIDEBAR AND CLICK RUN CREDIT ASSESSMENT
    </div>
    """, unsafe_allow_html=True)

else:
    # ============================================================
    # RUN ASSESSMENT
    # ============================================================
    input_df     = pd.DataFrame([inputs])
    outliers     = detect_outliers(inputs)
    rule_score, flags = compute_risk_score(inputs)
    band, band_color, band_emoji = risk_band(rule_score)
    ref_id       = generate_ref_id(inputs)
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    cash_flow    = monthly_revenue - monthly_expenses
    coverage_ratio = collateral_value / max(monthly_revenue * 6, 1)

    # ML prediction
    ml_prediction = model.predict(input_df)[0]
    ml_proba      = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    ml_default_prob = float(ml_proba[1]) if ml_proba is not None else 0.5

    # Override if hard flags + outliers
    hard_override = any(s in ("CRITICAL",) for s, _ in flags) and (
        cash_flow < 0 or existing_loans > 9 or loan_repayment_history <= 2
    )
    final_default = 1 if (ml_prediction == 1 or hard_override) else 0

    # Blended probability (70% ML, 30% rule)
    blended_prob = 0.7 * ml_default_prob + 0.3 * (rule_score / 100)

    # ── Outlier warning ──
    if outliers:
        st.warning(f"⚠️ **{len(outliers)} input(s) outside training distribution.** Predictions may be less reliable.\n\n" +
                   "\n".join(f"- {o}" for o in outliers))

    # ── Reference bar ──
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                font-family:'DM Mono',monospace; font-size:0.65rem; color:#334155;
                border-bottom:1px solid #1E2D40; padding-bottom:0.75rem; margin-bottom:1.5rem;">
        <span>REF: <span style="color:#475569;">{ref_id}</span></span>
        <span>ASSESSED: <span style="color:#475569;">{timestamp}</span></span>
        <span>ENGINE: <span style="color:#475569;">CreditIQ v2.1</span></span>
    </div>
    """, unsafe_allow_html=True)

    # ── VERDICT BANNER ──
    if final_default == 1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
            border: 1px solid rgba(239,68,68,0.4);
            border-left: 4px solid #EF4444;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#EF4444;
                            text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.4rem;">
                    ◈ Credit Decision
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#FCA5A5;">
                    ✗ &nbsp;APPLICATION DECLINED
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#7F1D1D; margin-top:0.3rem;">
                    Risk profile exceeds acceptable underwriting threshold
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:#EF4444; line-height:1;">
                    {blended_prob:.0%}
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#7F1D1D; text-transform:uppercase;">
                    Default Probability
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0,212,170,0.10), rgba(0,212,170,0.03));
            border: 1px solid rgba(0,212,170,0.35);
            border-left: 4px solid #00D4AA;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#00D4AA;
                            text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.4rem;">
                    ◈ Credit Decision
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#6EE7B7;">
                    ✓ &nbsp;APPLICATION APPROVED
                </div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#064E3B; margin-top:0.3rem;">
                    Risk profile within acceptable underwriting parameters
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:#00D4AA; line-height:1;">
                    {blended_prob:.0%}
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#064E3B; text-transform:uppercase;">
                    Default Probability
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI ROW ──
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Risk Band",       band)
    with k2: st.metric("Risk Score",      f"{rule_score:.0f} / 100")
    with k3: st.metric("Monthly Cash Flow", format_kes(cash_flow), delta=f"{cash_flow/max(monthly_revenue,1)*100:.1f}% margin")
    with k4: st.metric("Collateral Cover", f"{coverage_ratio:.1f}×", delta="of 6-mo revenue")
    with k5: st.metric("Repayment Score",  f"{loan_repayment_history} / 10")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["RISK BREAKDOWN", "FEATURE ANALYSIS", "AUDIT LOG"])

    # ── TAB 1: Risk Breakdown ──
    with tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-label">Risk Flags</div>', unsafe_allow_html=True)
            if flags:
                severity_colors = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#EAB308"}
                for severity, msg in sorted(flags, key=lambda x: ["CRITICAL","HIGH","MEDIUM"].index(x[0])):
                    c = severity_colors.get(severity, "#64748B")
                    st.markdown(f"""
                    <div style="display:flex; align-items:flex-start; gap:0.75rem; margin-bottom:0.6rem;
                                background:#0D1117; border:1px solid #1E2D40; border-left:3px solid {c};
                                border-radius:8px; padding:0.7rem 1rem;">
                        <span style="font-family:'DM Mono',monospace; font-size:0.6rem; font-weight:600;
                                     color:{c}; text-transform:uppercase; white-space:nowrap; padding-top:1px;">
                            {severity}
                        </span>
                        <span style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#94A3B8; line-height:1.4;">
                            {msg}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#0D1117; border:1px solid #1E2D40; border-radius:8px;
                            padding:1.5rem; text-align:center; color:#334155;
                            font-family:'DM Mono',monospace; font-size:0.75rem;">
                    NO RISK FLAGS DETECTED
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-label">Risk Score Composition</div>', unsafe_allow_html=True)
            factor_scores = {
                "Repayment History":  max(0, (10 - loan_repayment_history) * 3),
                "Existing Loans":     min(existing_loans * 2.5, 25),
                "Cash Flow Health":   20 if cash_flow < 0 else (10 if cash_flow < monthly_revenue * 0.1 else 0),
                "Profit Margin":      15 if profit_margin < 0 else (5 if profit_margin < 10 else 0),
                "Account Balance":    15 if avg_account_balance < 2000 else (7 if avg_account_balance < 10000 else 0),
                "Business Maturity":  8 if business_age < 2 else 0,
            }
            for factor, fscore in sorted(factor_scores.items(), key=lambda x: -x[1]):
                pct = fscore / 100
                color = "#EF4444" if pct > 0.15 else ("#F97316" if pct > 0.07 else "#00D4AA")
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                        <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#94A3B8;">{factor}</span>
                        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:{color};">{fscore:.0f} pts</span>
                    </div>
                    <div style="background:#1E2D40; border-radius:3px; height:5px; overflow:hidden;">
                        <div style="background:{color}; width:{min(pct*100*1.5, 100):.0f}%; height:100%;
                                    border-radius:3px; transition:width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ML confidence
            if ml_proba is not None:
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">ML Model Confidence</div>', unsafe_allow_html=True)
                for cls, prob in zip(model.classes_, ml_proba):
                    label = "Default" if str(cls) == "1" else "No Default"
                    c = "#EF4444" if str(cls) == "1" else "#00D4AA"
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                            <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#94A3B8;">{label}</span>
                            <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:{c};">{prob:.1%}</span>
                        </div>
                        <div style="background:#1E2D40; border-radius:3px; height:5px; overflow:hidden;">
                            <div style="background:{c}; width:{prob*100:.0f}%; height:100%; border-radius:3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 2: Feature Analysis ──
    with tab2:
        fa1, fa2 = st.columns(2)

        with fa1:
            st.markdown('<div class="section-label">Financial Health Summary</div>', unsafe_allow_html=True)
            metrics = [
                ("Monthly Revenue",    format_kes(monthly_revenue),    None),
                ("Monthly Expenses",   format_kes(monthly_expenses),   None),
                ("Net Cash Flow",      format_kes(cash_flow),          "positive" if cash_flow >= 0 else "negative"),
                ("Profit Margin",      f"{profit_margin:.1f}%",        "positive" if profit_margin >= 15 else "negative"),
                ("Account Balance",    format_kes(avg_account_balance), None),
                ("Collateral Value",   format_kes(collateral_value),   None),
                ("Collateral Cover",   f"{coverage_ratio:.2f}×",       "positive" if coverage_ratio >= 1 else "negative"),
            ]
            for label, val, sentiment in metrics:
                c = "#00D4AA" if sentiment == "positive" else ("#EF4444" if sentiment == "negative" else "#94A3B8")
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.6rem 0; border-bottom:1px solid #0F1923;">
                    <span style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#475569;">{label}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.82rem; color:{c}; font-weight:500;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        with fa2:
            st.markdown('<div class="section-label">Business Profile</div>', unsafe_allow_html=True)
            profile = [
                ("Sector",                sector),
                ("Location",              location),
                ("Business Age",          f"{business_age} years"),
                ("Employees",             str(employees)),
                ("Transaction Frequency", f"{transaction_frequency}/month"),
                ("Repayment History",     f"{loan_repayment_history}/10"),
                ("Existing Loans",        str(existing_loans)),
            ]
            for label, val in profile:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.6rem 0; border-bottom:1px solid #0F1923;">
                    <span style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#475569;">{label}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#94A3B8; font-weight:500;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Audit Log ──
    with tab3:
        st.markdown('<div class="section-label">Decision Audit Trail</div>', unsafe_allow_html=True)
        audit = {
            "reference_id":        ref_id,
            "timestamp":           timestamp,
            "model_version":       "RandomForestClassifier v2.1",
            "ml_prediction":       int(ml_prediction),
            "ml_default_prob":     f"{ml_default_prob:.4f}",
            "rule_risk_score":     f"{rule_score:.1f}/100",
            "blended_prob":        f"{blended_prob:.4f}",
            "hard_override":       str(hard_override),
            "final_decision":      "DECLINED" if final_default == 1 else "APPROVED",
            "risk_band":           band,
            "flags_count":         len(flags),
            "outliers_detected":   len(outliers),
        }
        for k, v in audit.items():
            st.markdown(f"""
            <div style="display:flex; gap:2rem; padding:0.45rem 0; border-bottom:1px solid #0F1923;">
                <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#334155;
                             min-width:200px; text-transform:uppercase; letter-spacing:0.05em;">{k}</span>
                <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#64748B;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Raw Feature Vector</div>', unsafe_allow_html=True)
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top:4rem; padding-top:1.5rem; border-top:1px solid #0F1923;
            display:flex; justify-content:space-between; align-items:center;">
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#1E3A5F;">
        CREDITIQ KENYA &nbsp;·&nbsp; SME RISK INTELLIGENCE &nbsp;·&nbsp; FOR INSTITUTIONAL USE ONLY
    </div>
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#1E3A5F;">
        NOT A SUBSTITUTE FOR FULL CREDIT DUE DILIGENCE
    </div>
</div>
""", unsafe_allow_html=True)
