import streamlit as st
import pandas as pd
import io
from pathlib import Path
import joblib, json, hashlib, subprocess, sys
from datetime import datetime

st.set_page_config(page_title="CreditIQ Kenya", page_icon="▣", layout="wide", initial_sidebar_state="collapsed")

# ── Auto-train ───────────────────────────────────────────────
if not Path("models/kenya_sme_credit_model.pkl").exists():
    with st.spinner("Initialising intelligence core..."):
        subprocess.run([sys.executable, "setup.py"], check=True)
        st.rerun()

@st.cache_resource
def load():
    return joblib.load("models/kenya_sme_credit_model.pkl"), \
           joblib.load("models/feature_columns.pkl")
model, _ = load()

SECTORS   = ["Retail","Manufacturing","Agriculture","Services","Technology"]
LOCATIONS = ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret"]
SECTOR_MAP   = {s:i for i,s in enumerate(SECTORS)}
LOCATION_MAP = {l:i for i,l in enumerate(LOCATIONS)}
INTEREST_RATE = 0.18

# ═══════════════════════════════════════════════════════════════
# CSS — Luxury dark editorial. Instrument Serif + DM Mono.
# Deep navy canvas, razor gold accents, cinematic spacing.
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700&display=swap');

:root {
  --bg:       #05080F;
  --bg2:      #090D16;
  --bg3:      #0D1220;
  --border:   #141E30;
  --border2:  #1C2A40;
  --gold:     #C9A84C;
  --gold2:    #E8C96A;
  --teal:     #00BFA5;
  --red:      #E05252;
  --orange:   #E07A30;
  --yellow:   #D4B84A;
  --dim:      #3A4A60;
  --muted:    #5A7090;
  --text:     #C8D8E8;
  --bright:   #EAF0F8;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main > div { background: var(--bg) !important; }

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; display: none !important; }

[data-testid="collapsedControl"] { display: none !important; }

.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Master layout shell ── */
.ciq-shell {
  min-height: 100vh;
  background: var(--bg);
  font-family: 'Sora', sans-serif;
  color: var(--text);
}

/* ── Top bar ── */
.ciq-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.1rem 3rem;
  border-bottom: 1px solid var(--border);
  background: rgba(5,8,15,0.97);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 999;
}
.ciq-logo {
  font-family: 'Instrument Serif', serif;
  font-size: 1.45rem;
  color: var(--bright);
  letter-spacing: 0.01em;
}
.ciq-logo span { color: var(--gold); }
.ciq-tagline {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  color: var(--dim);
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.ciq-badge {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  color: var(--gold);
  border: 1px solid var(--gold);
  border-radius: 3px;
  padding: 0.2rem 0.6rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.8;
}

/* ── Page content ── */
.ciq-page { padding: 2.5rem 3rem 4rem 3rem; }

/* ── Section titles ── */
.ciq-section {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.ciq-section::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Cards ── */
.ciq-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.6rem;
  transition: border-color 0.2s;
}
.ciq-card:hover { border-color: var(--border2); }

.ciq-card-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.ciq-card-value {
  font-family: 'Instrument Serif', serif;
  font-size: 2.2rem;
  color: var(--bright);
  line-height: 1;
}
.ciq-card-sub {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  color: var(--muted);
  margin-top: 0.35rem;
}

/* ── Decision banner ── */
.ciq-approved {
  background: linear-gradient(135deg, rgba(0,191,165,0.08), rgba(0,191,165,0.02));
  border: 1px solid rgba(0,191,165,0.25);
  border-left: 4px solid var(--teal);
  border-radius: 12px;
  padding: 2rem 2.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}
.ciq-declined {
  background: linear-gradient(135deg, rgba(224,82,82,0.08), rgba(224,82,82,0.02));
  border: 1px solid rgba(224,82,82,0.25);
  border-left: 4px solid var(--red);
  border-radius: 12px;
  padding: 2rem 2.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}
.ciq-decision-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.ciq-decision-title {
  font-family: 'Instrument Serif', serif;
  font-size: 2.4rem;
  line-height: 1.1;
}
.ciq-decision-sub {
  font-family: 'Sora', sans-serif;
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 0.4rem;
}
.ciq-prob {
  text-align: right;
}
.ciq-prob-value {
  font-family: 'Instrument Serif', serif;
  font-size: 3.8rem;
  line-height: 1;
}
.ciq-prob-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* ── KPI row ── */
.ciq-kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}
.ciq-kpi {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
}
.ciq-kpi-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.ciq-kpi-val {
  font-family: 'Instrument Serif', serif;
  font-size: 1.7rem;
  color: var(--bright);
  line-height: 1;
}
.ciq-kpi-delta {
  font-family: 'DM Mono', monospace;
  font-size: 0.62rem;
  color: var(--teal);
  margin-top: 0.3rem;
}

/* ── Flag pills ── */
.ciq-flag {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 0.9rem 1.2rem;
  border-radius: 8px;
  margin-bottom: 0.6rem;
  border-left: 3px solid;
}
.ciq-flag-critical { background: rgba(224,82,82,0.06);  border-color: var(--red);    }
.ciq-flag-high     { background: rgba(224,122,48,0.06); border-color: var(--orange); }
.ciq-flag-medium   { background: rgba(212,184,74,0.06); border-color: var(--yellow); }
.ciq-flag-sev {
  font-family: 'DM Mono', monospace;
  font-size: 0.58rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  white-space: nowrap;
  padding: 0.15rem 0.5rem;
  border-radius: 3px;
  margin-top: 0.1rem;
}
.sev-critical { color: var(--red);    background: rgba(224,82,82,0.12);  }
.sev-high     { color: var(--orange); background: rgba(224,122,48,0.12); }
.sev-medium   { color: var(--yellow); background: rgba(212,184,74,0.12); }
.ciq-flag-msg {
  font-family: 'Sora', sans-serif;
  font-size: 0.82rem;
  color: var(--text);
  line-height: 1.5;
}

/* ── Data rows ── */
.ciq-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--border);
}
.ciq-row:last-child { border-bottom: none; }
.ciq-row-label {
  font-family: 'Sora', sans-serif;
  font-size: 0.8rem;
  color: var(--muted);
}
.ciq-row-val {
  font-family: 'DM Mono', monospace;
  font-size: 0.8rem;
  color: var(--text);
  font-weight: 500;
}

/* ── Bar track ── */
.ciq-bar-wrap { margin-bottom: 0.75rem; }
.ciq-bar-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.25rem;
}
.ciq-bar-name {
  font-family: 'Sora', sans-serif;
  font-size: 0.76rem;
  color: var(--text);
}
.ciq-bar-pct {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--gold);
}
.ciq-track {
  background: var(--border);
  border-radius: 2px;
  height: 3px;
  overflow: hidden;
}
.ciq-fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--gold), var(--gold2));
}

/* ── Ref strip ── */
.ciq-ref {
  font-family: 'DM Mono', monospace;
  font-size: 0.6rem;
  color: var(--dim);
  display: flex;
  gap: 2rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 0.8rem 1.6rem !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s !important;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
  color: var(--gold) !important;
  border-bottom: 2px solid var(--gold) !important;
  background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]    { display: none !important; }

/* inputs */
input, [data-baseweb="input"] input,
[data-baseweb="select"] > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 7px !important;
  color: var(--bright) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.85rem !important;
}
[data-baseweb="select"] span { color: var(--text) !important; }
label, [data-testid="stWidgetLabel"] p {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
/* slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--gold) !important;
  border: 2px solid var(--bg) !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
  font-family: 'DM Mono', monospace !important;
  color: var(--dim) !important;
  font-size: 0.65rem !important;
}
/* primary button */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
  color: var(--bg) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.08em !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.75rem 2rem !important;
  transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}
/* secondary button / download */
.stButton > button,
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em !important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--gold) !important;
  color: var(--gold) !important;
}
/* dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
/* metrics */
div[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 1.1rem 1.3rem !important;
}
div[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: 0.6rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--bright) !important;
  font-family: 'Instrument Serif', serif !important;
  font-size: 1.6rem !important;
}
/* file uploader */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 10px !important;
}
/* alerts */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 0.82rem !important;
}
/* divider */
hr { border-color: var(--border) !important; }

/* hide streamlit default metric delta color */
[data-testid="stMetricDelta"] svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def risk_score(d):
    s, flags = 0, []
    rph = d["loan_repayment_history"]
    s += (10 - rph) * 3
    if rph <= 2:   flags.append(("CRITICAL", "Repayment history critically poor"))
    elif rph <= 4: flags.append(("HIGH", "Below-average repayment history"))
    el = d["existing_loans"]
    s += min(el * 2.5, 25)
    if el > 9:   flags.append(("CRITICAL", f"{el} existing loans — above safe threshold"))
    elif el > 5: flags.append(("HIGH", f"{el} existing loans — elevated"))
    cf = d["monthly_revenue"] - d["monthly_expenses"]
    if cf < 0:
        s += 20; flags.append(("CRITICAL", f"Negative cash flow: KES {cf:,.0f}/mo"))
    elif cf < d["monthly_revenue"] * 0.1:
        s += 10; flags.append(("MEDIUM", "Thin cash flow margin (<10%)"))
    if d["profit_margin"] < 0:
        s += 15; flags.append(("HIGH", f"Negative profit margin: {d['profit_margin']:.1f}%"))
    if d["avg_account_balance"] < 2000:
        s += 15; flags.append(("HIGH", f"Critical low balance: KES {d['avg_account_balance']:,.0f}"))
    elif d["avg_account_balance"] < 10000:
        s += 7
    if d["business_age"] < 2:
        s += 8; flags.append(("MEDIUM", f"Early-stage business: {d['business_age']} yr(s)"))
    return min(s, 100), flags

def band(score):
    for t, l in [(75,"CRITICAL"),(55,"HIGH"),(35,"MODERATE"),(15,"LOW")]:
        if score >= t: return l
    return "MINIMAL"

def kes(v):
    return f"KES {v/1e6:.2f}M" if v>=1e6 else f"KES {v/1e3:.0f}K" if v>=1e3 else f"KES {v:.0f}"

def loan_recommendation(d, final, blended, cov):
    if final == 1: return 0, 0, "DECLINED", 0
    surplus = d["monthly_revenue"] - d["monthly_expenses"]
    max_repayment = surplus * 0.40
    loan_by_capacity = max_repayment * 24
    if cov >= 1.5:   cm = 0.70
    elif cov >= 1.0: cm = 0.50
    elif cov >= 0.5: cm = 0.30
    elif cov >= 0.2: cm = 0.15
    else:            cm = 0.10
    loan_by_collateral = d["collateral_value"] * cm
    rb = band(int(blended * 100))
    risk_mult = {"MINIMAL":1.0,"LOW":0.85,"MODERATE":0.65,"HIGH":0.40,"CRITICAL":0.0}[rb]
    raw = min(loan_by_capacity, loan_by_collateral)
    final_amount = min(raw * risk_mult, 5_000_000)
    tenure = {"MINIMAL":24,"LOW":18,"MODERATE":12,"HIGH":6,"CRITICAL":0}[rb]
    return final_amount, tenure, rb, max_repayment

def repayment_schedule(principal, annual_rate, months):
    if principal <= 0 or months <= 0: return pd.DataFrame()
    r = annual_rate / 12
    pmt = principal * (r*(1+r)**months) / ((1+r)**months - 1)
    rows, bal = [], principal
    for m in range(1, months+1):
        interest = bal * r
        princ    = pmt - interest
        bal     -= princ
        rows.append({"Month":m,"Monthly Payment":round(pmt),"Principal":round(princ),
                     "Interest":round(interest),"Balance":round(max(bal,0))})
    return pd.DataFrame(rows)

def assess_single(row):
    inp = {
        "business_age": row.get("business_age",0), "employees": row.get("employees",1),
        "sector": SECTOR_MAP.get(row.get("sector","Retail"),0),
        "location": LOCATION_MAP.get(row.get("location","Nairobi"),0),
        "monthly_revenue": row.get("monthly_revenue",0), "monthly_expenses": row.get("monthly_expenses",0),
        "profit_margin": row.get("profit_margin",0), "avg_account_balance": row.get("avg_account_balance",0),
        "transaction_frequency": row.get("transaction_frequency",0),
        "loan_repayment_history": row.get("loan_repayment_history",5),
        "existing_loans": row.get("existing_loans",0), "collateral_value": row.get("collateral_value",0),
    }
    df_inp = pd.DataFrame([inp])
    rs, flags = risk_score(inp)
    ml_pred   = model.predict(df_inp)[0]
    ml_proba  = model.predict_proba(df_inp)[0]
    ml_def    = float(ml_proba[1])
    override  = any(s=="CRITICAL" for s,_ in flags) and (
        inp["monthly_revenue"]-inp["monthly_expenses"]<0 or
        inp["existing_loans"]>9 or inp["loan_repayment_history"]<=2)
    final     = 1 if (ml_pred==1 or override) else 0
    blended   = 0.7*ml_def + 0.3*(rs/100)
    cf        = inp["monthly_revenue"] - inp["monthly_expenses"]
    cov       = inp["collateral_value"] / max(inp["monthly_revenue"]*6,1)
    loan_amt, tenure, _, _ = loan_recommendation(inp, final, blended, cov)
    return {"decision":"DECLINED" if final==1 else "APPROVED","risk_band":band(rs),
            "risk_score":round(rs,1),"default_prob_%":round(blended*100,1),
            "cash_flow":round(cf),"collateral_cover":round(cov,2),
            "recommended_loan_KES":round(loan_amt),"tenure_months":tenure,"flags":len(flags)}

# ═══════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="ciq-topbar">
  <div>
    <div class="ciq-logo">Credit<span>IQ</span> Kenya</div>
    <div class="ciq-tagline">SME Risk Intelligence Platform</div>
  </div>
  <div style="display:flex;gap:1rem;align-items:center;">
    <div class="ciq-tagline">RandomForest · 2,000 SME Records · 12 Features</div>
    <div class="ciq-badge">v2.1</div>
  </div>
</div>
<div class="ciq-page">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_assess, tab_loan, tab_batch, tab_perf = st.tabs([
    "Credit Assessment",
    "Loan Recommendation",
    "Batch Assessment",
    "Model Performance",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — CREDIT ASSESSMENT
# ══════════════════════════════════════════════════════════════
with tab_assess:
    st.markdown('<div class="ciq-section">SME Profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="ciq-card">', unsafe_allow_html=True)
        st.markdown("**Business Profile**")
        ba       = st.number_input("Business Age (yrs)", 0, 100, 5, key="ba")
        emp      = st.number_input("Employees", 1, 500, 10, key="emp")
        sector   = st.selectbox("Sector", SECTORS, key="sector")
        location = st.selectbox("Location", LOCATIONS, key="location")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ciq-card">', unsafe_allow_html=True)
        st.markdown("**Financial Metrics**")
        mr  = st.number_input("Monthly Revenue (KES)",  0, 10_000_000, 150_000, 5000, key="mr")
        me  = st.number_input("Monthly Expenses (KES)", 0, 10_000_000, 90_000,  5000, key="me")
        pm  = st.slider("Profit Margin (%)", -50, 100, 20, key="pm")
        ab  = st.number_input("Avg Bank Balance (KES)", 0, 10_000_000, 50_000, 1000, key="ab")
        tf  = st.number_input("Monthly Transactions", 0, 200, 15, key="tf")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="ciq-card">', unsafe_allow_html=True)
        st.markdown("**Credit History**")
        rph = st.slider("Repayment Score (0=Poor · 10=Excellent)", 0, 10, 7, key="rph")
        el  = st.number_input("Existing Loans", 0, 20, 1, key="el")
        cv  = st.number_input("Collateral Value (KES)", 0, 10_000_000, 200_000, 5000, key="cv")
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Run Credit Assessment", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if run:
        st.markdown("<br>", unsafe_allow_html=True)
        inp = {
            "business_age":ba,"employees":emp,"sector":SECTOR_MAP[sector],
            "location":LOCATION_MAP[location],"monthly_revenue":mr,"monthly_expenses":me,
            "profit_margin":pm,"avg_account_balance":ab,"transaction_frequency":tf,
            "loan_repayment_history":rph,"existing_loans":el,"collateral_value":cv,
        }
        df_inp        = pd.DataFrame([inp])
        rs, flags     = risk_score(inp)
        rb            = band(rs)
        ml_pred       = model.predict(df_inp)[0]
        ml_proba      = model.predict_proba(df_inp)[0]
        ml_def_prob   = float(ml_proba[1])
        hard_override = any(s=="CRITICAL" for s,_ in flags) and (mr-me<0 or el>9 or rph<=2)
        final         = 1 if (ml_pred==1 or hard_override) else 0
        blended       = 0.7*ml_def_prob + 0.3*(rs/100)
        cf            = mr - me
        cov           = cv / max(mr*6,1)
        rid           = "CIQ-" + hashlib.md5((str(inp)+datetime.now().isoformat()).encode()).hexdigest()[:8].upper()
        ts            = datetime.now().strftime("%Y-%m-%d  %H:%M UTC")

        st.session_state.update({"last_inp":inp,"last_final":final,"last_blended":blended,"last_cov":cov,"last_rid":rid})

        # Ref strip
        st.markdown(f'<div class="ciq-ref"><span>REF &nbsp;{rid}</span><span>ASSESSED &nbsp;{ts}</span><span>ENGINE &nbsp;CreditIQ v2.1 · RandomForest</span></div>', unsafe_allow_html=True)

        # Decision banner
        if final == 1:
            override_note = "<div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#E05252;margin-top:0.6rem;'>Hard override applied — extreme risk indicators</div>" if hard_override else ""
            st.markdown(f"""
            <div class="ciq-declined">
              <div>
                <div class="ciq-decision-label" style="color:#E05252;">Credit Decision</div>
                <div class="ciq-decision-title" style="color:#FFAAAA;">Application Declined</div>
                <div class="ciq-decision-sub">Risk profile exceeds acceptable underwriting threshold</div>
                {override_note}
              </div>
              <div class="ciq-prob">
                <div class="ciq-prob-value" style="color:#E05252;">{blended:.0%}</div>
                <div class="ciq-prob-label" style="color:#7A3030;">Default Probability</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ciq-approved">
              <div>
                <div class="ciq-decision-label" style="color:#00BFA5;">Credit Decision</div>
                <div class="ciq-decision-title" style="color:#AAFFF0;">Application Approved</div>
                <div class="ciq-decision-sub">Risk profile within acceptable underwriting parameters</div>
              </div>
              <div class="ciq-prob">
                <div class="ciq-prob-value" style="color:#00BFA5;">{blended:.0%}</div>
                <div class="ciq-prob-label" style="color:#004D40;">Default Probability</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # KPI row
        st.markdown(f"""
        <div class="ciq-kpi-grid">
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Risk Band</div>
            <div class="ciq-kpi-val">{rb}</div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Risk Score</div>
            <div class="ciq-kpi-val">{rs:.0f}<span style="font-size:1rem;color:var(--muted);">/100</span></div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Cash Flow / mo</div>
            <div class="ciq-kpi-val" style="font-size:1.4rem;">{kes(cf)}</div>
            <div class="ciq-kpi-delta">{cf/max(mr,1)*100:.1f}% of revenue</div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Collateral Cover</div>
            <div class="ciq-kpi-val">{cov:.2f}<span style="font-size:1rem;color:var(--muted);">×</span></div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Repayment Score</div>
            <div class="ciq-kpi-val">{rph}<span style="font-size:1rem;color:var(--muted);">/10</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Inner tabs
        r1, r2, r3 = st.tabs(["Risk Flags", "Financials", "Audit"])

        with r1:
            if flags:
                for sev, msg in sorted(flags, key=lambda x: ["CRITICAL","HIGH","MEDIUM"].index(x[0])):
                    cls = sev.lower()
                    st.markdown(f"""
                    <div class="ciq-flag ciq-flag-{cls}">
                      <span class="ciq-flag-sev sev-{cls}">{sev}</span>
                      <span class="ciq-flag-msg">{msg}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No risk flags detected — profile is clean.")

        with r2:
            fa, fb = st.columns(2, gap="large")
            with fa:
                st.markdown('<div class="ciq-section">Financial Health</div>', unsafe_allow_html=True)
                for lbl, val in [("Monthly Revenue",kes(mr)),("Monthly Expenses",kes(me)),
                                  ("Net Cash Flow",kes(cf)),("Profit Margin",f"{pm:.1f}%"),
                                  ("Avg Bank Balance",kes(ab)),("Collateral Value",kes(cv))]:
                    st.markdown(f'<div class="ciq-row"><span class="ciq-row-label">{lbl}</span><span class="ciq-row-val">{val}</span></div>', unsafe_allow_html=True)
            with fb:
                st.markdown('<div class="ciq-section">Business Profile</div>', unsafe_allow_html=True)
                for lbl, val in [("Sector",sector),("Location",location),("Business Age",f"{ba} yrs"),
                                  ("Employees",str(emp)),("Monthly Transactions",f"{tf}"),
                                  ("Existing Loans",str(el))]:
                    st.markdown(f'<div class="ciq-row"><span class="ciq-row-label">{lbl}</span><span class="ciq-row-val">{val}</span></div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="ciq-section">Decision Audit Trail</div>', unsafe_allow_html=True)
            for k, v in {"Reference ID":rid,"Timestamp":ts,"ML Default Probability":f"{ml_def_prob:.4f}",
                          "Rule-Based Score":f"{rs:.1f} / 100","Blended Probability":f"{blended:.4f}",
                          "Hard Override":str(hard_override),"Final Decision":"DECLINED" if final==1 else "APPROVED"}.items():
                st.markdown(f'<div class="ciq-row"><span class="ciq-row-label">{k}</span><span class="ciq-row-val">{v}</span></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_inp, use_container_width=True, hide_index=True)

        if final == 0:
            st.info("Switch to Loan Recommendation for the suggested loan amount and repayment schedule.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — LOAN RECOMMENDATION
# ══════════════════════════════════════════════════════════════
with tab_loan:
    if "last_inp" not in st.session_state or st.session_state.get("last_final",1) == 1:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
          <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:#3A4A60;margin-bottom:1rem;">
            No approved application
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2A3A50;letter-spacing:0.1em;">
            Run a successful Credit Assessment first, then return here.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        inp     = st.session_state["last_inp"]
        final   = st.session_state["last_final"]
        blended = st.session_state["last_blended"]
        cov     = st.session_state["last_cov"]
        rid     = st.session_state["last_rid"]

        loan_amt, tenure, rb, max_repayment = loan_recommendation(inp, final, blended, cov)
        surplus = inp["monthly_revenue"] - inp["monthly_expenses"]

        st.markdown(f'<div class="ciq-ref"><span>REF &nbsp;{rid}</span><span>LOAN RECOMMENDATION ENGINE v1.0</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="ciq-section">Recommended Loan</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ciq-kpi-grid">
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Recommended Loan</div>
            <div class="ciq-kpi-val" style="font-size:1.5rem;color:#C9A84C;">{kes(loan_amt)}</div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Tenure</div>
            <div class="ciq-kpi-val">{tenure}<span style="font-size:1rem;color:var(--muted);"> mo</span></div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Collateral Cover</div>
            <div class="ciq-kpi-val">{cov:.2f}<span style="font-size:1rem;color:var(--muted);">×</span></div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Risk Band</div>
            <div class="ciq-kpi-val">{rb}</div>
          </div>
          <div class="ciq-kpi">
            <div class="ciq-kpi-label">Max Monthly Repayment</div>
            <div class="ciq-kpi-val" style="font-size:1.4rem;">{kes(max_repayment)}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="ciq-section">Calculation Breakdown</div>', unsafe_allow_html=True)
        if cov >= 1.5:   cm, ct = 0.70, "Above 1.5×"
        elif cov >= 1.0: cm, ct = 0.50, "1.0× – 1.5×"
        elif cov >= 0.5: cm, ct = 0.30, "0.5× – 1.0×"
        elif cov >= 0.2: cm, ct = 0.15, "0.2× – 0.5×"
        else:            cm, ct = 0.10, "Below 0.2×"
        risk_mult = {"MINIMAL":1.0,"LOW":0.85,"MODERATE":0.65,"HIGH":0.40,"CRITICAL":0.0}[rb]
        cap_loan  = surplus * 0.40 * 24
        coll_loan = inp["collateral_value"] * cm

        for lbl, val in [("Monthly Surplus", kes(surplus)),
                          ("Max Repayment Capacity (40% of surplus)", kes(surplus*0.40)),
                          ("Loan by Capacity (24 months)", kes(cap_loan)),
                          ("Collateral Cover Tier", ct),
                          ("Collateral Multiplier", f"{cm:.0%}"),
                          ("Loan by Collateral", kes(coll_loan)),
                          ("Binding Ceiling (min of above)", kes(min(cap_loan,coll_loan))),
                          ("Risk Adjustment Multiplier", f"{risk_mult:.0%}"),
                          ("Final Recommended Loan", kes(loan_amt))]:
            bold = "font-weight:600;color:var(--gold);" if lbl == "Final Recommended Loan" else ""
            st.markdown(f'<div class="ciq-row"><span class="ciq-row-label">{lbl}</span><span class="ciq-row-val" style="{bold}">{val}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ciq-section">Repayment Schedule</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1: tenure_sel = st.selectbox("Tenure (months)", [6,12,18,24,36], index=[6,12,18,24,36].index(min(tenure,36)), key="t_sel")
        with sc2: rate_sel   = st.slider("Annual Interest Rate (%)", 10, 30, int(INTEREST_RATE*100), key="r_sel")

        sched = repayment_schedule(loan_amt, rate_sel/100, tenure_sel)
        if not sched.empty:
            pmt       = sched["Monthly Payment"].iloc[0]
            total_paid = sched["Monthly Payment"].sum()
            total_int  = sched["Interest"].sum()
            s1,s2,s3  = st.columns(3)
            s1.metric("Monthly Payment", kes(pmt))
            s2.metric("Total Repayable", kes(total_paid))
            s3.metric("Total Interest",  kes(total_int))
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(sched, use_container_width=True, hide_index=True)
            st.download_button("Download Repayment Schedule (CSV)", sched.to_csv(index=False),
                               f"repayment_{rid}.csv","text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 3 — BATCH ASSESSMENT
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="ciq-section">Batch Assessment</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Sora,sans-serif;font-size:0.85rem;color:var(--muted);margin-bottom:1.5rem;">Upload a CSV of SMEs and get a fully scored results table with loan recommendations.</p>', unsafe_allow_html=True)

    template_data = {
        "business_age":[8,1],"employees":[25,3],"sector":["Technology","Retail"],
        "location":["Nairobi","Kisumu"],"monthly_revenue":[300000,50000],
        "monthly_expenses":[180000,65000],"profit_margin":[35,-15],
        "avg_account_balance":[120000,1500],"transaction_frequency":[30,5],
        "loan_repayment_history":[9,2],"existing_loans":[1,11],"collateral_value":[400000,10000],
    }
    st.download_button("Download CSV Template", pd.DataFrame(template_data).to_csv(index=False),
                       "batch_template.csv","text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        required = list(template_data.keys())
        missing  = [c for c in required if c not in df_batch.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            with st.spinner(f"Assessing {len(df_batch)} SMEs..."):
                results    = [assess_single(r.to_dict()) for _,r in df_batch.iterrows()]
                results_df = pd.DataFrame(results)
                combined   = pd.concat([df_batch.reset_index(drop=True), results_df], axis=1)

            approved   = (results_df["decision"]=="APPROVED").sum()
            declined   = (results_df["decision"]=="DECLINED").sum()
            total_loan = results_df["recommended_loan_KES"].sum()
            approval_rate = approved / len(results_df) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ciq-kpi-grid">
              <div class="ciq-kpi"><div class="ciq-kpi-label">Total SMEs</div><div class="ciq-kpi-val">{len(combined)}</div></div>
              <div class="ciq-kpi"><div class="ciq-kpi-label">Approved</div><div class="ciq-kpi-val" style="color:#00BFA5;">{approved}</div></div>
              <div class="ciq-kpi"><div class="ciq-kpi-label">Declined</div><div class="ciq-kpi-val" style="color:#E05252;">{declined}</div></div>
              <div class="ciq-kpi"><div class="ciq-kpi-label">Approval Rate</div><div class="ciq-kpi-val">{approval_rate:.0f}<span style="font-size:1rem;color:var(--muted);">%</span></div></div>
              <div class="ciq-kpi"><div class="ciq-kpi-label">Total Loan Book</div><div class="ciq-kpi-val" style="font-size:1.3rem;color:#C9A84C;">{kes(total_loan)}</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(combined, use_container_width=True, hide_index=True)
            st.download_button("Download Full Results (CSV)", combined.to_csv(index=False),
                               "batch_results.csv","text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab_perf:
    mp = Path("models/model_metrics.json")
    if not mp.exists():
        st.warning("Run `python setup.py` to generate metrics.")
        st.stop()
    m = json.loads(mp.read_text())

    st.markdown('<div class="ciq-section">Model Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Accuracy",       f"{m['accuracy']:.1%}")
    c2.metric("ROC AUC",        f"{m['roc_auc']:.3f}")
    c3.metric("CV F1",          f"{m['cv_f1_mean']:.3f}")
    c4.metric("Default Recall", f"{m['default_recall']:.1%}")
    c5.metric("Default Prec",   f"{m['default_precision']:.1%}")
    c6.metric("Avg Precision",  f"{m['avg_precision']:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="ciq-section">ROC Curve</div>', unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"TPR": m["tpr"]}, index=m["fpr"]))
        st.markdown('<div class="ciq-section">Precision-Recall Curve</div>', unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"Precision": m["precision_curve"]}, index=m["recall_curve"]))
    with col2:
        st.markdown('<div class="ciq-section">Feature Importance</div>', unsafe_allow_html=True)
        fi = m["feature_importance"]

        # custom bar chart
        max_fi = max(fi.values())
        for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
            pct = imp / max_fi * 100
            st.markdown(f"""
            <div class="ciq-bar-wrap">
              <div class="ciq-bar-header">
                <span class="ciq-bar-name">{feat.replace('_',' ').title()}</span>
                <span class="ciq-bar-pct">{imp*100:.1f}%</span>
              </div>
              <div class="ciq-track"><div class="ciq-fill" style="width:{pct:.0f}%"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ciq-section">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = m["confusion_matrix"]
        ca,cb = st.columns(2)
        ca.metric("True Negative",  cm[0][0])
        cb.metric("False Positive", cm[0][1])
        ca.metric("False Negative", cm[1][0])
        cb.metric("True Positive",  cm[1][1])

# Close page div
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="border-top:1px solid #141E30;padding:1.5rem 3rem;display:flex;justify-content:space-between;
            font-family:'DM Mono',monospace;font-size:0.58rem;color:#1C2A40;margin-top:2rem;">
  <span>CREDITIQ KENYA &nbsp;·&nbsp; SME RISK INTELLIGENCE &nbsp;·&nbsp; FOR INSTITUTIONAL USE ONLY</span>
  <span>NOT A SUBSTITUTE FOR FULL CREDIT DUE DILIGENCE</span>
</div>""", unsafe_allow_html=True)
