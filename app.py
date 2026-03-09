import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib, json, hashlib, subprocess, sys
from datetime import datetime

st.set_page_config(page_title="CreditIQ Kenya", page_icon="🏦", layout="wide")

# ── Auto-train ───────────────────────────────────────────────
if not Path("models/kenya_sme_credit_model.pkl").exists():
    with st.spinner("First-time setup: training model..."):
        subprocess.run([sys.executable, "setup.py"], check=True)
        st.rerun()

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load():
    return joblib.load("models/kenya_sme_credit_model.pkl"), \
           joblib.load("models/feature_columns.pkl")
model, feature_cols = load()

# ── Constants ────────────────────────────────────────────────
SECTORS   = ["Retail","Manufacturing","Agriculture","Services","Technology"]
LOCATIONS = ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret"]
SECTOR_MAP   = {s:i for i,s in enumerate(SECTORS)}
LOCATION_MAP = {l:i for i,l in enumerate(LOCATIONS)}

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#060A0F!important;color:#E2E8F0!important;font-family:'Space Grotesk',sans-serif!important}
[data-testid="stAppViewContainer"]>.main{background:#060A0F!important}
[data-testid="stSidebar"]{background:#0A0F17!important;border-right:1px solid #1a2535!important}
[data-testid="stSidebar"] label{font-family:'JetBrains Mono',monospace!important;font-size:0.68rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important;color:#64748B!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#111827!important;border:1px solid #1a2535!important}
.stButton>button{background:linear-gradient(135deg,#00D4AA,#0EA5E9)!important;color:#060A0F!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;border:none!important;border-radius:8px!important;width:100%!important;padding:0.7rem!important}
div[data-testid="metric-container"]{background:#0D1117!important;border:1px solid #1a2535!important;border-radius:10px!important;padding:1rem!important}
[data-baseweb="tab"]{font-family:'JetBrains Mono',monospace!important;font-size:0.72rem!important;text-transform:uppercase!important;color:#475569!important}
[aria-selected="true"][data-baseweb="tab"]{color:#00D4AA!important;border-bottom:2px solid #00D4AA!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem!important;max-width:1400px!important}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────
def risk_score(d):
    s, flags = 0, []
    rph = d["loan_repayment_history"]
    s += (10 - rph) * 3
    if rph <= 2:   flags.append(("CRITICAL", "Repayment history critically poor"))
    elif rph <= 4: flags.append(("HIGH",     "Below-average repayment history"))
    el = d["existing_loans"]
    s += min(el * 2.5, 25)
    if el > 9:   flags.append(("CRITICAL", f"{el} existing loans — far above safe threshold"))
    elif el > 5: flags.append(("HIGH",     f"{el} existing loans — elevated"))
    cf = d["monthly_revenue"] - d["monthly_expenses"]
    if cf < 0:
        s += 20; flags.append(("CRITICAL", f"Negative cash flow: KES {cf:,.0f}/mo"))
    elif cf < d["monthly_revenue"] * 0.1:
        s += 10; flags.append(("MEDIUM", "Thin cash flow margin"))
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
    for threshold, label, color in [(75,"CRITICAL","#EF4444"),(55,"HIGH","#F97316"),(35,"MODERATE","#EAB308"),(15,"LOW","#22C55E")]:
        if score >= threshold: return label, color
    return "MINIMAL", "#00D4AA"

def kes(v):
    return f"KES {v/1e6:.1f}M" if v>=1e6 else f"KES {v/1e3:.0f}K" if v>=1e3 else f"KES {v:.0f}"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 CreditIQ Kenya")
    st.caption("SME Underwriting Engine")
    st.divider()

    page = st.radio("Page", ["📋 Assessment", "📊 Performance"], label_visibility="collapsed")
    st.divider()

    run = False
    ba = emp = mr = me = tf = el = cv = 0
    pm = ab = rph = 0
    sector = SECTORS[0]; location = LOCATIONS[0]

    if page == "📋 Assessment":
        st.caption("▸ Business Profile")
        ba       = st.number_input("Business Age (yrs)", 0, 100, 5)
        emp      = st.number_input("Employees", 1, 500, 10)
        sector   = st.selectbox("Sector", SECTORS)
        location = st.selectbox("Location", LOCATIONS)

        st.caption("▸ Financials")
        mr  = st.number_input("Monthly Revenue (KES)",  0, 10_000_000, 150_000, 5000)
        me  = st.number_input("Monthly Expenses (KES)", 0, 10_000_000, 90_000, 5000)
        pm  = st.slider("Profit Margin (%)", -50, 100, 20)
        ab  = st.number_input("Avg Bank Balance (KES)", 0, 10_000_000, 50_000, 1000)
        tf  = st.number_input("Monthly Transactions", 0, 200, 15)

        st.caption("▸ Credit History")
        rph = st.slider("Repayment Score (0=Poor, 10=Excellent)", 0, 10, 7)
        el  = st.number_input("Existing Loans", 0, 20, 1)
        cv  = st.number_input("Collateral Value (KES)", 0, 10_000_000, 200_000, 5000)

        st.markdown("")
        run = st.button("⚡ Run Assessment")

    st.divider()
    st.caption("RandomForest · 2,000 records · 12 features")

# ── HEADER ───────────────────────────────────────────────────
titles = {"📋 Assessment": "Credit Risk Assessment", "📊 Performance": "Model Performance"}
st.markdown(f"## {titles[page]}")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# PERFORMANCE PAGE
# ══════════════════════════════════════════════════════════════
if page == "📊 Performance":
    mp = Path("models/model_metrics.json")
    if not mp.exists():
        st.warning("Run `python setup.py` first to generate metrics.")
        st.stop()
    m = json.loads(mp.read_text())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Accuracy",          f"{m['accuracy']:.1%}")
    c2.metric("ROC AUC",           f"{m['roc_auc']:.3f}")
    c3.metric("CV F1",             f"{m['cv_f1_mean']:.3f}")
    c4.metric("Default Recall",    f"{m['default_recall']:.1%}")
    c5.metric("Default Precision", f"{m['default_precision']:.1%}")
    c6.metric("Avg Precision",     f"{m['avg_precision']:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("ROC CURVE")
        st.line_chart(pd.DataFrame({"TPR": m["tpr"]}, index=m["fpr"]))
        st.caption("PRECISION-RECALL CURVE")
        st.line_chart(pd.DataFrame({"Precision": m["precision_curve"]}, index=m["recall_curve"]))

    with col2:
        st.caption("FEATURE IMPORTANCE")
        fi = m["feature_importance"]
        df = pd.DataFrame({"Importance": list(fi.values())}, index=[k.replace("_"," ").title() for k in fi])
        st.bar_chart(df.sort_values("Importance"))

        st.caption("CONFUSION MATRIX")
        cm = m["confusion_matrix"]
        ca, cb = st.columns(2)
        ca.metric("True Negative",  cm[0][0])
        cb.metric("False Positive", cm[0][1])
        ca.metric("False Negative", cm[1][0])
        cb.metric("True Positive",  cm[1][1])
    st.stop()

# ══════════════════════════════════════════════════════════════
# ASSESSMENT PAGE
# ══════════════════════════════════════════════════════════════
if not run:
    col1, col2, col3 = st.columns(3)
    for col, title, desc in [
        (col1, "🌲 RandomForest Engine",    "100-tree ensemble trained on 2,000 Kenya SME records with class-balanced sampling."),
        (col2, "⚖️ Dual-Layer Scoring",     "ML probability + rule-based financial score for explainable credit decisions."),
        (col3, "🗺️ 5 Counties · 5 Sectors", "Nairobi, Mombasa, Kisumu, Nakuru, Eldoret · Retail, Agri, Manufacturing, Services, Tech."),
    ]:
        with col:
            st.info(f"**{title}**\n\n{desc}")
    st.markdown("*← Fill in the SME profile on the left and click ⚡ Run Assessment*")
    st.stop()

# ── Build input & predict ─────────────────────────────────────
inp = {
    "business_age": ba, "employees": emp,
    "sector": SECTOR_MAP[sector], "location": LOCATION_MAP[location],
    "monthly_revenue": mr, "monthly_expenses": me,
    "profit_margin": pm, "avg_account_balance": ab,
    "transaction_frequency": tf, "loan_repayment_history": rph,
    "existing_loans": el, "collateral_value": cv,
}
df_inp        = pd.DataFrame([inp])
rs, flags     = risk_score(inp)
rb, rc        = band(rs)
ml_pred       = model.predict(df_inp)[0]
ml_proba      = model.predict_proba(df_inp)[0]
ml_def_prob   = float(ml_proba[1])
hard_override = any(s=="CRITICAL" for s,_ in flags) and (mr-me<0 or el>9 or rph<=2)
final         = 1 if (ml_pred==1 or hard_override) else 0
blended       = 0.7*ml_def_prob + 0.3*(rs/100)
cf            = mr - me
cov           = cv / max(mr*6, 1)
rid           = "CIQ-" + hashlib.md5((str(inp)+datetime.now().isoformat()).encode()).hexdigest()[:8].upper()

# ── Decision banner ───────────────────────────────────────────
if final == 1:
    st.error(f"### ✗  APPLICATION DECLINED\n\nRisk profile exceeds acceptable threshold &nbsp;·&nbsp; Default Probability: **{blended:.0%}**" +
             ("\n\n⚠️ Hard override applied due to extreme risk indicators." if hard_override else ""))
else:
    st.success(f"### ✓  APPLICATION APPROVED\n\nRisk profile within acceptable parameters &nbsp;·&nbsp; Default Probability: **{blended:.0%}**")

# ── Key metrics ───────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Risk Band",        rb)
k2.metric("Risk Score",       f"{rs:.0f} / 100")
k3.metric("Cash Flow / mo",   kes(cf), f"{cf/max(mr,1)*100:.1f}% margin")
k4.metric("Collateral Cover", f"{cov:.1f}×")
k5.metric("Repayment Score",  f"{rph} / 10")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["RISK FLAGS", "FINANCIALS", "AUDIT"])

with t1:
    if flags:
        icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}
        for sev, msg in sorted(flags, key=lambda x: ["CRITICAL","HIGH","MEDIUM"].index(x[0])):
            st.markdown(f"{icons[sev]} **{sev}** — {msg}")
    else:
        st.success("No risk flags detected.")

with t2:
    fa, fb = st.columns(2)
    with fa:
        st.caption("FINANCIAL HEALTH")
        for lbl, val in [("Revenue",kes(mr)),("Expenses",kes(me)),("Net Cash Flow",kes(cf)),
                         ("Profit Margin",f"{pm:.1f}%"),("Bank Balance",kes(ab)),("Collateral",kes(cv))]:
            st.markdown(f"**{lbl}:** {val}")
    with fb:
        st.caption("BUSINESS PROFILE")
        for lbl, val in [("Sector",sector),("Location",location),("Business Age",f"{ba} yrs"),
                         ("Employees",str(emp)),("Transactions",f"{tf}/mo"),("Existing Loans",str(el))]:
            st.markdown(f"**{lbl}:** {val}")

with t3:
    for k, v in {"Reference":rid,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                 "ML Default Prob":f"{ml_def_prob:.4f}","Rule Score":f"{rs:.1f}/100",
                 "Blended Prob":f"{blended:.4f}","Hard Override":str(hard_override),
                 "Decision":"DECLINED" if final==1 else "APPROVED"}.items():
        st.markdown(f"`{k}:` {v}")
    st.dataframe(df_inp, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("CreditIQ Kenya · For institutional use only · Not a substitute for full credit due diligence")
