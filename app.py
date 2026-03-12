import streamlit as st
import pandas as pd
import io
from pathlib import Path
import joblib, json, hashlib, subprocess, sys
from datetime import datetime

st.set_page_config(page_title="CreditIQ Kenya", page_icon="🏦", layout="wide")

# ── Auto-train ───────────────────────────────────────────────
if not Path("models/kenya_sme_credit_model.pkl").exists():
    with st.spinner("First-time setup: training model..."):
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
INTEREST_RATE = 0.18  # 18% p.a. standard Kenya SME rate

# ── Helpers ──────────────────────────────────────────────────
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
    return f"KES {v/1e6:.1f}M" if v>=1e6 else f"KES {v/1e3:.0f}K" if v>=1e3 else f"KES {v:.0f}"

def loan_recommendation(d, final, blended, cov):
    """Calculate recommended loan amount based on capacity, collateral, and risk."""
    if final == 1:
        return 0, 0, "DECLINED", 0

    # Capacity-based ceiling
    surplus = d["monthly_revenue"] - d["monthly_expenses"]
    max_monthly_repayment = surplus * 0.40
    loan_by_capacity = max_monthly_repayment * 24  # 24-month base tenure

    # Collateral-based ceiling using tiered multipliers
    if cov >= 1.5:   coll_multiplier = 0.70
    elif cov >= 1.0: coll_multiplier = 0.50
    elif cov >= 0.5: coll_multiplier = 0.30
    elif cov >= 0.2: coll_multiplier = 0.15
    else:            coll_multiplier = 0.10

    loan_by_collateral = d["collateral_value"] * coll_multiplier

    # Risk adjustment multiplier
    rb = band(int(blended * 100))
    risk_mult = {"MINIMAL":1.0, "LOW":0.85, "MODERATE":0.65, "HIGH":0.40, "CRITICAL":0.0}[rb]

    raw = min(loan_by_capacity, loan_by_collateral)
    final_amount = min(raw * risk_mult, 5_000_000)  # cap at KES 5M

    # Suggested tenure based on risk
    tenure = {"MINIMAL":24, "LOW":18, "MODERATE":12, "HIGH":6, "CRITICAL":0}[rb]

    return final_amount, tenure, rb, max_monthly_repayment

def repayment_schedule(principal, annual_rate, months):
    """Generate monthly repayment schedule (reducing balance)."""
    if principal <= 0 or months <= 0:
        return pd.DataFrame()
    monthly_rate = annual_rate / 12
    monthly_payment = principal * (monthly_rate * (1+monthly_rate)**months) / ((1+monthly_rate)**months - 1)
    rows = []
    balance = principal
    for m in range(1, months+1):
        interest = balance * monthly_rate
        principal_paid = monthly_payment - interest
        balance -= principal_paid
        rows.append({
            "Month": m,
            "Monthly Payment (KES)": round(monthly_payment),
            "Principal (KES)": round(principal_paid),
            "Interest (KES)": round(interest),
            "Balance (KES)": round(max(balance, 0)),
        })
    return pd.DataFrame(rows)

def assess_single(row):
    """Run full assessment on a single dict/row."""
    inp = {
        "business_age": row.get("business_age", 0),
        "employees": row.get("employees", 1),
        "sector": SECTOR_MAP.get(row.get("sector","Retail"), 0),
        "location": LOCATION_MAP.get(row.get("location","Nairobi"), 0),
        "monthly_revenue": row.get("monthly_revenue", 0),
        "monthly_expenses": row.get("monthly_expenses", 0),
        "profit_margin": row.get("profit_margin", 0),
        "avg_account_balance": row.get("avg_account_balance", 0),
        "transaction_frequency": row.get("transaction_frequency", 0),
        "loan_repayment_history": row.get("loan_repayment_history", 5),
        "existing_loans": row.get("existing_loans", 0),
        "collateral_value": row.get("collateral_value", 0),
    }
    df_inp = pd.DataFrame([inp])
    rs, flags = risk_score(inp)
    rb = band(rs)
    ml_pred = model.predict(df_inp)[0]
    ml_proba = model.predict_proba(df_inp)[0]
    ml_def_prob = float(ml_proba[1])
    hard_override = any(s=="CRITICAL" for s,_ in flags) and (
        inp["monthly_revenue"]-inp["monthly_expenses"]<0 or
        inp["existing_loans"]>9 or inp["loan_repayment_history"]<=2
    )
    final = 1 if (ml_pred==1 or hard_override) else 0
    blended = 0.7*ml_def_prob + 0.3*(rs/100)
    cf = inp["monthly_revenue"] - inp["monthly_expenses"]
    cov = inp["collateral_value"] / max(inp["monthly_revenue"]*6, 1)
    loan_amt, tenure, _, _ = loan_recommendation(inp, final, blended, cov)
    return {
        "decision": "DECLINED" if final==1 else "APPROVED",
        "risk_band": rb,
        "risk_score": round(rs,1),
        "default_prob": round(blended*100,1),
        "cash_flow": round(cf),
        "collateral_cover": round(cov,2),
        "recommended_loan": round(loan_amt),
        "tenure_months": tenure,
        "flags": len(flags),
        "hard_override": hard_override,
    }

# ── HEADER ───────────────────────────────────────────────────
st.title("CreditIQ Kenya — SME Risk Intelligence")
st.caption("ML-powered underwriting · RandomForest · 2,000 Kenya SME records")
st.divider()

# ── TABS ─────────────────────────────────────────────────────
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
    st.subheader("SME Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Business Profile**")
        ba       = st.number_input("Business Age (yrs)", 0, 100, 5, key="ba")
        emp      = st.number_input("Employees", 1, 500, 10, key="emp")
        sector   = st.selectbox("Sector", SECTORS, key="sector")
        location = st.selectbox("Location", LOCATIONS, key="location")

    with col2:
        st.markdown("**Financial Metrics**")
        mr  = st.number_input("Monthly Revenue (KES)",  0, 10_000_000, 150_000, 5000, key="mr")
        me  = st.number_input("Monthly Expenses (KES)", 0, 10_000_000, 90_000,  5000, key="me")
        pm  = st.slider("Profit Margin (%)", -50, 100, 20, key="pm")
        ab  = st.number_input("Avg Bank Balance (KES)", 0, 10_000_000, 50_000, 1000, key="ab")
        tf  = st.number_input("Monthly Transactions", 0, 200, 15, key="tf")

    with col3:
        st.markdown("**Credit History**")
        rph = st.slider("Repayment Score (0=Poor, 10=Excellent)", 0, 10, 7, key="rph")
        el  = st.number_input("Existing Loans", 0, 20, 1, key="el")
        cv  = st.number_input("Collateral Value (KES)", 0, 10_000_000, 200_000, 5000, key="cv")
        st.markdown("")
        st.markdown("")
        run = st.button("Run Credit Assessment", use_container_width=True, type="primary")

    if run:
        st.divider()
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
        rb            = band(rs)
        ml_pred       = model.predict(df_inp)[0]
        ml_proba      = model.predict_proba(df_inp)[0]
        ml_def_prob   = float(ml_proba[1])
        hard_override = any(s=="CRITICAL" for s,_ in flags) and (mr-me<0 or el>9 or rph<=2)
        final         = 1 if (ml_pred==1 or hard_override) else 0
        blended       = 0.7*ml_def_prob + 0.3*(rs/100)
        cf            = mr - me
        cov           = cv / max(mr*6, 1)
        rid           = "CIQ-" + hashlib.md5((str(inp)+datetime.now().isoformat()).encode()).hexdigest()[:8].upper()

        # Store in session for Loan tab
        st.session_state["last_inp"]    = inp
        st.session_state["last_final"]  = final
        st.session_state["last_blended"]= blended
        st.session_state["last_cov"]    = cov
        st.session_state["last_rid"]    = rid

        if final == 1:
            st.error(f"### APPLICATION DECLINED\nDefault Probability: **{blended:.0%}**" +
                     ("\n\nHard override: extreme risk indicators detected." if hard_override else ""))
        else:
            st.success(f"### APPLICATION APPROVED\nDefault Probability: **{blended:.0%}**")

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Risk Band",        rb)
        k2.metric("Risk Score",       f"{rs:.0f}/100")
        k3.metric("Cash Flow/mo",     kes(cf), f"{cf/max(mr,1)*100:.1f}%")
        k4.metric("Collateral Cover", f"{cov:.2f}x")
        k5.metric("Repayment",        f"{rph}/10")

        st.divider()
        r1, r2, r3 = st.tabs(["RISK FLAGS", "FINANCIALS", "AUDIT"])

        with r1:
            if flags:
                icons = {"CRITICAL":"[CRITICAL]","HIGH":"[HIGH]","MEDIUM":"[MEDIUM]"}
                for sev, msg in sorted(flags, key=lambda x: ["CRITICAL","HIGH","MEDIUM"].index(x[0])):
                    if sev == "CRITICAL":   st.error(f"**{sev}** — {msg}")
                    elif sev == "HIGH":     st.warning(f"**{sev}** — {msg}")
                    else:                   st.info(f"**{sev}** — {msg}")
            else:
                st.success("No risk flags detected.")

        with r2:
            fa, fb = st.columns(2)
            with fa:
                st.caption("FINANCIAL HEALTH")
                for lbl, val in [("Revenue",kes(mr)),("Expenses",kes(me)),("Net Cash Flow",kes(cf)),
                                 ("Profit Margin",f"{pm:.1f}%"),("Bank Balance",kes(ab)),("Collateral",kes(cv))]:
                    st.markdown(f"**{lbl}:** {val}")
            with fb:
                st.caption("BUSINESS PROFILE")
                for lbl, val in [("Sector",sector),("Location",location),("Age",f"{ba} yrs"),
                                 ("Employees",str(emp)),("Transactions",f"{tf}/mo"),("Existing Loans",str(el))]:
                    st.markdown(f"**{lbl}:** {val}")

        with r3:
            for k, v in {"Reference":rid,"Time":datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                         "ML Default Prob":f"{ml_def_prob:.4f}","Rule Score":f"{rs:.1f}/100",
                         "Blended":f"{blended:.4f}","Override":str(hard_override),
                         "Decision":"DECLINED" if final==1 else "APPROVED"}.items():
                st.markdown(f"`{k}:` {v}")
            st.dataframe(df_inp, use_container_width=True, hide_index=True)

        if final == 0:
            st.info("Go to the **Loan Recommendation** tab for the recommended loan amount and repayment schedule.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — LOAN RECOMMENDATION
# ══════════════════════════════════════════════════════════════
with tab_loan:
    if "last_inp" not in st.session_state or st.session_state.get("last_final",1) == 1:
        st.info("Run a successful (approved) Credit Assessment first, then return here.")
    else:
        inp     = st.session_state["last_inp"]
        final   = st.session_state["last_final"]
        blended = st.session_state["last_blended"]
        cov     = st.session_state["last_cov"]
        rid     = st.session_state["last_rid"]

        loan_amt, tenure, rb, max_repayment = loan_recommendation(inp, final, blended, cov)

        st.subheader("Loan Recommendation")
        st.caption(f"Reference: {rid}")
        st.divider()

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Recommended Loan", kes(loan_amt))
        l2.metric("Tenure",           f"{tenure} months")
        l3.metric("Collateral Cover", f"{cov:.2f}x")
        l4.metric("Risk Band",        rb)

        st.divider()
        st.markdown("**How this amount was calculated**")

        surplus = inp["monthly_revenue"] - inp["monthly_expenses"]
        cap_loan = surplus * 0.40 * 24
        if cov >= 1.5:   cm, ct = 0.70, "Above 1.5x"
        elif cov >= 1.0: cm, ct = 0.50, "1.0x – 1.5x"
        elif cov >= 0.5: cm, ct = 0.30, "0.5x – 1.0x"
        elif cov >= 0.2: cm, ct = 0.15, "0.2x – 0.5x"
        else:            cm, ct = 0.10, "Below 0.2x"
        coll_loan = inp["collateral_value"] * cm
        risk_mult = {"MINIMAL":1.0,"LOW":0.85,"MODERATE":0.65,"HIGH":0.40,"CRITICAL":0.0}[rb]

        breakdown = {
            "Monthly Surplus":              kes(surplus),
            "Max Monthly Repayment (40%)":  kes(surplus * 0.40),
            "Loan by Capacity (24mo)":      kes(cap_loan),
            "Collateral Tier":              ct,
            "Collateral Multiplier":        f"{cm:.0%}",
            "Loan by Collateral":           kes(coll_loan),
            "Binding Ceiling":              kes(min(cap_loan, coll_loan)),
            "Risk Multiplier":              f"{risk_mult:.0%}",
            "Final Recommended Loan":       kes(loan_amt),
        }
        for k, v in breakdown.items():
            st.markdown(f"**{k}:** {v}")

        st.divider()
        st.subheader("Repayment Schedule")

        c1, c2 = st.columns(2)
        with c1:
            tenure_input = st.selectbox("Adjust Tenure", [6,12,18,24,36], index=[6,12,18,24,36].index(min(tenure,36)), key="tenure_sel")
        with c2:
            rate_input = st.slider("Annual Interest Rate (%)", 10, 30, int(INTEREST_RATE*100), key="rate_sel")

        schedule = repayment_schedule(loan_amt, rate_input/100, tenure_input)

        if not schedule.empty:
            monthly_pmt = schedule["Monthly Payment (KES)"].iloc[0]
            total_paid  = schedule["Monthly Payment (KES)"].sum()
            total_int   = schedule["Interest (KES)"].sum()

            s1, s2, s3 = st.columns(3)
            s1.metric("Monthly Payment", kes(monthly_pmt))
            s2.metric("Total Repayable", kes(total_paid))
            s3.metric("Total Interest",  kes(total_int))

            st.dataframe(schedule, use_container_width=True, hide_index=True)

            # CSV download
            csv = schedule.to_csv(index=False)
            st.download_button("Download Repayment Schedule (CSV)", csv,
                               f"repayment_{rid}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 3 — BATCH ASSESSMENT
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Assessment")
    st.markdown("Upload a CSV of SMEs and get a scored results table instantly.")
    st.divider()

    # Template download
    template_cols = ["business_age","employees","sector","location","monthly_revenue",
                     "monthly_expenses","profit_margin","avg_account_balance",
                     "transaction_frequency","loan_repayment_history","existing_loans","collateral_value"]
    template_data = {
        "business_age":           [8,  1],
        "employees":              [25, 3],
        "sector":                 ["Technology","Retail"],
        "location":               ["Nairobi","Kisumu"],
        "monthly_revenue":        [300000, 50000],
        "monthly_expenses":       [180000, 65000],
        "profit_margin":          [35, -15],
        "avg_account_balance":    [120000, 1500],
        "transaction_frequency":  [30, 5],
        "loan_repayment_history": [9, 2],
        "existing_loans":         [1, 11],
        "collateral_value":       [400000, 10000],
    }
    template_df = pd.DataFrame(template_data)
    st.download_button("Download CSV Template", template_df.to_csv(index=False),
                       "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        missing = [c for c in template_cols if c not in df_batch.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            with st.spinner(f"Assessing {len(df_batch)} SMEs..."):
                results = []
                for _, row in df_batch.iterrows():
                    results.append(assess_single(row.to_dict()))
                results_df = pd.DataFrame(results)
                combined   = pd.concat([df_batch.reset_index(drop=True), results_df], axis=1)

            st.success(f"Done. {len(combined)} SMEs assessed.")

            approved = (results_df["decision"]=="APPROVED").sum()
            declined = (results_df["decision"]=="DECLINED").sum()
            total_loan = results_df["recommended_loan"].sum()

            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Total SMEs",     len(combined))
            b2.metric("Approved",       approved)
            b3.metric("Declined",       declined)
            b4.metric("Total Loan Book",kes(total_loan))

            st.divider()
            st.dataframe(combined, use_container_width=True, hide_index=True)

            st.download_button("Download Full Results (CSV)",
                               combined.to_csv(index=False),
                               "batch_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab_perf:
    mp = Path("models/model_metrics.json")
    if not mp.exists():
        st.warning("Run `python setup.py` to generate metrics.")
        st.stop()

    m = json.loads(mp.read_text())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Accuracy",       f"{m['accuracy']:.1%}")
    c2.metric("ROC AUC",        f"{m['roc_auc']:.3f}")
    c3.metric("CV F1",          f"{m['cv_f1_mean']:.3f}")
    c4.metric("Default Recall", f"{m['default_recall']:.1%}")
    c5.metric("Default Prec",   f"{m['default_precision']:.1%}")
    c6.metric("Avg Precision",  f"{m['avg_precision']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.caption("ROC CURVE")
        st.line_chart(pd.DataFrame({"TPR": m["tpr"]}, index=m["fpr"]))
        st.caption("PRECISION-RECALL CURVE")
        st.line_chart(pd.DataFrame({"Precision": m["precision_curve"]}, index=m["recall_curve"]))

    with col2:
        st.caption("FEATURE IMPORTANCE")
        fi = m["feature_importance"]
        st.bar_chart(pd.DataFrame({"Importance": list(fi.values())},
                     index=[k.replace("_"," ").title() for k in fi]).sort_values("Importance"))
        st.caption("CONFUSION MATRIX")
        cm = m["confusion_matrix"]
        ca, cb = st.columns(2)
        ca.metric("True Negative",  cm[0][0])
        cb.metric("False Positive", cm[0][1])
        ca.metric("False Negative", cm[1][0])
        cb.metric("True Positive",  cm[1][1])

st.divider()
st.caption("CreditIQ Kenya · For institutional use only · Not a substitute for full credit due diligence")
