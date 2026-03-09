"""
Kenya SME Credit Scoring - One-time setup script
Run this after cloning: python setup.py
Generates data, trains model, and saves evaluation report.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from pathlib import Path
import joblib
import json

print("=" * 50)
print("  CreditIQ Kenya — Setup")
print("=" * 50)

# ── Step 1: Data ──
data_path = Path("data/kenya_sme_dataset.csv")
if not data_path.exists():
    print("\n[1/3] Generating dataset...")
    from generate_data import generate
    df = generate()
else:
    df = pd.read_csv(data_path)
    print(f"\n[1/3] Dataset found: {df.shape[0]} rows")

# ── Step 2: Train ──
print("\n[2/3] Training model...")

X = df.drop(columns=["company_id", "credit_default"])
y = df["credit_default"]

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Metrics ──
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]
cm          = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_test, y_prob)
avg_prec    = average_precision_score(y_test, y_prob)
cv_scores   = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)
feat_imp    = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
report      = classification_report(y_test, y_pred, target_names=["No Default", "Default"],
                                     output_dict=True, zero_division=0)

accuracy = model.score(X_test, y_test)
print(f"    Accuracy  : {accuracy:.2%}")
print(f"    ROC AUC   : {roc_auc:.4f}")
print(f"    CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"    numpy     : {np.__version__}")
print(f"    sklearn   : {__import__('sklearn').__version__}")
print()
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"], zero_division=0))

# ── Step 3: Save model ──
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/kenya_sme_credit_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

# Save metrics as JSON for app to load
metrics = {
    "accuracy":       round(accuracy, 4),
    "roc_auc":        round(roc_auc, 4),
    "avg_precision":  round(avg_prec, 4),
    "cv_f1_mean":     round(float(cv_scores.mean()), 4),
    "cv_f1_std":      round(float(cv_scores.std()), 4),
    "default_precision": round(report["Default"]["precision"], 4),
    "default_recall":    round(report["Default"]["recall"], 4),
    "default_f1":        round(report["Default"]["f1-score"], 4),
    "no_default_precision": round(report["No Default"]["precision"], 4),
    "no_default_recall":    round(report["No Default"]["recall"], 4),
    "no_default_f1":        round(report["No Default"]["f1-score"], 4),
    "confusion_matrix":  cm.tolist(),
    "fpr":  [round(x, 4) for x in fpr.tolist()],
    "tpr":  [round(x, 4) for x in tpr.tolist()],
    "precision_curve": [round(x, 4) for x in prec.tolist()],
    "recall_curve":    [round(x, 4) for x in rec.tolist()],
    "feature_importance": {k: round(float(v), 4) for k, v in feat_imp.items()},
    "train_size": len(X_train),
    "test_size":  len(X_test),
    "n_features": len(X.columns),
    "n_estimators": 100,
}
with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Model saved to models/")

# ── Step 4: Generate HTML report ──
print("\n[3/3] Generating evaluation report...")

Path("reports").mkdir(exist_ok=True)

fi_bars = ""
colors  = ["#00D4AA","#0EA5E9","#A855F7","#F97316","#EAB308","#EF4444","#22C55E","#64748B","#EC4899","#14B8A6","#8B5CF6","#F59E0B"]
for i, (feat, imp) in enumerate(feat_imp.items()):
    pct = imp * 100
    fi_bars += f"""
    <div class="fi-row">
        <div class="fi-label">{feat.replace('_',' ').title()}</div>
        <div class="fi-bar-wrap">
            <div class="fi-bar" style="width:{pct*5:.0f}px; background:{colors[i % len(colors)]};"></div>
        </div>
        <div class="fi-val">{pct:.1f}%</div>
    </div>"""

fpr_js  = json.dumps([round(x,4) for x in fpr.tolist()])
tpr_js  = json.dumps([round(x,4) for x in tpr.tolist()])
prec_js = json.dumps([round(x,4) for x in prec.tolist()])
rec_js  = json.dumps([round(x,4) for x in rec.tolist()])
cm_data = cm.tolist()

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CreditIQ Kenya — Model Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#080C14;color:#E2E8F0;font-family:'DM Sans',sans-serif;padding:2rem;}}
.header{{border-bottom:1px solid #1E2D40;padding-bottom:1.5rem;margin-bottom:2rem;display:flex;justify-content:space-between;align-items:flex-end;}}
.header h1{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#F1F5F9;}}
.header .sub{{font-family:'DM Mono',monospace;font-size:0.65rem;color:#00D4AA;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.4rem;}}
.header .meta{{font-family:'DM Mono',monospace;font-size:0.65rem;color:#334155;text-align:right;line-height:1.8;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:1rem;margin-bottom:2rem;}}
.kpi{{background:#0D1117;border:1px solid #1E2D40;border-radius:10px;padding:1rem;}}
.kpi-label{{font-family:'DM Mono',monospace;font-size:0.6rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;}}
.kpi-value{{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:#F1F5F9;}}
.kpi-value.green{{color:#00D4AA;}} .kpi-value.blue{{color:#0EA5E9;}} .kpi-value.purple{{color:#A855F7;}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;}}
.card{{background:#0D1117;border:1px solid #1E2D40;border-radius:12px;padding:1.5rem;}}
.card-title{{font-family:'DM Mono',monospace;font-size:0.65rem;color:#00D4AA;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;}}
.cm-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;}}
.cm-cell{{border-radius:8px;padding:1.2rem;text-align:center;}}
.cm-cell .val{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;}}
.cm-cell .lbl{{font-family:'DM Mono',monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:0.3rem;}}
.fi-row{{display:flex;align-items:center;gap:0.75rem;margin-bottom:0.6rem;}}
.fi-label{{font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#94A3B8;min-width:160px;}}
.fi-bar-wrap{{flex:1;}}
.fi-bar{{height:8px;border-radius:4px;min-width:4px;}}
.fi-val{{font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;min-width:40px;text-align:right;}}
.metric-row{{display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #0F1923;}}
.metric-row .ml{{font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#475569;}}
.metric-row .mr{{font-family:'DM Mono',monospace;font-size:0.82rem;color:#94A3B8;font-weight:500;}}
.footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid #0F1923;font-family:'DM Mono',monospace;font-size:0.6rem;color:#1E3A5F;display:flex;justify-content:space-between;}}
canvas{{max-height:280px;}}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="sub">◈ CreditIQ Kenya · Model Evaluation Report</div>
    <h1>Model Performance Analysis</h1>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#475569;margin-top:0.4rem;">
      RandomForestClassifier · Kenya SME Credit Scoring · {len(X_train)+len(X_test):,} samples
    </div>
  </div>
  <div class="meta">
    ALGORITHM: RandomForestClassifier<br>
    ESTIMATORS: 100 trees<br>
    TEST SIZE: 20% stratified<br>
    CV FOLDS: 5-fold<br>
    CLASS WEIGHT: balanced
  </div>
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Accuracy</div><div class="kpi-value green">{accuracy:.1%}</div></div>
  <div class="kpi"><div class="kpi-label">ROC AUC</div><div class="kpi-value blue">{roc_auc:.4f}</div></div>
  <div class="kpi"><div class="kpi-label">CV F1 Score</div><div class="kpi-value purple">{cv_scores.mean():.4f}</div></div>
  <div class="kpi"><div class="kpi-label">Default Recall</div><div class="kpi-value green">{report['Default']['recall']:.1%}</div></div>
  <div class="kpi"><div class="kpi-label">Default Precision</div><div class="kpi-value blue">{report['Default']['precision']:.1%}</div></div>
  <div class="kpi"><div class="kpi-label">Avg Precision</div><div class="kpi-value purple">{avg_prec:.4f}</div></div>
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-title">ROC Curve</div>
    <canvas id="rocChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Precision-Recall Curve</div>
    <canvas id="prChart"></canvas>
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-title">Confusion Matrix</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#334155;margin-bottom:0.75rem;">
      Predicted →
    </div>
    <div class="cm-grid">
      <div class="cm-cell" style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);">
        <div class="val" style="color:#00D4AA;">{cm_data[0][0]}</div>
        <div class="lbl" style="color:#00D4AA;">True Negative</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#064E3B;margin-top:0.3rem;">Correctly identified no-default</div>
      </div>
      <div class="cm-cell" style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);">
        <div class="val" style="color:#EF4444;">{cm_data[0][1]}</div>
        <div class="lbl" style="color:#EF4444;">False Positive</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#7F1D1D;margin-top:0.3rem;">No-default predicted as default</div>
      </div>
      <div class="cm-cell" style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);">
        <div class="val" style="color:#F97316;">{cm_data[1][0]}</div>
        <div class="lbl" style="color:#F97316;">False Negative</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#7C2D12;margin-top:0.3rem;">Default missed by model</div>
      </div>
      <div class="cm-cell" style="background:rgba(0,212,170,0.12);border:1px solid rgba(0,212,170,0.3);">
        <div class="val" style="color:#00D4AA;">{cm_data[1][1]}</div>
        <div class="lbl" style="color:#00D4AA;">True Positive</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#064E3B;margin-top:0.3rem;">Correctly identified default</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Class Performance</div>
    {''.join([f'<div class="metric-row"><span class="ml">{k}</span><span class="mr">{v}</span></div>' for k,v in [
        ("No Default — Precision", f"{report['No Default']['precision']:.1%}"),
        ("No Default — Recall",    f"{report['No Default']['recall']:.1%}"),
        ("No Default — F1",        f"{report['No Default']['f1-score']:.1%}"),
        ("Default — Precision",    f"{report['Default']['precision']:.1%}"),
        ("Default — Recall",       f"{report['Default']['recall']:.1%}"),
        ("Default — F1",           f"{report['Default']['f1-score']:.1%}"),
        ("CV F1 Mean",             f"{cv_scores.mean():.4f}"),
        ("CV F1 Std Dev",          f"± {cv_scores.std():.4f}"),
        ("Training Samples",       f"{len(X_train):,}"),
        ("Test Samples",           f"{len(X_test):,}"),
    ]])}
  </div>
</div>

<div class="card" style="margin-bottom:1.5rem;">
  <div class="card-title">Feature Importance</div>
  {fi_bars}
</div>

<div class="footer">
  <span>CREDITIQ KENYA · SME RISK INTELLIGENCE · MODEL EVALUATION REPORT</span>
  <span>RandomForestClassifier · {len(X_train)+len(X_test):,} Kenya SME Records</span>
</div>

<script>
const rocCtx = document.getElementById('rocChart').getContext('2d');
new Chart(rocCtx, {{
  type: 'line',
  data: {{
    labels: {fpr_js},
    datasets: [
      {{ label: 'ROC Curve (AUC={roc_auc:.4f})', data: {tpr_js},
         borderColor:'#00D4AA', borderWidth:2, pointRadius:0, fill:false, tension:0.1 }},
      {{ label: 'Random', data: {fpr_js},
         borderColor:'#334155', borderWidth:1, borderDash:[5,5], pointRadius:0, fill:false }}
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{ labels:{{ color:'#64748B', font:{{ family:'DM Mono', size:10 }} }} }} }},
    scales:{{ x:{{ ticks:{{ color:'#334155', font:{{ size:9 }} }}, grid:{{ color:'#0F1923' }}, title:{{ display:true, text:'False Positive Rate', color:'#475569', font:{{ size:10 }} }} }},
              y:{{ ticks:{{ color:'#334155', font:{{ size:9 }} }}, grid:{{ color:'#0F1923' }}, title:{{ display:true, text:'True Positive Rate', color:'#475569', font:{{ size:10 }} }} }} }} }}
}});

const prCtx = document.getElementById('prChart').getContext('2d');
new Chart(prCtx, {{
  type: 'line',
  data: {{
    labels: {rec_js},
    datasets: [
      {{ label: 'PR Curve (AP={avg_prec:.4f})', data: {prec_js},
         borderColor:'#0EA5E9', borderWidth:2, pointRadius:0, fill:false, tension:0.1 }}
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{ labels:{{ color:'#64748B', font:{{ family:'DM Mono', size:10 }} }} }} }},
    scales:{{ x:{{ ticks:{{ color:'#334155', font:{{ size:9 }} }}, grid:{{ color:'#0F1923' }}, title:{{ display:true, text:'Recall', color:'#475569', font:{{ size:10 }} }} }},
              y:{{ ticks:{{ color:'#334155', font:{{ size:9 }} }}, grid:{{ color:'#0F1923' }}, title:{{ display:true, text:'Precision', color:'#475569', font:{{ size:10 }} }} }} }} }}
}});
</script>
</body>
</html>"""

with open("reports/model_evaluation.html", "w") as f:
    f.write(html)

print("✅ Report saved to reports/model_evaluation.html")
print("\n  Run the app with:")
print("      streamlit run app.py\n")
