# Kenya Small and Medium Enterprises Credit Scoring Model

## Problem Definition
 
Most Kenyan Small and Medium Enterprises (SMEs) struggle to access credit because lenders rely on collateral-based assessments and lack reliable data-driven tools to evaluate business risk. As a result, many viable businesses are denied funding, while lenders face uncertainty in loan decisions.

This project solves that problem by building a machine learning–based credit scoring model that predicts SME creditworthiness using real financial and transactional data. It introduces explainable AI to ensure transparency and trust in every lending decision.


##  Objectives
- Build predictive models for SME credit risk assessment
- Implement explainable AI for transparent decision-making
- Create deployable API and dashboard for lenders

## Dataset
- Simulated Kenyan SME transaction data
- 1,000 SME records across various sectors and locations
- Features include financial metrics, business characteristics, and banking behavior

## Best Model
- **Model**: Random Forest
- **AUC Score**: 1.0000
- **Accuracy**: 0.9900
1. Clone the repository

git clone https://github.com/EmmanuelOchieng01/Small-and-medium-Enterprises-credit-scoring.git

then; 

cd Small-and-medium-Enterprises-credit-scoring

2. Install dependencies

pip install -r requirements.txt

3. Run the credit scoring model

python kenya_sme_credit.py

4. Run the dashboard

streamlit run app.py

after running the dashboard open:

http://localhost:8501


##structure 
kenya-sme-credit/
├── models/            # Saved ML models
├── data/              # Dataset files
├── reports/           # Model performance reports
├── notebooks/         # Jupyter notebooks for experiments
├── kenya_sme_credit.py # Main ML model script
├── app.py             # Streamlit dashboard
└── requirements.txt   # Project dependencies


## Key Features
- Multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- SHAP explainability for model interpretability
- Bias and fairness analysis
- API deployment ready
- Interactive dashboard

##  Contributors
Emmanuel Ochieng




# 🏦 CreditIQ Kenya — SME Credit Scoring System

ML-powered credit risk assessment platform for Kenya's small and medium enterprise sector.

---

## ⚡ Quickstart (any machine, any numpy version)

```bash
# 1. Clone the repo
git clone https://github.com/EmmanuelOchieng01/Small-and-medium-Enterprises-credit-scoring
cd Small-and-medium-Enterprises-credit-scoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (only needed once)
python setup.py

# 4. Launch the app
streamlit run app.py
```

> **Note:** `setup.py` trains the model fresh on your machine — this ensures full compatibility with your local numpy/scikit-learn versions. The app also auto-trains on first launch if no model is found.

---

## 🏗️ Architecture

### Dual-Layer Risk Engine
| Layer | Method | Weight |
|---|---|---|
| ML Model | RandomForestClassifier (100 trees, class-balanced) | 70% |
| Rules Engine | Financial logic scoring (0–100) | 30% |
| Hard Override | Extreme case detection | Overrides both |

### Features Used (12 indicators)
| Feature | Description |
|---|---|
| `business_age` | Years in operation |
| `employees` | Headcount |
| `sector` | Industry (Retail / Manufacturing / Agriculture / Services / Tech) |
| `location` | County (Nairobi / Mombasa / Kisumu / Nakuru / Eldoret) |
| `monthly_revenue` | Gross monthly revenue (KES) |
| `monthly_expenses` | Total monthly expenses (KES) |
| `profit_margin` | Net profit margin (%) |
| `avg_account_balance` | Average bank balance (KES) |
| `transaction_frequency` | Monthly banking transactions |
| `loan_repayment_history` | Score 0–10 (10 = perfect) |
| `existing_loans` | Number of active loans |
| `collateral_value` | Value of pledged assets (KES) |

---

## 🛡️ Risk Framework

**CRITICAL** — Immediate decline triggers:
- Loan repayment history ≤ 2/10
- Existing loans > 9
- Negative monthly cash flow

**HIGH** — Significant risk factors:
- Below-average repayment history
- Negative profit margin
- Critically low account balance

**MEDIUM** — Monitored risk factors:
- Thin cash flow margin
- Insufficient collateral coverage
- Early-stage business (< 2 years)

---

## 📁 Project Structure

```
├── app.py                  # Streamlit app (self-healing)
├── setup.py                # One-time model training script
├── generate_data.py        # Synthetic dataset generator
├── requirements.txt        # Unpinned dependencies (version-agnostic)
├── data/
│   └── kenya_sme_dataset.csv
├── models/                 # Auto-generated (not committed to git)
│   ├── kenya_sme_credit_model.pkl
│   └── feature_columns.pkl
└── reports/
```

---

## 🔬 Model Performance

Trained on 2,000 synthetic Kenya SME records with realistic default logic (~30% default rate). Evaluated with stratified k-fold cross-validation.

---

*Built by Emmanuel Ochieng · Kenya SME Credit Scoring System*

##  License
MIT License
