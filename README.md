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
## Quick Start

1. Install dependencies

pip install -r requirements.txt

2. Run the credit scoring model

python kenya_sme_credit.py

3. Run the dashboard

streamlit run app.py

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

##  License
MIT License
