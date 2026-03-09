"""
Kenya SME Credit Scoring - Realistic Dataset Generator
Run this once to generate training data, or it will be called automatically by setup.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate(n=2000, seed=42):
    np.random.seed(seed)
    N = n

    business_age           = np.random.randint(1, 30, N)
    employees              = np.random.randint(1, 100, N)
    sectors                = np.random.choice(["Retail", "Manufacturing", "Agriculture", "Services", "Technology"], N)
    locations              = np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"], N)
    monthly_revenue        = np.random.uniform(10000, 500000, N)
    monthly_expenses       = monthly_revenue * np.random.uniform(0.4, 1.3, N)
    profit_margin          = ((monthly_revenue - monthly_expenses) / monthly_revenue * 100).clip(-50, 60)
    avg_account_balance    = np.random.uniform(1000, 300000, N)
    transaction_frequency  = np.random.randint(1, 60, N)
    loan_repayment_history = np.random.randint(0, 11, N)
    existing_loans         = np.random.randint(0, 10, N)
    collateral_value       = np.random.uniform(0, 500000, N)

    # Risk score driven by financial logic
    risk = np.zeros(N)
    cash_flow = monthly_revenue - monthly_expenses
    risk += (10 - loan_repayment_history) * 0.25
    risk += np.where(cash_flow < 0, 2.0, 0)
    risk += np.where(profit_margin < 0, 1.5, 0)
    risk += np.where(profit_margin < 10, 0.5, 0)
    risk += existing_loans * 0.3
    risk += np.where(collateral_value < monthly_revenue, 1.0, 0)
    risk += np.where(avg_account_balance < 5000, 1.5, 0)
    risk += np.where(business_age < 2, 1.0, 0)
    risk += np.where(transaction_frequency < 5, 0.5, 0)
    risk += np.random.normal(0, 0.5, N)

    threshold      = np.percentile(risk, 70)
    credit_default = (risk >= threshold).astype(int)

    df = pd.DataFrame({
        "company_id":             [f"SME{str(i).zfill(4)}" for i in range(N)],
        "business_age":           business_age.astype(int),
        "employees":              employees.astype(int),
        "sector":                 sectors,
        "location":               locations,
        "monthly_revenue":        monthly_revenue.round(2),
        "monthly_expenses":       monthly_expenses.round(2),
        "profit_margin":          profit_margin.round(2),
        "avg_account_balance":    avg_account_balance.round(2),
        "transaction_frequency":  transaction_frequency.astype(int),
        "loan_repayment_history": loan_repayment_history.astype(int),
        "existing_loans":         existing_loans.astype(int),
        "collateral_value":       collateral_value.round(2),
        "credit_default":         credit_default,
    })

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/kenya_sme_dataset.csv", index=False)
    print(f"✅ Dataset generated: {N} rows | Default rate: {credit_default.mean():.1%}")
    return df

if __name__ == "__main__":
    generate()
