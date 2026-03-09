"""
Kenya SME Credit Scoring - Realistic Dataset Generator
Generates data where defaults are driven by logical financial rules
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 2000

# --- Base features ---
business_age           = np.random.randint(1, 30, N)
employees              = np.random.randint(1, 100, N)
sectors                = np.random.choice(["Retail", "Manufacturing", "Agriculture", "Services", "Technology"], N)
locations              = np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"], N)
monthly_revenue        = np.random.uniform(10000, 500000, N)
monthly_expenses       = monthly_revenue * np.random.uniform(0.4, 1.3, N)  # some are loss-making
profit_margin          = ((monthly_revenue - monthly_expenses) / monthly_revenue * 100).clip(-50, 60)
avg_account_balance    = np.random.uniform(1000, 300000, N)
transaction_frequency  = np.random.randint(1, 60, N)
loan_repayment_history = np.random.randint(0, 11, N)   # 0=very bad, 10=perfect
existing_loans         = np.random.randint(0, 10, N)
collateral_value       = np.random.uniform(0, 500000, N)

# --- Credit default: driven by realistic financial logic ---
# Each factor contributes a risk score
risk = np.zeros(N)

# Poor repayment history → big risk
risk += (10 - loan_repayment_history) * 0.25

# Negative cash flow → risk
cash_flow = monthly_revenue - monthly_expenses
risk += np.where(cash_flow < 0, 2.0, 0)
risk += np.where(profit_margin < 0, 1.5, 0)
risk += np.where(profit_margin < 10, 0.5, 0)

# Too many existing loans → risk
risk += existing_loans * 0.3

# Low collateral relative to revenue → risk
risk += np.where(collateral_value < monthly_revenue, 1.0, 0)

# Low account balance → risk
risk += np.where(avg_account_balance < 5000, 1.5, 0)

# Young business → higher risk
risk += np.where(business_age < 2, 1.0, 0)

# Low transaction frequency → risk
risk += np.where(transaction_frequency < 5, 0.5, 0)

# Add noise
risk += np.random.normal(0, 0.5, N)

# Convert risk score to binary default (target ~25-30% default rate — realistic)
threshold = np.percentile(risk, 70)
credit_default = (risk >= threshold).astype(int)

print(f"Default rate: {credit_default.mean():.1%}")
print(f"Defaults: {credit_default.sum()} | Non-defaults: {(credit_default==0).sum()}")

# --- Build dataframe ---
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
print(f"\n✅ Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nCorrelations with credit_default:")
nums = df.drop(columns=["company_id","sector","location"]).corr()["credit_default"].drop("credit_default").sort_values()
print(nums)
