# utils/default_catalog.py
from __future__ import annotations

# Immutable defaults (read-only). You can expand safely; never modify at runtime.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Housing": ["Rent/Mortgage", "HOA", "Utilities", "Maintenance", "Property Tax", "Insurance"],
    "Transportation": ["Fuel", "Public Transit", "Rideshare", "Parking", "Insurance", "Maintenance"],
    "Groceries": ["Produce", "Pantry", "Meat/Seafood", "Beverages", "Household Supplies"],
    "Dining & Coffee": ["Restaurants", "Takeout", "Cafés", "Bars"],
    "Health": ["Insurance Premiums", "Prescriptions", "Copays", "Dental", "Vision"],
    "Personal & Family": ["Childcare", "School", "Gifts", "Subscriptions", "Clothing"],
    "Debt": ["Credit Card Payment", "Student Loan", "Auto Loan", "Personal Loan"],
    "Savings & Investing": ["Emergency Fund", "Retirement", "Brokerage", "Goal Savings"],
    "Income Adjustments": ["Reimbursements", "Refunds", "Transfers (non-expense)"],
}

# Helpful “Income by / Source” seeds (purely suggestions; user can type their own):
DEFAULT_INCOME_SOURCES: list[str] = [
    "Me", "Partner", "Household", "Side Hustle", "Company A", "Company B"
]

def flat_default_categories() -> list[str]:
    return list(DEFAULT_CATEGORIES.keys())

def defaults_for(category: str) -> list[str]:
    return DEFAULT_CATEGORIES.get(category, [])
