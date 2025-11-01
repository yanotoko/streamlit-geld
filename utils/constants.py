# utils/constants.py

SAMPLE_CSV = """Category,Bank Account,Subcategory,Amount,Income
Food & Groceries,Bank3,Utilities,100,Income 1
Food & Groceries,Bank1,Travel,100,Income 1
Travel,Bank2,train,100,Income 2
Housing,Bank2,internet,100,Income 2
"""

DEFAULT_HEADERS = ["Category", "Bank Account", "Subcategory", "Amount","Income"]

# Upload hardening
MAX_UPLOAD_MB = 10  # change if you like
ALLOWED_EXTS = {".csv", ".xlsx", ".xls", ".xlsm"}
ALLOWED_MIME = {
    "text/csv",
    "application/vnd.ms-excel",  # .xls
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel.sheet.macroEnabled.12",  # .xlsm
    # Some browsers use this for Excel; allow it but still require an allowed extension:
    "application/octet-stream",
}

# Lookup sheet: per-workspace
LOOKUP_SHEET_PREFIX = "IncomeMaster"   # final sheet will be: IncomeMaster__<workspace_tab>

# Lookup schema
LOOKUP_HEADERS = ["Level1", "Frequency", "Factor_per_month"]

# Optional: allowed frequency values (for UI validation/help text)
FREQUENCY_CHOICES = ["weekly", "biweekly", "semimonthly", "monthly", "quarterly", "yearly"]

# Transaction sheets
TX_SHEET_PREFIX = "Transactions"
MERCHANTS_SHEET_PREFIX = "Merchants"
