# utils/constants.py

SAMPLE_CSV = """Category,Bank Account,Subcategory,Amount
Income 1,Utilities,Bank3,100
Income 1,Travel,Bank1,100
Income 2,Saving 4,Bank2,100
Income 2,Saving 5,Bank2,100
"""

DEFAULT_HEADERS = ["Category", "Bank Account", "Subcategory", "Amount"]

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
