# import pandas as pd
# from utils.sankey import prepare_sankey_data

# def test_prepare_sankey_data_basic():
#     df = pd.DataFrame({
#         "Income": ["A","A","B"],
#         "Category": ["X","Y","X"],
#         "Bank Account": ["B1","B2","B1"],
#         "Amount": [10, 20, 30],
#     })
#     d = prepare_sankey_data(df, "Income", "Category", "Bank Account", "Amount")
#     assert set(d.keys()) == {"nodes", "sources", "targets", "values"}
#     assert sum(d["values"]) == 10+20+10+20+30  # two layers combined

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
gc = gspread.authorize(creds)
ss = gc.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
print("OK. Worksheets:", [w.title for w in ss.worksheets()])
