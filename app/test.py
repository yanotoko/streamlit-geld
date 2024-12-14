import pandas as pd
from pprint import pprint
# Update the budget data and save to CSV
def update_and_save_budget(data, file_path):
    try:
        budget_df = pd.DataFrame(data)
        print(budget_df)
        # Validate data
        if not all(col in budget_df.columns for col in ["Category", "Subcategory", "Amount", "Bank Account"]):
            raise ValueError("Invalid data format")
        budget_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error updating budget: {e}")
    return budget_df

    # if request.method == 'POST':
    #     # Update the budget data using the utility function
    #     data = request.json
    #     print(data)
    #     budget_df = update_and_save_budget(data, '.\\data\\budget.csv')
    # else:
    #     # Load the budget dataframe
    #     budget_df = load_budget('.\\data\\budget.csv')

#df = pd.read_csv('.\\data\\budget.csv')
data = [{'Category': '', 'Subcategory': '', 'Amount': '', 'Bank Account': None}, {'Category': 'Income', 'Subcategory': 'Salary', 'Amount': '5000', 'Bank Account': None}, {'Category': 'Income', 'Subcategory': 'Freelance', 'Amount': '2000', 'Bank Account': None}, {'Category': 'Expenses', 'Subcategory': 'Rent', 'Amount': '1500', 'Bank Account': None}, {'Category': 'Expenses', 'Subcategory': 'Groceries', 'Amount': '400', 'Bank Account': None}, {'Category': 'Savings', 'Subcategory': 'Emergency Fund', 'Amount': '1000', 'Bank Account': None}, {'Category': 'Savings', 'Subcategory': 'Retirement Fund', 'Amount': '800', 'Bank Account': None}]
pprint(data)
print(pd.DataFrame(data))
#budget_df = update_and_save_budget(data, '..\\data\\budget.csv')

#print(budget_df.head())