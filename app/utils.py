import pandas as pd
import os

# Path to the CSV file
file_path = '.\\data\\budget.csv'

# Load the CSV data into a DataFrame
def load_budget(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(df.head())  # Display the first few rows
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    

# Prepare data for Sankey chart
def prepare_sankey_data(df):
    # Identify unique nodes (categories, subcategories, accounts)
    unique_categories = df['Category'].unique().tolist()
    unique_subcategories = df['Subcategory'].unique().tolist()
    unique_accounts = df['Bank Account'].unique().tolist()

    # Combine all unique nodes into a single list
    nodes = unique_categories + unique_subcategories + unique_accounts

    # Create source and target indices for Sankey links
    source = []
    target = []
    value = []

    # Map category -> subcategory
    for _, row in df.iterrows():
        source.append(nodes.index(row['Category']))
        target.append(nodes.index(row['Subcategory']))
        value.append(row['Amount'])

    # Map subcategory -> bank account
    for _, row in df.iterrows():
        source.append(nodes.index(row['Subcategory']))
        target.append(nodes.index(row['Bank Account']))
        value.append(row['Amount'])

    return {
        "nodes": nodes,
        "links": {
            "source": source,
            "target": target,
            "value": value
        }
    }

# Update the budget data and save to CSV
def update_and_save_budget(data, file_path):
    try:
        budget_df = pd.DataFrame(data)
        print('doing function')
        print(budget_df)
        # Validate data
        if not all(col in budget_df.columns for col in ["Category", "Subcategory", "Amount", "Bank Account"]):
            raise ValueError("Invalid data format")
        budget_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error updating budget: {e}")
    return budget_df

# Test the function
if __name__ == "__main__":
    file_path = '.\\data\\budget.csv'
    # budget = load_budget(file_path)
    # sankey_data = prepare_sankey_data(budget)

    # Save the updated data
    # save_budget(budget, file_path)
