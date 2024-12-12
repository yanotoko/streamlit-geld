from flask import Flask, render_template_string
from app.utils import load_budget, prepare_sankey_data
import plotly.graph_objects as go

# Create the Flask application instance
app = Flask(__name__)

# Define a simple route with a button to access the budget
@app.route('/')
def home():
    return render_template_string('''
    <html>
    <head>
        <title>Home</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Budget Visualization App!</h1>
            <a href="/budget" class="btn btn-primary">View Budget</a>
            <a href="/sankey" class="btn btn-secondary">View Sankey Chart</a>
        </div>
    </body>
    </html>
    ''')

# Define a route to display the budget dataframe
@app.route('/budget')
def display_budget():
    # Load the budget dataframe
    budget_df = load_budget('.\\data\\budget.csv')
    
    # Convert the dataframe to an HTML table
    budget_html = budget_df.to_html(classes='table table-striped', index=False)
    
    # Render the HTML table
    return render_template_string('''
    <html>
    <head>
        <title>Budget Data</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container">
            <h1>Budget Data</h1>
            {{ budget_html|safe }}
        </div>
    </body>
    </html>
    ''', budget_html=budget_html)

# Define a route to display the Sankey chart
@app.route('/sankey')
def display_sankey():
    # Load the budget dataframe
    budget_df = load_budget('.\\data\\budget.csv')
    
    # Prepare the data for the Sankey chart
    sankey_data = prepare_sankey_data(budget_df)
    
    # Create the Sankey chart
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['nodes']
        ),
        link=dict(
            source=sankey_data['links']['source'],
            target=sankey_data['links']['target'],
            value=sankey_data['links']['value']
        )
    )])
    
    # Convert the chart to HTML
    sankey_html = fig.to_html(full_html=False)
    
    # Render the Sankey chart
    return render_template_string('''
    <html>
    <head>
        <title>Sankey Chart</title>
    </head>
    <body>
        <div class="container">
            <h1>Sankey Chart</h1>
            {{ sankey_html|safe }}
        </div>
    </body>
    </html>
    ''', sankey_html=sankey_html)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
