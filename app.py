from flask import Flask, render_template_string, request
from app.utils import load_budget, prepare_sankey_data, update_and_save_budget
import plotly.graph_objects as go
import pandas as pd

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
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Budget Visualization App!</h1>
            <a href="/budget" class="btn btn-primary">View Budget and Sankey Chart</a>
        </div>
    </body>
    </html>
    ''')

# Define a route to display the budget dataframe and Sankey chart
@app.route('/budget', methods=['GET', 'POST'])
def display_budget_and_sankey():
    budget_df = None
    if request.method == 'POST':
        # Update the budget data using the utility function
        data = request.json
        if not data:
            return{"error": "No data received"}, 400
        update_and_save_budget(data, '.\\data\\budget.csv')
        return {"success":True},200

    # Load the budget dataframe
    budget_df = load_budget('.\\data\\budget.csv')
    
    # Convert the dataframe to an editable HTML table
    budget_html = budget_df.to_html(classes='table table-striped', index=False, border=0, table_id='budgetTable')
    
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
    
    # Render the budget table and Sankey chart
    return render_template_string('''
    <html>
    <head>
        <title>Budget and Sankey Chart</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script>
            $(document).ready(function() {
                // Make table cells editable
                $('#budgetTable').on('click', 'td', function() {
                    var $cell = $(this);
                    if (!$cell.hasClass('editing')) {
                        $cell.addClass('editing');
                        var originalContent = $cell.text();
                        $cell.html('<input type="text" value="' + originalContent + '" />');
                        $cell.find('input').focus().blur(function() {
                            var newContent = $(this).val();
                            $cell.removeClass('editing').text(newContent);
                        });
                    }
                });

                // Add Row functionality
                $('#addRowButton').click(function() {
                    // Get the number of columns from the first row
                    var colCount = $('#budgetTable thead th').length || $('#budgetTable tr:first td').length;
                    var newRow = '<tr>';
                    for (var i = 0; i < colCount; i++) {
                        newRow += '<td></td>';
                    }
                    newRow += '</tr>';
                    $('#budgetTable tbody').append(newRow);
                });

                // Save updated data
                $('#saveButton').click(function() {
                    var tableData = []; // Declare the variable inside the click event
                        $('#budgetTable tr').each(function (row, tr) {
                            var rowData = {
                                "Category": $(tr).find('td:eq(0)').text().trim(),
                                "Subcategory": $(tr).find('td:eq(1)').text().trim(),
                                "Amount": $(tr).find('td:eq(2)').text().trim(),
                                "Bank Account": $(tr).find('td:eq(3)').text().trim()
                            };
                            // Add only valid rows
                            if (rowData.Category && rowData.Subcategory && rowData.Amount) {
                                tableData.push(rowData);
                            }
                        });

                        //console.log("Table Data to Send:", tableData); // Debugging to confirm data is collected

                    $.ajax({
                        url: '/budget',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify(tableData),
                        success: function(response) {
                            alert('Data saved successfully!');
                            location.reload();
                        },
                        error: function(xhr, status, error) {
                            alert('Error saving data: ' + xhr.responseJSON.error);
                        }
                    });
                });
            });
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Budget Data and Sankey Chart</h1>
            <h2>Budget Table</h2>
            <button id="addRowButton" class="btn btn-secondary mb-2">Add Row</button>
            {{ budget_html|safe }}
            <button id="saveButton" class="btn btn-success">Save</button>
            <h2>Sankey Chart</h2>
            {{ sankey_html|safe }}
        </div>
    </body>
    </html>
    ''', budget_html=budget_html, sankey_html=sankey_html)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
