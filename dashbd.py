# Import necessary libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# Import your custom module 'wrangle' (assuming it contains relevant data preprocessing functions)

# --------------------------------------------------
# Data Preparation

# Define a function to generate trigrams from lemmatized text
def generate_trigrams(lemmatized):
    words = lemmatized.split()
    trigrams = []

    if len(words) < 3:
        return trigrams

    for i in range(len(words) - 1):
        if len(words[i]) > 1 and len(words[i + 1]) > 1:
            trigram = " ".join(words[i:i + 3])
            trigrams.append(trigram)

    return trigrams


# Read data from CSV files
three_star = pd.read_csv("data/three_star_tf_idf_scores.csv", index_col=0)
four_star = pd.read_csv("data/four_star_tf_idf_scores.csv", index_col=0)

# Combine the two dataframes together
df = pd.concat([three_star, four_star], axis=0)

# --------------------------------------------------
# Initialize the Dash application

# Create a Dash application instance
app = Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    # Set application title
    html.H1(children='Glassdoor Companies Rating', style={'text-align': 'center'}),

    # Dropdown for selecting the rating (3-star or 4-star)
    dcc.Dropdown(id="select_rating",
                 options=[{"label": "3-star", "value": 3},
                          {"label": "4-star", "value": 4}],
                 multi=False,
                 value=3,
                 style={'width': "50%"}),

    # Dropdown for selecting review type (Pros or Cons)
    dcc.Dropdown(id="select_review_type",
                 options=[{"label": "Pros", "value": "pros"},
                          {"label": "Cons", "value": 'cons'}],
                 multi=False,
                 value="pros",
                 style={'width': "50%"}),

    # Container for displaying trigram word count data
    html.Div(id='trigram_value_count_container', children=[]),
    html.Br(),

    # Graph for displaying the trigram word count
    dcc.Graph(id='trigram_value_count_graph', figure={})
])


# --------------------------------------------------
# Define callback function to update the trigram word count graph

@app.callback([Output(component_id='trigram_value_count_container', component_property='children'),
               Output(component_id='trigram_value_count_graph', component_property='figure')],
              [Input(component_id='select_review_type', component_property='value'),
               Input(component_id='select_rating', component_property='value')])
def update_trigram_value_count_graph(select_review, select_rating):
    # Copy the original dataframe
    document = df.copy()

    # Filter data based on selected review type and rating
    document = document[(document["doc"] == select_review) & (document["rating"] == select_rating)].head(5)

    # Create a bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=document["count"],
        y=document["word"],
        orientation='h',
        marker=dict(color='green'),
    ))

    # Update the layout of the bar chart
    fig.update_layout(
        title="Trigram Frequency by Review Type and Star Rating",
        xaxis_title="Count",
        yaxis_title="Word",
        yaxis_categoryorder="total ascending",
    )

    # Return an empty list for the first output and the figure for the second output
    return [], fig


# --------------------------------------------------
# Run the Dash application

if __name__ == "__main__":
    app.run_server(debug=True)
