import pandas as pd
import wrangle
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)


# Get data and transform
# --------------------------------------------------
# tri-grams generation function
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


# get data
three_star = pd.read_csv("data/three_star_tf_idf_scores.csv", index_col=0)
four_star = pd.read_csv("data/four_star_tf_idf_scores.csv", index_col=0)
# combine the two dataframes together
df = pd.concat([three_star, four_star], axis=0)

# Initialize application
# --------------------------------------------------
# create a dash application
app = Dash(__name__)
# initilize app layout
app.layout = html.Div([
    # set application title
    html.H1(children='Glassdoor Companies Rating)', style={'text-align': 'center'}),

    # # create drop down selection box for rating
    #     dcc.Dropdown(id="select_rating",
    #                  options=[{"label": "3-star", "value": 3},
    #                           {"label": "4-star", "value": 4}],
    #                  multi=False,
    #                  value="3-star",
    #                  style={'width': "50%"}),
    #
    #     # create drop down selection box for n-grams
    #     dcc.Dropdown(id="select_n_grams",
    #                      options=[{"label": "Unigram", "value": "unigram"},
    #                             {"label": "bigram", "value": "bigram"},
    #                               {"label": "trigram", "value": 'trigram'}],
    #                      multi=False,
    #                      value="unigram",
    #                      style={'width': "50%"}),

    # create drop down selection box for pros and cons
    dcc.Dropdown(id="select_review_type",
                 options=[{"label": "Pros", "value": "pros"},
                          {"label": "Cons", "value": 'cons'}],
                 multi=False,
                 value="pros",
                 style={'width': "50%"}),

    html.Div(id='pros_cons_container', children=[]),
    html.Br(),

    dcc.Graph(id='pros_cons_graph', figure={})
])


# Connect the Plotly graphs with Dash Components
# -----------------------------------------------
@app.callback([Output(component_id='pros_cons_container', component_property='children'),
               Output(component_id='pros_cons_graph', component_property='figure')],
              [Input(component_id='select_review_type', component_property='value')])
def update_pros_cons_graph(select_review):
    container = f"The  review type chosen by user was: {select_review}"

    document = df.copy()
    document = document[document["doc"] == select_review].head(20)

    fig = px.bar(data_frame=document, x="word", y="count")
    return container, fig


if __name__ == "__main__":
    app.run_server(debug=True)
