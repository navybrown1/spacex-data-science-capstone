import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Load the dataset
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Launch site options for dropdown
site_options = [{'label': 'All Sites', 'value': 'ALL'}]
site_options += [{'label': site, 'value': site} for site in spacex_df['Launch Site'].unique()]

# Initialize dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1(
        "SpaceX Launch Records Dashboard",
        style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        options=site_options,
        value='ALL',
        placeholder="Select a Launch Site here",
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart'),
    html.Br(),
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(
        id='payload-slider',
        min=int(min_payload),
        max=int(max_payload),
        step=1000,
        marks={int(min_payload): str(int(min_payload)), int(max_payload): str(int(max_payload))},
        value=[int(min_payload), int(max_payload)],
    ),
    html.Br(),
    dcc.Graph(id='success-payload-scatter-chart')
])

# Pie chart callback
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        fig = px.pie(
            spacex_df,
            names='Launch Site',
            values='class',
            title='Total Successful Launches by Site'
        )
    else:
        site_data = spacex_df[spacex_df['Launch Site'] == selected_site]
        fig = px.pie(
            site_data,
            names='class',
            title=f'Success vs. Failed Launches for site {selected_site}'
        )
    return fig

# Scatter chart callback
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter(selected_site, payload_range):
    low, high = payload_range
    filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= low) & (spacex_df['Payload Mass (kg)'] <= high)]
    if selected_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == selected_site]
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title='Correlation between Payload and Success'
    )
    return fig

# Use Dash v2+ syntax for running the app
if __name__ == "__main__":
    app.run(debug=True)  # Correct for dash >=2.0
