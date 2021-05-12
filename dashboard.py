import os
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Load Data---------------------------------------------------------------------------------------------------------
# Stocks
my_path = os.path.join(os.path.dirname(__file__), 'exportables/')
df_stock = pd.read_csv(f'{my_path}simulate_stock_teco.csv')
df_stock.fecha = pd.to_datetime(df_stock.fecha)
df_stock_cluster = df_stock.groupby(['fecha', 'cluster']).sum().reset_index()
df_stock_cram = df_stock.groupby(["fecha", "cram"])["stock"].sum().reset_index()

# Tickets
df_tickets = pd.read_csv(f'{my_path}/simulate_tickets_teco.csv')
df_tickets.fecha = pd.to_datetime(df_tickets.fecha)
df_tickets_cram = df_tickets.groupby(['fecha', 'cram'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
df_tickets_cram.columns = ["fecha", 'cram', 'suma', 'cantidad']
df_tickets_cluster = df_tickets.groupby(['fecha', 'cluster'])["Nivel_de_servicio"].agg(
    ['sum', 'count']).reset_index()
df_tickets_cluster.columns = ["fecha", 'cluster', 'suma', 'cantidad']


# Dash--------------------------------------------------------------------------------------------------------------
# List of colors
list_of_crams = list(df_stock.cram.unique())
list_of_cluster = list(df_stock.cluster.unique())
list_of_geus = list(df_stock.geu.unique())
list_of_colors = px.colors.qualitative.Dark24
dictionary_of_colors = {}

for i, cr in enumerate(list_of_crams):
    dictionary_of_colors[cr] = list_of_colors[i]

for i, cl in enumerate(list_of_cluster):
    dictionary_of_colors[cl] = list_of_colors[i]

colors = {
    'background': '#f6f6f6',
    'text': 'purple'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

"""
   html.Label('Seleccionar GEU:'),
   dcc.Dropdown(
       id='geu',
       options=[{'label': i, 'value': i} for i in list_of_geus],
       value='1'
   ),
   """

app.layout = html.Div([

    html.H1(
        children='Model vs History',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div([

        html.Div([
            # dcc.Markdown('''**Selector GEU**'''),
            dcc.Dropdown(
                id='geu',
                options=[{'label': i, 'value': i} for i in list_of_geus],
                value='',
                multi=True)
        ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),

        html.Div([
                 dcc.Markdown('''**Mostrar Total**'''),
                 dcc.Checklist(
                     id='total',
                     options=[{'label': 'Total', 'value': 'total'}],
                     value=['total'],
                     labelStyle={'display': 'inline-block'})
                 ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),


        html.Div([
                 dcc.Markdown('''**CRAM**'''),
                 dcc.Checklist(
                     id='cram',
                     options=[{'label': i, 'value': i} for i in list_of_crams],
                     value=[],
                     labelStyle={'display': 'inline-block'})
                 ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'}),

        html.Div([
                 dcc.Markdown('''**Cluster**'''),
                 dcc.Checklist(
                     id='cluster',
                     options=[{'label': i, 'value': i} for i in list_of_cluster],
                     value=[],
                     labelStyle={'display': 'inline-block'})
                 ], style={'width': '25%', 'display': 'inline-block', 'text-align': 'center'})
        ]),

    dcc.Graph(id='indicator-graphic')

    ], style={'backgroundColor': colors['background']})

"""  
dcc.Slider(
    id='year--slider',
    min=df_stock['fecha'].min(),
    max=df_stock['fecha'].max(),
    value=df_stock['fecha'].max(),
    marks={str(day): str(day) for day in df_stock['fecha'].unique()},
    step=None
)
"""


@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('geu', 'value'),
     Input('total', 'value'),
     Input('cram', 'value'),
     Input('cluster', 'value')])
def update_graph(geus_list, is_total, cram, cluster):
    # Initialize plot
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f'Evolución de Stock',
                                        f'Evolución de Nivel de Servicio Acumulado'),
                        vertical_spacing=0.12)

    # Plots---------------------------------------------------------------------------------------------------------
    # Stock plot
    if is_total == ['total']:
        if geus_list:
            subset = df_stock[df_stock.geu.isin(geus_list)]
            subset = subset.groupby(["fecha"])["stock"].sum().reset_index()
            subset.columns = ["fecha", "stock"]
        else:
            subset = df_stock.groupby(["fecha"])["stock"].sum().reset_index()
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='Stock Total',
                                 mode='lines',
                                 line_color='white',),
                      row=1, col=1)

    for cr in cram:
        if geus_list:
            subset = df_stock[df_stock.geu.isin(geus_list)]
            subset = subset.groupby(["fecha", "cram"])["stock"].sum().reset_index()
            subset = subset[subset.cram == cr]
        else:
            subset = df_stock_cram[df_stock_cram.cram == cr]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='<Stock - CRAM> ' + str(cr),
                                 mode='lines',
                                 line_color=dictionary_of_colors[cr]),
                      row=1, col=1)

    for cl in cluster:
        if geus_list:
            subset = df_stock[df_stock.geu.isin(geus_list)]
            subset = subset.groupby(["fecha", "cluster"])["stock"].sum().reset_index()
            subset = subset[subset.cluster == cl]
        else:
            subset = df_stock_cluster[df_stock_cluster.cluster == cl]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='<Stock - CRAM> ' + str(cl),
                                 mode='lines',
                                 line_color=dictionary_of_colors[cl]),
                      row=1, col=1)

    # Tickets Plot
    if is_total == ['total']:
        if geus_list:
            subset = df_tickets[df_tickets.geu.isin(geus_list)]
            subset = subset.groupby(['fecha'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
            subset.columns = ["fecha", 'suma', 'cantidad']
        else:
            subset = df_tickets.groupby(['fecha'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
            subset.columns = ["fecha", 'suma', 'cantidad']
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                 mode='lines',
                                 name='Nivel de Servicio',
                                 line_color='white'),
                      row=2, col=1)

    for cr in cram:
        if geus_list:
            subset = df_tickets[df_tickets.geu.isin(geus_list)]
            subset = subset.groupby(['fecha', 'cram'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
            subset.columns = ["fecha", 'cram', 'suma', 'cantidad']
            subset = subset[subset.cram == cr]
        else:
            subset = df_tickets_cram[df_tickets_cram.cram == cr]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                 mode='lines',
                                 name=f'<SL - CRAM> {cr}',
                                 visible=True,
                                 line_color=dictionary_of_colors[cr]),
                      row=2, col=1)

    for cl in cluster:
        if geus_list:
            subset = df_tickets[df_tickets.geu.isin(geus_list)]
            subset = subset.groupby(['fecha', 'cluster'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
            subset.columns = ["fecha", 'cluster',  'suma', 'cantidad']
            subset = subset[subset.cluster == cl]
        else:
            subset = df_tickets_cluster[df_tickets_cluster.cluster == cl]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                 mode='lines',
                                 name=f'<SL - CRAM> {cl}',
                                 visible=True,
                                 line_color=dictionary_of_colors[cl]),
                      row=2, col=1)
    """
    # Cluster Stock
    for cluster in list(df_stock.cluster.unique()):
        subset = df_stock_cluster[df_stock_cluster.cluster == cluster]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='<Stock - CLUSTER> ' + str(cluster),
                                 mode='lines',
                                 line_color=dictionary_of_colors[cluster],
                                 visible=False),
                      row=1, col=1)

    # Cluster Tickets
    for cluster in list(df_tickets_cluster.cluster.unique()):
        subset = df_tickets_cluster[df_tickets_cluster.cluster == cluster]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                 mode='lines',
                                 name=f'<SL - CLUSTER> {cluster}',
                                 visible=False,
                                 line_color=dictionary_of_colors[cluster]),
                  row=2, col=1)
    """
    # Buttons-------------------------------------------------------------------------------------------------------
    """
    button1 = dict(method='update',
                   args=[{"visible": [False,                                                # Total stock
                                      True, True, True, True, True, True, True ,            # Stock CRAMs
                                      False,                                                # Total Service Level
                                      True, True, True, True, True, True, True,             # Service Level CRAMs
                                      False, False, False, False, False,                    # Stock cluster
                                      False, False, False, False, False]}],                 # Service Level cluster
                   label="<CRAM>")


    button2 = dict(method='update',
                   args=[{"visible": [False,
                                      False, False, False, False, False, False, False ,
                                      False,
                                      False, False, False, False, False, False, False,
                                      True, True, True, True, True,
                                      True, True, True, True, True]}],
                   label="<CLUSTER>")


    button3 = dict(method='update',
                   args=[{"visible": [True,
                                      False, False, False, False, False, False, False ,
                                      True,
                                      False, False, False, False, False, False, False,
                                      False, False, False, False, False,
                                      False, False, False, False, False]}],
                   label="<TOTAL>")
    """
    # LayOut-------------------------------------------------------------------------------------------------------
    # Update Layout
    """
    fig.update_layout(
        updatemenus=[dict(type='buttons',
                          buttons=[button1, button2, button3],
                          x=1.05,
                          xanchor="left",
                          y=0.2,
                          yanchor="top")])
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    """

    fig.update_yaxes(title_text="Stock", row=1, col=1)

    fig.update_xaxes(title_text="Month",
                     row=2, col=1)
    fig.update_yaxes(title_text="Service Level", row=2, col=1)

    fig.update_layout(template="plotly_dark",
                      margin=dict(b=0, t=30,))
    # plotly.offline.plot(fig, filename=f'history_new_scope.html')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
