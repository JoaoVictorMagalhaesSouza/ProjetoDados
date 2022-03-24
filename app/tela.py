import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import base64
import orjson
import sys
import main
from utils.style import create_big_card

def init_app(server):
    app = dash.Dash(
        __name__, 
        suppress_callback_exceptions=True,
        server=server,
        external_stylesheets=[dbc.themes.CERULEAN]
    )
    
    server = app.server

    # styling the sidebar
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # padding for the page content
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    sidebar = html.Div(
    [
        #html.H2("Bem vindo!", className="display-4"),
        #html.Img(src='data:image/png;base64,{}'.format(test_base64), style={'height':'10%',}),
        html.Hr(),
        # html.P(
        #     "Barra de Navegação", className="lead"
        # ),
        dbc.Nav(
            [
                dbc.NavLink("XGBoost", href="/", active="exact"),
                dbc.NavLink("LSTM", href="/lstm", active="exact"),
                dbc.NavLink("CatBoost", href="/catboost", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
    )
    
    content = html.Div(id="page-content", children=[], style=CONTENT_STYLE) 
    
    app.layout = html.Div([
        dcc.Location(id="url"),
        sidebar,
        content,
        
    ])

    @app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
    )
    def render_page_content(pathname):
        if pathname == "/":
            return [
                    
                    html.Div([
                        dcc.Graph(figure=main.make_fig(**main.realizar_predicao('XGBoost'),model_name="XGBoost"))],
                        id='xgboost-chart', style={'border-color':'#2fa4e7', 'border-style':'solid','height':'450px', 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px'}),
                    html.Br(),
                    html.Div([
                        dbc.Col(
                            dbc.Card(
                                f"MAE: {round(main.calcula_metrica(**main.realizar_predicao('XGBoost'))[0],2)}", color="#d3d3d3", inverse=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px !important'
                        
                            }
                        
                        
                       )

                        ),
                         dbc.Col(
                        dbc.Card(
                                f"MSE: {round(main.calcula_metrica(**main.realizar_predicao('XGBoost'))[1],2)}", color="#d3d3d3", inverse=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px !important'
                        
                            }
                        
                        
                        )
                         ),
                         dbc.Col(
                        dbc.Card(
                                f"%ERRO: {round(main.calcula_metrica(**main.realizar_predicao('XGBoost'))[2],2)}", color="#d3d3d3", inverse=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px'
                        
                            }
                        
                        
                        )
                         ),
                         
                        ], id='xgboost-metrics', style={ 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px', 'display':'flex', 'flex-direction':'row', 'justify-content':'center','align-items':'center'}),
                    ]



    
    return app
