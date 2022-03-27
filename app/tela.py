import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import main
from utils.style import create_big_card
import chart_real_time as crt
import predict_real_time as prt

global dict_predict
dict_predict = main.realizar_predicao('XGBoost')
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
                dbc.NavLink("Histórico", href="/", active="exact"),
                dbc.NavLink("Desempenho prático", href='/real-time', active="exact"),
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
                        dcc.Graph(figure=main.make_fig(**dict_predict,model_name="XGBoost"))],
                        id='xgboost-chart', style={'border-color':'#fd5800', 'border-style':'solid','height':'450px', 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px'}),
                    html.Br(),
                    html.Div([
                        dbc.Col(
                            dbc.Card(
                                f"MAE: {round(main.calcula_metrica(**dict_predict)[0],2)}", color="#fd5800", inverse=True,outline=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px !important'
                        
                            }
                        
                        
                       )

                        ),
                         dbc.Col(
                        dbc.Card(
                                f"%ERRO: {round(main.calcula_metrica(**dict_predict)[2],2)}", color="#fd5800", inverse=True, outline=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold', 'left':'100px',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px !important'
                        
                            }
                        
                        
                        )
                         ),
                         dbc.Col(
                        dbc.Card(
                                f"MSE: {round(main.calcula_metrica(**dict_predict)[1],2)}", color="#fd5800", inverse=True,outline=True, style={'height':'100px', 'width':'120px', 'border-radius':'10px', 'text-align':'center', 
                        'padding':'10px', 'align-items':'center', 'justify-content':'center', 'font-size':'20px', 'font-weight':'bold', 'left':'235px',
                        'color':'#2fa4e7', 'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s', 'margin-right':'5px'
                        
                            }
                        
                        
                        )
                         ),
                         
                        ], id='xgboost-metrics', style={'border-color':'#fd5800', 'border-style':'solid', 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px', 'display':'flex', 'flex-direction':'row', 'justify-content':'center','align-items':'center'}),
                    ]

        elif pathname == '/real-time':
            return[
            html.Div([
                        dcc.Graph(figure=crt.make_fig_rt())],
                        id='xgboost-chart-rt', style={'border-color':'#fd5800', 'border-style':'solid','height':'450px', 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px'}),
            html.Br(),
            html.Div([
                dcc.Graph(figure = crt.char_feat_importance())
            ], id='feat-impt', style={'border-color':'#fd5800', 'border-style':'solid', 'width':'100%', 'border-width':'6pxpx', 'padding':'10px',
                        'box-shadow':'0px 8px 16px 0px rgba(0,0,0,0.2)', 'transition': '0.3s','border-radius':'10px', 'display':'flex', 'flex-direction':'row', 'justify-content':'center','align-items':'center'}),
            
            ]
        


        dcc.Interval(id='interval-component', interval=90000000, n_intervals=0)
        @app.callback(
            Output('xgboost-chart-rt', 'figure'),
            Output('xgboost-chart', 'figure'),
            Input('acquisition-interval', 'n_intervals')
        )
        def att_graph(n):
            prt.real_time_prediction()
            dict_predict = main.realizar_predicao('XGBoost')
            return crt.make_fig_rt()
    
    return app
