import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
def create_big_card(title, body, bottom_text='', info_button_id=''):
    bottom = [
        html.Span(bottom_text)
    ]
    
    if len(info_button_id) != 0: 
        bottom.append(
            html.Button(
                id=info_button_id, 
                className='info-button',
                n_clicks=0)
            )

    content = [
        html.H2(title),
        body,
        html.Div([*bottom], className='card-bottom-text')
    ]
    
    return html.Div(content, className='big-card-layout')