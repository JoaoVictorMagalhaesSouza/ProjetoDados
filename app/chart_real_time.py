import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
def make_fig_rt():
    data = pd.read_csv('metricas_diarias.csv')
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
    go.Scatter(
    x=data['Data'],
    y=data['Preco Real'],
    name='BTC Real',
    mode='markers+lines',
    marker_color='#000000',
    ), secondary_y=False)

    
    fig.add_trace(
        go.Scatter(
        x=data["Data"],
        y=data["Preco Previsto"].astype(float),
        name='BTC Previsto',
        mode='markers+lines',
        marker_color='#fd5800',#'#ccff33',
    ), secondary_y=False)

   

    fig.update_yaxes(
        title_text="Preço",
        
            secondary_y=False, 
            gridcolor='#d3d3d3', 
            zerolinecolor='black')

    fig.update_xaxes(
        title_text="Data",
            gridcolor='#d3d3d3', 
            zerolinecolor='black')

    fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=100, r=0, b=50, t=50),
            height=350,
            title={'text': 'Previsão do BTC - Previsão diária real', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            )
    return fig

def char_feat_importance():
    
    with open("xgb_reg.pkl", 'rb') as f:
        model = pickle.load(f)
    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    fig = px.bar(data, x="score", y=data.index, orientation='h', title='Importância das Features',
    labels={'score': 'Importância','index':'Feature'}, color='score')
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=0, b=50, t=50),
        height=800,
        width=800
    )
    return fig
