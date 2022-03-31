#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
#%%
df = DataAcquisition().get_data()
#%% Displots of variables
color = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
for cor,col  in enumerate(df.columns):
    sns.displot(df[col],kind='kde',fill=True,color=color[cor])
    plt.savefig(f'plots/{col}.png')

# %%

color = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
for cor,col in enumerate(df.columns):
    fig = px.histogram(df, x=col)
    fig.update_layout(title=f'Distribuição de {col}',
                        title_x=0.5,
                        xaxis_title=col,
                        yaxis_title='Frequência')
    #Remove background
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.write_image(f'plots/{col}.png')
    fig.show()
# %%
red = df.loc[:,['Close','Adj Close']]
# %%
for col in red.columns:
    fig = px.histogram(df, x=col)
    fig.update_layout(title=f'Distribuição de {col}',
                        title_x=0.5,
                        xaxis_title=col,
                        yaxis_title='Frequência')
    #Remove background
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.write_image(f'plots/data_prep/{col}.png')
    fig.show()

# %%
import plotly.express as px
df_fe = FeatureEngineering(df).pipeline_feat_eng()
df_fe['target'] = df['Close'].shift(-1) #Close_Tomorrow
df_corr = df_fe.corr()
df_corr = df_corr[df_corr.index=='target']
fig = px.imshow(df_corr)
#Rotate the labels
fig.update_xaxes(tickangle=-45)
fig.write_image(f'plots/feature_eng/corr.png')
# %%

def make_fig(y_true,y_pred,index,model_name):
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
    go.Scatter(
    x=index,
    y=y_true,
    name='BTC Real',
    mode='markers+lines',
    marker_color='#993399',
    ), secondary_y=False)

    
    fig.add_trace(
        go.Scatter(
        x=index,
        y=y_pred,
        name='BTC Previsto',
        mode='lines',
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
            title={'text': 'Comportamento do BTC de 2018 a 2022 com Shuffle', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            )
    return fig

make_fig(df['Close'].sample(frac=1),[],[x for x in range(len(df))],model_name='BTC')


# %% Residual
# *** Gráfico de resíduos ***
import main
import numpy as np
result = main.realizar_predicao('XGBoost')
y_true = result['y_true']
y_pred = result['y_pred']
err = y_true - y_pred

fig = make_subplots(rows=1, cols=2, subplot_titles=('Resíduo', 'Predição vs Real'))

fig.add_trace(
    go.Scatter(
        x=y_true,
        y=err,
        mode='markers',
        name='Resíduo',
        marker_color='#fd5800',
    ), 
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=y_true,
        y=np.zeros_like(err),
        marker=dict(color='#000000'),
    ),
    row=1,
    col=1
)
fig.update_xaxes(title_text='Real', row=1, col=1)
fig.update_yaxes(title_text='Resíduo', row=1, col=1)

# fig.add_trace(
#     go.Scatter(
#         x=y_true,
#         y=y_pred,
#         mode='markers',
#         name='Predição'
#     ),
#     row=1,
#     col=2
# )
# fig.add_trace(
#     go.Scatter(
#         x=y_true,
#         y=y_true,
#         marker=dict(color='red'),
#     ),
#     row=1,
#     col=2
# )
# fig.update_xaxes(title_text='Real', row=1, col=2)
# fig.update_yaxes(title_text='Predição', row=1, col=2)
fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=100, r=0, b=50, t=50),
            height=350,
            )
fig.show()

# %% Boxplot err
import plotly.express as px
err = pd.DataFrame(err)
fig = px.box(err, x=0)
fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=100, r=0, b=50, t=50),
            height=350,
            )
fig.update_xaxes(title_text='Resíduo', row=1, col=1)        
# %% Nosso gráfico de predição para o conjunto de testes 
import main
predicao = main.realizar_predicao('XGBoost')
main.make_fig(predicao)


# %%
mae, mse, percentual = main.calcula_metrica(predicao['y_true'],predicao['y_pred'],predicao['index'])
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'Percentual do erro: {percentual}')
#%%