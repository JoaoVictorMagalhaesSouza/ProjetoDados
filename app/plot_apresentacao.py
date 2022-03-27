#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
#%%
df = DataAcquisition().get_data()
#%% Displots of variables
color = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
for cor,col  in enumerate(df.columns):
    sns.displot(df[col],kind='kde',fill=True,color=color[cor])
    plt.savefig(f'plots/{col}.png')

# %%
import plotly.express as px
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
            title={'text': 'Comportamento do BTC de 2018 a 2022', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            )
    return fig

make_fig(df['Close'].sample(frac=1),[],[x for x in range(len(df))],model_name='BTC')
# %%

# %%
