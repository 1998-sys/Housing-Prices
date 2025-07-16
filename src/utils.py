import pandas as pd
import numpy as np
import seaborn as sns


def relacao_variaveis(df, x, y, title=None):
    """
    Plota um scaterplot para visualizar a relação entre duas variáveis 

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    x (str): Nome da variável para o eixo x.
    y (str): nome da variável para o eixo y.
    title (str, opcional): Título do gráfico. Se None, não será adicionado título.

    Retorna:
    Axes: Objeto Axes do matplotlib com o gráfico plotado.
    """
    ax = sns.scatterplot(data=df, x=x, y=y)
    
    if title:
        ax.set_title(title)
    
    return ax