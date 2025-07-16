import pandas as pd
import numpy as np
import seaborn as sns


def iqr(df, coluna):
    """
    Detecação de outliers com iqr 

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    coluna (str): Nome da variável para o cálculo.

    Retorna:
    sup: limite superior.
    inf: limite inferior
    """
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)

    iqr = Q3 - Q1
    inf= Q1 - 1.5 * iqr
    sup= Q3 + 1.5 * iqr

    return sup, inf