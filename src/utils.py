import pandas as pd
import numpy as np
import seaborn as sns
from statstests.process import stepwise
from statstests.tests import shapiro_francia


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

def shapiro_test(modelo):
    """
    Teste de normalidade dos resíduos do modelo

    Parâmetros:
    modelo: Objeto do modelo de regressão linear ajustado.

    Retorna:
    p_value: Valor p do teste de Shapiro-Wilk.
    """
    teste_sf = shapiro_francia(modelo.resid)
    teste_sf = teste_sf.items()
    method, statistic_w, statistic_z, p = teste_sf
    print('Statistics_W=%.5f, p-value=%.6f' % (statistic_w[1], p[1]))
    alpha = 0.05

    if p[1] > alpha:
        print('Não se rejeita H0 - Distribuição aderente à normalidade')
    else:
        print('Rejeita-se H0 - Distribuição não aderente à normalidade')