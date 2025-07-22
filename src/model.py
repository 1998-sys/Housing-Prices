import statsmodels.api as sm


def modelo_reg(formula, df):
    """
    Cria um modelo de regressão 

    Parâmetros:
    variaveis (str): String com as variáveis independentes separadas por '+'.
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    model: Objeto do modelo de regressão linear ajustado.
    """
    model = sm.formula.ols(formula, df).fit()
    
    return model
