import functools
import math
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize


def get_category_modes(df):
    """
    カテゴリータイプまたはオブジェクトタイプの列の最頻値を返す関数
    """
    category_variables = df.dtypes[(df.dtypes == 'category') | (df.dtypes == 'object')].index
    category_mode = {}
    for variable in category_variables:
        category_mode[variable] = df[variable].value_counts().index[0]
    return category_mode

def _optimize(func, init_value):
    """
    最適化関数、現状はscipy使うだけ
    """
    return optimize.minimize(func, init_value)


def logistic_ln_likelihood(b, X, y):
    """
    bはdummy化したXの列と同じ長さのnp.arrayを想定
    ln_likelihood_funcでのfunctoolsを利用する関係上変数bを先頭においている

    """
    ln_likelihood = 0
    for X_i, y_i in zip(X.itertuples(name=None), y):
        X_i = X_i[1:]
        hat_y = 1/(1+math.exp(-np.dot(X_i, b)))
        if y_i == 1 and hat_y == 1:
            continue
        if y_i == 0 and hat_y == 0:
            continue
        ln_likelihood += ((1-y_i)*math.log(1-hat_y)+y_i*math.log(hat_y))
    # print(-ln_likelihood)
    return - ln_likelihood

def ln_likelihood_func(X, y):
    return functools.partial(logistic_ln_likelihood, X=X, y=y)


def logistic(X, y, init_value):
    """
    X: explanatory variables(pandas.Dateframe)
    y: dependent variables(pandas.Series)
    defaultでは、オブジェクト型であればダミー変数化して処理する。
    数値型ならそのまま(get_dummiesの仕様と同じ)
    """
    number_of_obs = min(len(y), len(X))

    X_dummy = pd.get_dummies(X)
    category_modes = get_category_modes(X)
    for index, value in category_modes.items():
        X_dummy.drop(index + '_' + value, axis=1, inplace=True)

    if not '_cons' in X_dummy:
        X_dummy['_cons'] = 1
    print(X_dummy.head())

    ln_likelihood = ln_likelihood_func(X_dummy, y)

    result = _optimize(ln_likelihood, init_value)


    log_likelihood = -result.fun


    degree_of_freedom = 42.195
    lr_chi2 = 42.195
    prob_chi2 = 0.42195
    pseudo_r2 = 0.42195
    odds_ratio = {'age':42.195, 'lwt':42.195,}
    std_err = {'age':42.195, 'lwt':42.195,}
    z = {'age':42.195, 'lwt':42.195,}
    p_z = {'age':42.195, 'lwt':42.195,}
    conf_interval_bottom = {'age':42.195, 'lwt':42.195,}
    conf_interval_top  ={'age':42.195, 'lwt':42.195,}
    print('Logistic regression')
    print('Number of obs = ' + str(number_of_obs))
    print('Log likelihood = ' + str(round(log_likelihood, 3)))
    print('LR chi2(' + str(degree_of_freedom) + ') = ' + str(lr_chi2))
    print('Prob > chi2 = ' + str(prob_chi2))
    print('Pseudo R2 = ' + str(pseudo_r2))
    print('Odds Ratio = ' + str(odds_ratio))
    print('Std. Err. = ' + str(std_err))
    print('z = ' + str(z))
    print('p_z = ' + str(p_z))
    print('95% Conf. Interval bottom = ' + str(conf_interval_bottom))
    print('95% Conf. Interval top = ' + str(conf_interval_top))
    print(result)

def cmmixlogit():
    "Mixed logit regression"
    print('Hallo World!')
    return
