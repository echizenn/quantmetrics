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

def optimize(ln_likelyhood):
    return


def logistic_ln_likelyhood(X, y, b):
    """
    bはdummy化したXの列と同じ長さのnp.arrayを想定
    """
    ln_likelyhood = 0
    for X_i, y_i in zip(X.itertuples(name=None), y):
        if y_i:
            ln_likelyhood += math.log(math.exp(np.dot(X_i, b))/(1+math.exp(np.dot(X_i, b))))
        else:
            ln_likelyhood += 1-math.log(math.exp(np.dot(X_i, b))/(1+math.exp(np.dot(X_i, b))))
    return - ln_likelyhood

def ln_likelyhood_func(X, y):
    return functools.partial(logistic_ln_likelyhood, X=X, y=y)


def logistic(X, y):
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
    print(X_dummy.head())

    ln_likelyhood = ln_likelyhood(X_dummy, y)


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
    print('LR chi2(' + str(degree_of_freedom) + ') = ' + str(lr_chi2))
    print('Prob > chi2 = ' + str(prob_chi2))
    print('Pseudo R2 = ' + str(pseudo_r2))
    print('Odds Ratio = ' + str(odds_ratio))
    print('Std. Err. = ' + str(std_err))
    print('z = ' + str(z))
    print('p_z = ' + str(p_z))
    print('95% Conf. Interval bottom = ' + str(conf_interval_bottom))
    print('95% Conf. Interval top = ' + str(conf_interval_top))

def cmmixlogit():
    "Mixed logit regression"
    print('Hallo World!')
    return
