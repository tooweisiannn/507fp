import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.othermod import betareg
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump, load

def missing_data(data):
    fig = px.imshow(data.isnull(), color_continuous_scale=px.colors.sequential.Cividis_r, range_color=[0,1])
    return fig


### LOGISITC REGRESSION

def load_depression_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    depression = df[df.Question == "Depression among adults"]

    depression = depression[["YearStart", "LocationDesc", "DataValue", "LowConfidenceLimit", "HighConfidenceLimit", "StratificationCategory1", "Stratification1"]]

    depression['LocationDesc'] = depression['LocationDesc'].astype('category')
    depression['Stratification1'] = depression['Stratification1'].astype('category')

    return depression

def fit_logistic_regression(depression):
    formula = "np.log(DataValue / (100 - DataValue)) ~ YearStart + C(LocationDesc) + C(Stratification1)"
    model = ols(formula=formula, data=depression).fit_regularized()
    return model

def predict_depression(data, model, year, location, stratification):
    new_data = pd.DataFrame({
        'YearStart': [year],
        'LocationDesc': pd.Categorical([location], categories=data['LocationDesc'].cat.categories),
        'Stratification1': pd.Categorical([stratification], categories=data['Stratification1'].cat.categories)
    })

    new_data_with_intercept = sm.add_constant(new_data, has_constant='add')

    predictions = model.predict(new_data_with_intercept)

    probabilities = 1 / (1 + np.exp(-predictions))

    return probabilities



