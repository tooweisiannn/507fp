import pandas as pd
from catboost import CatBoostRegressor

# Load the trained CatBoost model from the file

def update_catboost(year, location, stratification):

    catboost_model = CatBoostRegressor()
    catboost_model.load_model('catboost_model.cbm', format='cbm')

    # Prepare the new data (ensure it has the same columns and data types as your training set)
    new_data = pd.DataFrame({
        'Stratification1': [stratification],
        'LocationDesc': [location],
        'YearStart': [year]
    })

    # Ensure categorical data is encoded the same way
    new_data['Stratification1'] = new_data['Stratification1'].astype('category')
    new_data['LocationDesc'] = new_data['LocationDesc'].astype('category')

    # Make predictions with the trained model
    predictions = catboost_model.predict(new_data)

    return predictions
