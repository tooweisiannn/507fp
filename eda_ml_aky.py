import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices

def load_model(file_path):
    return load(file_path)

def load_scaler(file_path):
    return load(file_path)

def predict_and_evaluate(model, X_test, y_test, scaler):
    scaled_data = scaler.fit_transform(X_test)
    predictions = model.predict(scaled_data)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2

def create_feature_importance_graph(importances_df_gb):
    fig = px.bar(importances_df_gb, x='Importance', y='Feature',
                 labels={'Feature': 'Features', 'Importance': 'Importance'},
                 orientation='h')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_correlation_heatmap(dataframe, excluded_columns = ['YearStart', 'LocationAbbr','Unnamed: 0']):
    feature_importance_labels = {
    'Adult Binge Prev': "Binge Drinking",
    'Adult Checkup': "Regular Medical Visits",
    'Adult Obesity': "Obesity",
    'Adult Smoking': "Smoking",
    'Adults No Ins': "Insurance",
    'Below Poverty Line': "Poverty",
    'HS Completion 18-24': "High School Completion",
    'Low Fruit Intake Adults': "Low Fruit Intake",
    'Low Veg Intake Adults': "Low Vegetable Intake",
    'No Adult Leisure Activity': "No Physical Leisure Activity",
    'No Broadband': "Broadband Internet Access",
    'Per Capita Alcohol': "Alcohol Consumption",
    'Unemployment Rate': "Unemployment"
}



    # Filter the DataFrame to exclude specific columns and rename the columns
    filtered_df_corr = dataframe.drop(columns=excluded_columns)
    filtered_df_corr.rename(columns=feature_importance_labels, inplace=True)
    filtered_df_corr.rename(index=feature_importance_labels, inplace=True)
    # Calculate the correlation matrix
    corr_matrix = filtered_df_corr.corr()

    # Convert the correlation matrix to a numpy array and round to two decimals
    corr_matrix_values = corr_matrix.values.round(2)
    
    # Convert annotation text to a list of lists (strings) for Plotly
    annotation_text = corr_matrix_values.astype(str).tolist()
    
    # Create a heatmap from the correlation matrix
    fig = ff.create_annotated_heatmap(
        z=corr_matrix_values,  # numpy array of values
        x=list(corr_matrix.columns),  # list of column names
        y=list(corr_matrix.index),  # list of index names
        annotation_text=annotation_text,  # list of lists of strings for text
        colorscale='Viridis',
        showscale=True
    )
    
    # Add titles and adjust the layout as needed
    fig.update_layout(
        title='Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables',
        xaxis=dict(tickangle=-45, side='bottom'),
        height=500,
        margin=dict(l=70, r=70, t=70, b=70),
    )

    return fig

