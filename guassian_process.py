import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Data provided
def GP(strat, df, year):
    df = df[[str(strat)]]

    df["YearStart"] = df.index
    # Prepare Gaussian Process data
    X = df[['YearStart']].values  # Years
    y = df[str(strat)].values  # Values

    # Define a Gaussian Process model with an RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit the Gaussian Process model
    gp.fit(X, y)

    # Create a range of years to predict for plotting
    years = np.arange(2011, int(year)).reshape(-1, 1)  # From 2011 to 2023
    predictions, std_devs = gp.predict(years, return_std=True)

    # Create a plot with the original data and the Gaussian Process predictions
    fig = go.Figure()

    # Plot the original data points
    fig.add_trace(go.Scatter(x=df['YearStart'], y=df[str(strat)], mode='markers', 
                            marker=dict(size=8, color='blue'),
                            name='Original Data'))

    # Plot the Gaussian Process predictions as a line
    fig.add_trace(go.Scatter(x=years.flatten(), y=predictions.flatten(), mode='lines', 
                            line=dict(color='red'),
                            name='GP Predictions'))

    # Add confidence bands using standard deviation
    fig.add_trace(go.Scatter(x=years.flatten(), y=(predictions - std_devs).flatten(), 
                            mode='lines', line=dict(width=0), 
                            fill=None, showlegend=False))

    fig.add_trace(go.Scatter(x=years.flatten(), y=(predictions + std_devs).flatten(), 
                            mode='lines', line=dict(width=0), 
                            fill='tonexty', 
                            fillcolor='rgba(255, 0, 0, 0.3)', 
                            name='Confidence Band'))

    # Update plot layout with titles and labels
    fig.update_layout(title='Gaussian Process Predictions with Confidence Bands', 
                    xaxis_title='Year', yaxis_title='Value', 
                    template='plotly_white')

    # Show the plot
    return fig
