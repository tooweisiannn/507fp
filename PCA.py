import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from statsmodels.regression.linear_model import OLS
import pickle

def save_figure_as_pickle(fig, filename):
    with open(filename, 'wb') as f:
        pickle.dump(fig, f)

def analyze_depression_clusters(csv_path, n_clusters=7, drop_columns=None):
    # Load the dataset
    disease = pd.read_csv(csv_path)
    disease = disease[disease["Question"] == "Depression among adults"]

    # Ensure all columns are consistent data types
    for col in disease.columns:
        disease[col] = disease[col].astype(str)

    # Apply label encoding for object columns and fill NaN values with a consistent type
    label_encoders = {}
    for col in disease.select_dtypes(include=["object"]).columns:
        lbl = LabelEncoder()
        disease[col] = lbl.fit_transform(disease[col].fillna("-1"))
        label_encoders[col] = lbl

    # Feature scaling with MinMaxScaler
    if drop_columns is None:
        drop_columns = [
            "StratificationCategoryID1",
            "StratificationCategory3",
            "Stratification3",
            "StratificationCategoryID3",
            "ResponseID",
            "StratificationID3",
            "Stratification2",
            "StratificationCategory2",
        ]

    X = disease.drop(drop_columns, axis=1)
    Y = disease["Stratification1"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    clf = KMeans(n_clusters=n_clusters, n_init=10)
    Xr = clf.fit_transform(X_scaled)

    cluster_labels = clf.labels_

    # Create a 3D scatter plot with Plotly
    trace = go.Scatter3d(
        x=Xr[:, 0],
        y=Xr[:, 1],
        z=Xr[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color = Y,
            colorscale="Viridis",
            opacity=0.7,
        ),
    )
    layout = go.Layout(
        title=f"Clusters of {n_clusters} KMeans",
        scene=dict(
            xaxis=dict(title="1st Dimension"),
            yaxis=dict(title="2nd Dimension"),
            zaxis=dict(title="3rd Dimension"),
        ),
        height = 800
    )
    fig = go.Figure(data=[trace], layout=layout)

    # Perform OLS regression
    res = OLS(Y, Xr).fit()

    return fig, res.summary()  # Return the Plotly graph object and OLS summary

# Example usage:
fig, summary = analyze_depression_clusters("U.S._Chronic_Disease_Indicators.csv")
save_figure_as_pickle(fig, 'figure.pickle')

