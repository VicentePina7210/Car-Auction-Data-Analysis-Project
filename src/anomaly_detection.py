import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def detect_anomalies_iforest(df, features, contamination=0.001, n_estimators=100):
    """
    Detect anomalies using Isolation Forest.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[features])
    iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    labels = iforest.fit_predict(features_scaled)
    df['iforest_anomaly_label'] = labels
    anomalies = df[df['iforest_anomaly_label'] == -1]
    normal_data = df[df['iforest_anomaly_label'] == 1]
    return anomalies, normal_data

def detect_anomalies_lof(df, features, contamination=0.001, n_neighbors=20):
    """
    Detect anomalies using Local Outlier Factor.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[features])
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(features_scaled)
    df['lof_anomaly_label'] = labels
    anomalies = df[df['lof_anomaly_label'] == -1]
    normal_data = df[df['lof_anomaly_label'] == 1]
    return anomalies, normal_data