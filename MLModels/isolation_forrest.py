import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest


class IsolationForrest():
    def __init__(self):
        self.model_iso_forrest = IsolationForest(contamination=0.1, random_state=42)

    def fit_model(self, df, inputs): 
        self.model_iso_forrest.fit(df[inputs])
    
    def predict(self, df, inputs):
        df["anomaly_scores"] = self.model_iso_forrest.decision_function(df[inputs])
        df['anomaly_predicted'] = self.model_iso_forrest.predict(df[inputs])
        return df