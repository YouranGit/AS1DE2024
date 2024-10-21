import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning
import h5py

class CaliforniaHousingModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_cleaned = None
        self.df_scaled = None
        self.X = None
        self.y = None
        self.model = None
        self.y_test = None
        self.y_pred = None
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    def load_data(self):
        """Load the dataset."""
        self.df = pd.read_csv(self.data_path)
        return self.df

    def remove_outliers(self):
        """Remove outliers using the IQR method."""
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df_cleaned = self.df[~((self.df < lower_bound) | (self.df > upper_bound)).any(axis=1)]
        return self.df_cleaned
    
    def scale_data(self):
        """Scale numeric features using StandardScaler."""
        scaler = StandardScaler()
        numerical_columns = self.df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        self.df_scaled = self.df_cleaned.copy()
        self.df_scaled[numerical_columns] = scaler.fit_transform(self.df_cleaned[numerical_columns])
        self.df_scaled = self.df_cleaned
        return self.df_scaled
    
    def select_features(self):
        """Select features for training."""
        self.X = self.df_scaled.drop(['Median_House_Value'], axis=1)
        self.y = self.df_scaled['Median_House_Value']
        return self.X
    
    def train_model_rf(self):
        """Train Random Forest model."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        self.model = rf_model
        self.y_test = y_test
        self.y_pred = y_pred_rf
        
        return rf_model, y_pred_rf

    def evaluate_model(self):
        """Evaluate model performance and print MSE, MAE, and R2 scores."""
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        print(f"Random Forest - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        return {'mse': mse, 'mae': mae, 'r2': r2}

# Example usage:
if __name__ == "__main__":
    model = CaliforniaHousingModel(data_path="California_Houses.csv")
    model.load_data()
    model.remove_outliers()
    model.scale_data()
    model.select_features()
    rf_model, y_pred = model.train_model_rf()
    model.evaluate_model()

with h5py.File('random_forest_model_data.h5', 'w') as hf:
    # Save model parameters or important data
    hf.create_dataset('feature_importances', data=model.model.feature_importances_)
    hf.create_dataset('y_test', data=np.array(model.y_test))
    hf.create_dataset('y_pred', data=np.array(model.y_pred))

print("Model data saved in random_forest_model_data.h5")

