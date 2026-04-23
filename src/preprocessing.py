
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Columns with invalid zeros
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with NaN
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Fill missing values with median
    for col in cols_with_zero:
        df[col].fillna(df[col].median(), inplace=True)

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler