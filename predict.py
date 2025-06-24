import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataframe
b = pd.read_csv('dataset/Bangalore  house data.csv')

# Function to convert total_sqft to float, handling ranges
def convert_sqft_to_float(x):
    try:
        return float(x)
    except ValueError:
        if isinstance(x, str) and '-' in x:
            parts = x.split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return None
        else:
            return None

# Apply conversion
b['total_sqft'] = b['total_sqft'].apply(convert_sqft_to_float)

# Convert columns to float
b['bath'] = b['bath'].astype(float)
b['balcony'] = b['balcony'].astype(float)
b['price'] = b['price'].astype(float)

# Prepare features and target
X = b.iloc[:, 5:-1]
y = b['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'bangalore_rf_model.pkl')
print("Model saved as 'bangalore_rf_model.pkl'")
