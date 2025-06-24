import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
p=pd.read_csv('dataset/Pune house data.csv')
p1=p[['total_sqft','bath','balcony','price']]

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


p1['total_sqft'] = p1['total_sqft'].apply(convert_sqft_to_float)


p1['bath'] = p1['bath'].astype(float)
p1['balcony'] = p1['balcony'].astype(float)
p1['price'] = p1['price'].astype(float)

# Prepare features and target
X = p1.iloc[:,:-1]
y = p1['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

joblib.dump(model, 'pune_rf_model.pkl')
