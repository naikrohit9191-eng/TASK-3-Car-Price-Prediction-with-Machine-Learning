# Car Price Prediction using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("car_price_dataset.csv")

print("First 5 Rows:")
print(df.head())

# Encode categorical columns
brand_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

df['Brand'] = brand_encoder.fit_transform(df['Brand'])
df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Compare Actual vs Predicted
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\nPredictions:")
print(results)

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

# Predict New Car Price
new_car = pd.DataFrame({
    'Brand':[brand_encoder.transform(['Honda'])[0]],
    'Year':[2023],
    'Horsepower':[130],
    'Mileage':[18],
    'Fuel_Type':[fuel_encoder.transform(['Petrol'])[0]]
})

predicted_price = model.predict(new_car)

print("\nPredicted Price for New Car:")
print(f"₹ {predicted_price[0]:,.2f}")
