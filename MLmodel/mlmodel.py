import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv(r'O:\uni\4th sem\Interpretable AI\class\MLmodel\city_day.csv')

df = df.drop(["Date"], axis=1, errors="ignore")  
df = df.select_dtypes(include=[np.number])
df.fillna(df.mean(), inplace=True)

if "AQI" not in df.columns:
    raise ValueError("The column 'AQI' is missing from the dataset. Check the column names.")

X = df.drop("AQI", axis=1)
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

print("Features used for training:", X.columns)
print("Number of features:", X.shape[1])

new_data = np.array([[150, 180, 40, 1.2, 10, 50, 20, 35, 70, 2.5, 5, 15]])  

if new_data.shape[1] != X.shape[1]:
    raise ValueError(f"Expected {X.shape[1]} features, but got {new_data.shape[1]}.")

predicted_aqi = model.predict(new_data)
print(f"Predicted AQI: {predicted_aqi[0]:.2f}")

plt.scatter(y_test, predictions, color="blue")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

