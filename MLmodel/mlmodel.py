import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"O:\uni\4th sem\Interpretable AI\class\MLmodel\city_day.csv")

df.dropna(inplace=True)

categorical_columns = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(columns=["AQI"])
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

new_data = X_test.iloc[0:1]
predicted_aqi = model.predict(new_data)

plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Regression Plot)")
plt.show()

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Predicted AQI: {predicted_aqi[0]:.2f}")
