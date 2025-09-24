import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSV oku
df = pd.read_csv("new_york_listings_2024.csv", encoding="ISO-8859-9")

# Değişkenler
X = df[["latitude"]]  # Bağımsız değişken: enlem
y = df[["price"]]     # Bağımlı değişken: fiyat

# Train/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Skorlar
print("Eğitim R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

# Tahminler
y_pred = model.predict(X_test)

# Hata metrikleri
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# Örnek tahmin: Enlem verip fiyat tahmini
example_latitude = 40.7128  # Örneğin New York City
predicted_price = model.predict([[example_latitude]])
print(f"Enlem {example_latitude} için tahmin edilen fiyat:", predicted_price[0][0])

