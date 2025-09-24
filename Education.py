import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSV oku
df = pd.read_csv("bi.csv", encoding="ISO-8859-9")

# Değişkenler
x = df[["Age"]]         
y = df[["studyHOURS"]]

# Train/test ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# Skorlar
r2_train = model.score(x_train, y_train)
r2_test = model.score(x_test, y_test)

print("Eğitim R²:", r2_train)
print("Test R²:", r2_test)

# Tahminler
y_pred = model.predict(x_test)

# Hata metrikleri
mae = mean_absolute_error(y_test, y_pred)   # Ortalama mutlak hata
mse = mean_squared_error(y_test, y_pred)    # Ortalama karesel hata
rmse = mse ** 0.5                           # Kök ortalama karesel hata

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

print("Tahminler:\n", y_pred[:10])  # İlk 10 tahmini göster
print("Testler:\n", y_test[:10]) 

tahmin = model.predict([[34.0]])
print(tahmin)