from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Veri
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),  #PCA olmadan da yapılabilir
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Modeli eğit
pipe.fit(x_train, y_train)

# Tahmin
y_pred = pipe.predict(x_test)

# Doğruluk
print(accuracy_score(y_test, y_pred)) 
