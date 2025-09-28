from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),  #PCA olmadan da mumkun
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

# Dogruluk
print(accuracy_score(y_test, y_pred)) 


