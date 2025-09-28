from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
import numpy as np

X = load_iris().data

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(random_state=42))
])

param_grid = {
    "kmeans__n_clusters": list(range(2, 11)),
    "kmeans__init": ["k-means++", "random"],
    "kmeans__algorithm": ["lloyd", "elkan"]
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    pipe.set_params(**params)
    labels = pipe.fit_predict(X)
    if len(set(labels)) < 2:
        score = -1
    else:
        score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_params = params

print("En iyi parametreler:", best_params)
print("En iyi silhouette skoru:", best_score)
