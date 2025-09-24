import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn import tree

# Veri oku
df = pd.read_csv("diabetes_data_upload.csv")


df = df.replace({'Yes': 1, 'No': 0})
df['class'] = df['class'].replace({'Negative': 0, 'Positive': 1})



x = df[['Polyuria', 'Polydipsia', 'weakness', 'Genital thrush']]
y = df['class']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Model
dt = DecisionTreeClassifier(max_depth=100, random_state=42)
dt.fit(x_train, y_train)

# Cross validation
scores = cross_val_score(dt, x, y, cv=5)

# Karar ağacı görselleştir
plt.figure(figsize=(12, 8))
tree.plot_tree(dt, feature_names=x.columns, class_names=["Risk Yok", "Risk Var"], filled=True, rounded=True)
plt.show()

# Tahmin
y_pred = dt.predict(x_test)

# Metrikler
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation skorları
print("Her fold için doğruluk skoru:", scores)
print("Ortalama doğruluk skoru:", scores.mean())

# GridSearch parametreleri
paramgrid = {"max_depth": [3, 5, 7, None, 15]}

gridsearch = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=paramgrid,
    cv=5,
    scoring="accuracy"
)

gridsearch.fit(x_train, y_train)

print("En iyi parametre:", gridsearch.best_params_)
print("En iyi cross validation skoru:", gridsearch.best_score_)

# En iyi model
bestmodel = gridsearch.best_estimator_
y_pred = bestmodel.predict(x_test)

print("Best Model Classification Report:\n", classification_report(y_test, y_pred))
