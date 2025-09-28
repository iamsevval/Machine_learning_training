import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


df = pd.read_csv("oral_cancer_prediction_dataset.csv")

# Kategorik degiskenleri donustur
df = df.replace({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})

# X ve y ayir
x = df[['Age', 'Gender', 'Tobacco Use', 'Poor Oral Hygiene', 'Family History of Cancer', 'Difficulty Swallowing', 'White or Red Patches in Mouth']]
y = df['Oral Cancer (Diagnosis)']

# Egitim-test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

print("Ozellik Onemleri:", rf.feature_importances_)

y_pred = rf.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# GridSearch parametreleri
paramgrid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, None, 15]
}

gridsearch = GridSearchCV(
    estimator=rf,
    param_grid=paramgrid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

gridsearch.fit(x_train, y_train)

print("En iyi parametre:", gridsearch.best_params_)
print("En iyi cross validation skoru:", gridsearch.best_score_)

bestmodel = gridsearch.best_estimator_
y_pred = bestmodel.predict(x_test)

print("Best Model Classification Report:\n", classification_report(y_test, y_pred))

