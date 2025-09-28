import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)
df = pd.read_csv("oral_cancer_prediction_dataset.csv")

df = df.replace({'Yes': 1, 'No': 0})

x = df[['Tobacco Use', 'Age', 'Tumor Size (cm)', 'Betel Quid Use', 'Family History of Cancer', 'Unexplained Bleeding']]
y = df['Oral Cancer (Diagnosis)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nPredictions:", y_pred)
print("Actual:", y_test.values)
#Sonucunda 2 boyutu tek boyuta indirgedik.