import pandas as pd

df = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = df.replace({'Yes' : 1 , 'No' : 0 })
x = df[['Glucose' , 'Insulin' , 'BMI' , 'Age' ]]
y = df['Outcome']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, random_state= 42 )

knn = KNeighborsClassifier(n_neighbors = 21)

knn.fit(x_train , y_train)

y_pred = knn.predict(x_test)

print(accuracy_score(y_pred , y_test))
print(confusion_matrix(y_pred , y_test))
print(f1_score(y_pred , y_test))
print(classification_report(y_pred, y_test))

pipe = Pipeline([
    ("Scaler" , StandardScaler()),
    ("knn" , KNeighborsClassifier()) ])

paramgrid = {
    "knn__n_neighbors" : list(range(1 , 32 , 2 )),
    "knn__weights" : ["uniform" , "distance"] , 
    "knn__algorithm" : ["auto"] }

cv = StratifiedKFold( n_splits = 5 , shuffle = True , random_state = 42 )

grid = GridSearchCV(
    estimator = pipe , 
    param_grid = paramgrid, 
    cv=cv, 
    scoring = "accuracy", 
    n_jobs = 1 )


grid.fit(x, y )

print( "EN İYİ PARAMETRE" , grid.best_params_)
print( "En iyi cross validation skoru" , grid.best_score_)

bestmodel = grid.best_estimator_

y_pred = bestmodel.predict(x_test)

print(classification_report(y_pred, y_test))
