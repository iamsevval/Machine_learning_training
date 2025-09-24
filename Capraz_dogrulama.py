import pandas as pd

df = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn import tree

df = df.replace({'Yes' : 1 , 'No' : 0 })
x = df[['Glucose' , 'Insulin' , 'BMI' , 'Age' ]]
y = df['Outcome']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.5, random_state= 42 )

dt = DecisionTreeClassifier(max_depth = 100 , random_state = 42)
dt.fit(x_train ,y_train ) 

scores = cross_val_score(dt,x, y, cv= 5)
#5 kere birbirinden farklı olmak üzere veri setinden bilgi alıp öğret.(5'e ayırdık)
#5 kere alınan veri ilk test verisi olarak alınır kalanı test verisi olarak kullanılır.
#Egitmeden verilip de olabilir.

plt.figure(figsize =(12,8))
tree.plot_tree(dt, feature_names = x.columns, class_names = ["Sağlam" , "Kanser" ] ,filled = True , rounded = True )
plt.show()

y_pred = dt.predict(x_test)

print(accuracy_score(y_pred , y_test))
print(confusion_matrix(y_pred , y_test))
print(f1_score(y_pred , y_test))
print(classification_report(y_pred, y_test))
#Çapraz doğrulama 
print("Her fold için doğruluk skoru:", scores)
print("Ortalama doğruluk scoru:", scores.mean())
#Verileri verip eğitim ve test verisi çıkart

paramgrid = {
    "max_depth" : [ 3, 5, 7, None, 15]}

gridsearch = GridSearchCV( estimator = DecisionTreeClassifier ( random_state = 42),
                          param_grid = paramgrid,
                          cv = 5,
                          scoring = "accuracy")

gridsearch.fit(x, y)

print("En iyi parametre" , gridsearch.best_params_)
print("En iyi cross validation skoru" , gridsearch.best_score_)

bestmodel = gridsearch.best_estimator_

y_pred = bestmodel.predict(x_test)

print(classification_report(y_pred, y_test))
  