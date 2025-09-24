import pandas as pd

df = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split, cross_validate
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

plt.figure(figsize =(12,8))
tree.plot_tree(dt, feature_names = x.columns, class_names = ["Sağlam" , "Kanser" ] ,filled = True , rounded = True )
plt.show()

y_pred = dt.predict(x_test)

print(accuracy_score(y_pred , y_test))
print(confusion_matrix(y_pred , y_test))
print(f1_score(y_pred , y_test))
print(classification_report(y_pred, y_test))

