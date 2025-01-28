# Wine-dataset.csv
!pip install pandas numpy

import pandas as pd
import numpy as np

wine_data = pd.read_csv("C:\\Users\\GTX\\Desktop\\WINE CSV\\wine_dataset.csv")

print(wine_data)

type(wine_data)

print(wine_data.columns)

y=wine_data["class"]

x = wine_data.drop("class", axis=1)

x = wine_data.drop("class", axis=1)
print(x)

print(y)

y_new = np.zeros(len(wine_data))

print(y_new)

for i in range(len(y)):
    if y[i] == "1":
        y[i] = 0
    elif y[i] == "2":
        y[i] = 1
    else:
        y[i] = 2

print(y)

y_new = np.zeros(178)

print(y_new)

for i in range(len(y)):
    if y[i] == 0:
        y_new[i] = 0
    elif y[i] == 1:
        y_new[i] = 1
    else:
        y_new[i] = 2

print(y_new)

print(y)

pip install scikit-learn

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y_new, test_size=0.2)

print(xtrain.shape)

print(xtest.shape)

print(ytest)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(xtrain , ytrain)

y_predict = model.predict(xtest)

print(y_predict) 

print(ytest)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_predict, ytest))

from sklearn.metrics import classification_report 

print(classification_report(y_predict, ytest))
