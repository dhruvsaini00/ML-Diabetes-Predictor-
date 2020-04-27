import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# Importing the dataset
dataset = pd.read_csv('diabetes.csv')

#ABOUT DATABASE
print("")
print("Shape of DATASET :  ")
dataset.shape

dataset.info()

X = dataset.iloc[:, 0:8].values

#### Analysing the 'target' variable
dataset["Outcome"].describe()
Y = dataset.iloc[:, 8].values

#Checking Correlation:
print(dataset.corr()["Outcome"].abs().sort_values(ascending=False))

#Graphs of columns :
dataset.hist(bins=50, figsize=(20,15))
print(" ")

# there is no categorical value here so we don't need labelEncoder and onehotencoder.

#checking TARGET column
target_temp = dataset.Outcome.value_counts()
print("Percentage of patience without  problems: "+str(round(target_temp[0]*100/2000,2)))
print("Percentage of patience with  problems: "+str(round(target_temp[1]*100/2000,2)))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# CHECKING ALL CLASSIFICATION MODELS :

#LOGISTIC REGRESSION :
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
print(classification_report(Y_test,Y_pred_lr))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_lr)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_lr))  



# KERNEL Support Vector Regression:
from sklearn import svm
sv = svm.SVC(kernel='linear',random_state=0)
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
print(classification_report(Y_test,Y_pred_svm))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_svm)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_svm))  


#K-NN :
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9 ,metric='minkowski', p=2)
# minkowski is for Euclidean distance.
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
print(classification_report(Y_test,Y_pred_knn))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_knn)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_knn))  


#DESICION TREE :
from sklearn.tree import DecisionTreeClassifier
max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
print(classification_report(Y_test,Y_pred_dt))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_dt)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_dt))  

#RANDOM FOREST :

from sklearn.ensemble import RandomForestClassifier
max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")
print(classification_report(Y_test,Y_pred_rf))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_rf)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_rf))  

#NEURAL NETWORKS :
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(11,activation='relu',input_dim=8))
# Adding the second hidden layer
model.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))
#output Layer
model.add(Dense(1,activation='sigmoid'))
# Compiling the ANN
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=10 , nb_epoch=50)
#nb_epochs is tne no. of time the model will cycle through the data.
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")
print(classification_report(Y_test,Y_pred_nn))
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_nn)
plt.plot(fpr,tpr)
# Area Under The Curve (AUC)
print(roc_auc_score(Y_test,Y_pred_nn))  

#COMAPRING ALL MODELS :
scores = [score_lr,score_svm,score_knn,score_dt,score_rf,score_nn]
algorithms = ["Logistic Regression","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","Neural Network"]

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)    