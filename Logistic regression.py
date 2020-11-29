# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:29:48 2020

@author: ld
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#load the breast cancer dataset
cancer_data = load_breast_cancer()

#print(cancer_data.keys())

#Print the features names
#print(cancer_data['feature_names'])

#Create a DataFrame with our data and features as columns
df = pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])

#insert the target to the DataFrame
df['target'] = cancer_data['target']

#build our feature matrix and target array
X = df[cancer_data['feature_names']].values
y = df['target'].values

#create a logistic regression object and build the model
model = LogisticRegression(solver='liblinear')
model.fit(X,y)

#print the predictions
print("prediction for datapoint 0:", model.predict([X[0]]))
print(model.score(X, y))

y_pred = model.predict(X)

#print the accuracy, precision, recall and f1 score
#Precision is the percent of the model’s positive predictions that are correct.
#Recall is the percent of positive cases that the model predicts correctly.
#f1 is the average of precision and recall

#The sensitivity is the true positive rate
#The specificity is the true negative rate

#the ROC curve is the graph of the sensitivity versus the specificity


print("accuracy: ",accuracy_score(y,y_pred))
print("precision :",precision_score(y,y_pred))
print("recall: ",recall_score(y,y_pred))
print("f1_score :",f1_score(y,y_pred))

#to avoid overfitting we split our dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.6,random_state=24)

#compute the sensitivity and specificity

#construct the ROC curve
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()


#compute the AUC ( area under the ROC curve) the higher the better
auc = roc_auc_score(y_test, y_pred_proba[:,1])
print("the AUC for this model is : {0}".format(auc))








