# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:06:06 2026

@author: S Jilla
"""

import pandas as pd #Pandas is for dealing with Datasets
#sklearn : Scintific Kit Learn/ SciKit Learn
from sklearn import tree #For Decissin Tree

#Read Train Data file
titanic_train = pd.read_csv("C:\\Data Science\\Titanic\\train.csv")

#os.chdir("C:/Data Science/Data/")

titanic_train.shape #No of rows and columns
titanic_train.info() # Data types and nullable columns
titanic_train.describe() #Statistical information

#Let's start the journey with non categorical and non missing data columns
X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier() #declaration
dt.fit(X_titanic_train, y_titanic_train)
#Entire logic and patterns are stored/captured in dt

#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("C:\\Data Science\\Titanic\\test.csv")
titanic_test.shape
titanic_test.info() # Data types and nullable columns
titanic_test.describe() #Statistical information

X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
import os
os.getcwd() #To get current working directory

titanic_test.to_csv("submission_Attempt1.csv", columns=['PassengerId','Survived'], index=False)
