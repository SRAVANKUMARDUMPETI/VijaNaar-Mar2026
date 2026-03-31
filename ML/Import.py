# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:25:01 2026

@author: S Jilla
"""

import os
import pandas as pd
import joblib

#changes working directory
os.chdir("C:/Users/S Jilla/titanic")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], axis=1, inplace=False)

#Use load method to load Pickle file
#Path = 'AWSCloud/192.1.1.3'
os.getcwd()
VNLogic = joblib.load("TitanicVer2.pkl")
titanic_test['Survived'] = VNLogic.predict(X_test)
titanic_test.to_csv("submissionUsingJobLib2.csv", columns=['PassengerId','Survived'], index=False)
