from ast import increment_lineno
from cgi import test
from pickle import NONE
from random import seed
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn import datasets
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#PART 1 - IMPORT THE DATA SET

#Database creation
base_de_dados=pd.read_csv("Ftol_esf+Desv_esf+Desv_div+Temp.csv",encoding='ISO-8859-1', delimiter=";")
#z is the referencing of the dataset. We are getting all values from rows and columns, but only numbers
z=base_de_dados.iloc[:,:].values
#database variable type information
base_de_dados.info()
print(f'{base_de_dados:}')


#PART 2 - DEFINE VARIABLES (INDEPENDENT/DEPENDENT) AND DEFINE TRAINING PERCENTAGE

#x is the set of independent variables. Search only the independent variables, in this case exclude the last column.
x=base_de_dados.iloc[:,:-1].values
#y is the set with dependent variable. Search only the dependent variable, in this case it is the last column.
y=base_de_dados.iloc[:,-1].values
#Separating training and testing using the sklearn function
from sklearn.model_selection import train_test_split
#defining the variables for testing and defining the sample size for testing
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25,random_state=None)


#PART 3 - IMPLEMENTATION OF THE REGRESSION DECISION TREE MODEL

#training the model with the fit method (training model)
regressor=DecisionTreeRegressor(random_state=0,min_samples_split=2,min_samples_leaf=3, max_depth=None)
regressor.fit(xtrain, ytrain)

#k-fold cross-validation method for scaling as a method to incrementally improve model performance
kf=KFold(n_splits=5,random_state=None)
i=0
for train,test in kf.split(x):
    i=i+1
    print(f'Separação {i}:\n treino: {x[train]} \n\n teste: {x[test]}\n')

#function to calculate the MAE (MSE observation only)
MAE=cross_val_score(regressor,x,y,cv=kf,scoring='neg_mean_absolute_error')
MSE=cross_val_score(regressor,x,y,cv=kf,scoring='neg_mean_squared_error')
print('ANÁLISE MAE')
print(-MAE)
print('MAE Folds: ', -MAE.mean())
print('ANÁLISE MSE')
print(-MSE)
print('MSE Folds: ', -MSE.mean())

#Export the file for creating the tree on the dreampuf website
export_graphviz(regressor, out_file ='tree.dot', feature_names =['Temperature',"Deviation(D1)","OOT(D1)", "Deviation(D2)"])