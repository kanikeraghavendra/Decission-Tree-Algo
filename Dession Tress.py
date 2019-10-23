# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:42:10 20194

@author: ragha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#df1 = pd.read_csv("titanic_data.csv")
#What a "Transformer" function does?
colors =["yellow","yellow","yellow","yellow","yellow","yellow","yellow","yellow","yellow","yellow","purple","purple","purple","purple","purple","purple","purple","purple","purple","purple"]
size =["small","small","small","small","small","large","large","large","large","large","small","small","small","small","small","large","large","large","large","large"]
action =["stretch","stretch","stretch","dip","dip","stretch","stretch","stretch","dip","dip","stretch","stretch","stretch","dip","dip","stretch","stretch","stretch","dip","dip"]
age =["adult","adult","child","adult","child","adult","adult","child","adult","child","adult","adult","child","adult","child","adult","adult","child","adult","child"]
inflated =[True,True,False,False,False,True,True,False,False,False,True,True,False,False,False,True,True,False,False,False]
data =[colors,size,action,age,inflated]
df = pd.DataFrame(data={"colors":colors,"size":size,"action":action,"age":age,"inflated":inflated})
df
df=pd.get_dummies(df,drop_first=True)
x =df.iloc[:,1:]
y =df.iloc[:,0]
#x =df[["inflated"]]
#y =df[["colors","size","action","age"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.1,random_state=123)
from sklearn.tree import DecisionTreeClassifier
dct= DecisionTreeClassifier()
dct.fit(x_train,y_train)
pred =dct.predict(x_test)
p#red
#df.to_csv("Baloon_DT.csv")#to write the data
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test,pred)
cr = classification_report(y_test,pred)
print(cr)
#decision tree
dtree_gini = DecisionTreeClassifier()