import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df = pd.read_csv("ILPD.CSV")
df.isnull().sum()
df.dropna(inplace=True)
df
df.describe()
df.info()
#%%
sns.boxplot(x="X11",y="X7",data=df)#25,87
sns.boxplot(x="X11",y="X6",data=df)#23,61
sns.boxplot(x="X11",y="X3",data=df)#0.8,2.6
sns.boxplot(x="X11",y="X8",data=df)#5.8,7.2
#sns.boxplot(x="X11",y="X1",data=df)
#sns.boxplot(x="X11",y="X2",data=df)
sns.boxplot(x="X11",y="X4",data=df)#0.2,1.3
sns.boxplot(x="X11",y="X5",data=df)#175.5,298
sns.boxplot(x="X11",y="X9",data=df)#2.6,3.8
sns.boxplot(x="X11",y="X10",data=df)#0.7,1.1
#%%
df["X7"].describe()
#%%
df[(df["X7"]<25.00)|(df["X7"]>87.00)].count()#279 outlets
df[(df["X6"]<23.00)|(df["X6"]>61.00)].count()#281
df[(df["X5"]<175.5)|(df["X5"]>298.00)].count()#284
df[(df["X4"]<0.2)|(df["X4"]>1.3)].count()#199
df[(df["X8"]<5.8)|(df["X8"]>7.2)].count()#275
df[(df["X9"]<2.6)|(df["X9"]>7.9)].count()#139
df[(df["X10"]<0.7)|(df["X10"]>1.1)].count()#223
df[(df["X3"]<0.8)|(df["X3"]>2.6)].count()#272
#%%
df=pd.get_dummies(df,drop_first=True)
df
df=df[["X1","X2_Male","X3","X4","X5","X6","X7","X8","X9","X10","X11"]]
x =df.iloc[:,:-1]
y =df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=123)
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
param_grid = {"criterion": ["gini","entropy"],
              "min_samples_split": [35,40,45,50],
              "min_samples_leaf": [3,5,8],
              "max_depth": [5]}
cv=GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="accuracy",n_jobs=-1,cv=10).fit(x_train,y_train)
#%%
from sklearn.metrics import accuracy_score
cv.best_params_
sel_tree = cv.best_estimator_
grid_pred = sel_tree.predict(x_test)
accuracy_score(y_test,grid_pred)