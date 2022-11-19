import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv("dataset_website.csv")
x=data.iloc[:,1:35].values
y=data.iloc[:,-1].values
print(x,y)
[[-1  1  1 ...  1 -1 -1]
 [ 1  1  1 ...  1  1 -1]
 [ 1  0  1 ...  0 -1 -1]
 ...
 [ 1 -1  1 ...  0  1 -1]
 [-1 -1  1 ...  1  1 -1]
 [-1 -1  1 ...  1 -1 -1]] [-1 -1 -1 ... -1 -1 -1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
LogisticRegression()
y_pred1=lr.predict(x_test)
from sklearn.metrics import accuracy_score
log_reg=accuracy_score(y_test,y_pred1)
log_reg
1.0
import pickle
pickle.dump(lr,open('phishing_website.pkl','wb'))