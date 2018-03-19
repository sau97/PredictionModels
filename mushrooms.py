# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:02:58 2017

@author: Saurabh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, datasets, tree, neighbors
from matplotlib import style
from sklearn.feature_selection import RFE
style.use('ggplot')
df=pd.read_csv('Documents\\mushrooms.csv')
#print(df.head())
model=svm.SVR(kernel='linear')
selector=RFE(model,5)
df["class"].replace("e",1,inplace=True)
df["class"].replace("p",0,inplace=True)
#print(df.head())
a=list(df)
con=preprocessing.LabelEncoder()
for c in a:
  df[c]=con.fit_transform(df[c])  

#print(df.head())
model=svm.SVC(kernel='linear')
selector=RFE(model,5)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
print('Confidence using SVC')
clf = svm.SVC()

clf.fit(X_train, y_train)
selector.fit(X_train,y_train)
accu = clf.score(X_test, y_test)
print(accu)
print(selector.ranking_)
print(selector.support_)

