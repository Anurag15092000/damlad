# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:48:23 2019

@author: asus
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset=pd.read_csv('Bankruptcy.csv')



x=dataset.iloc[:, 3:27].values
y=dataset.iloc[:, 27].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(x_train)
X_test=sc_x.fit_transform(x_test)
x_random=sc_x.fit_transform(X_random)

X_random=[[0.4,0.5,0.4,0.33,0.6,0.1,0.32,0.11,0.467,0.71,0.45,0.33,0.57,0.12,0.24,0.32,0.11,0.8,0.7,0.22,0.6,0.7,0.4,0.1]]
len(X_random[0])

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
x_random=pca.transform(x_random)
explained_variance=pca.explained_variance_ratio_


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

X_test

y_pred=classifier.predict(x_random)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from matplotlib.colors import ListedColormap
x_set,y_set=X_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:, 0].min()-1,stop=x_set[:, 0].max()+1,step=0.01),
                  np.arange(start=x_set[:, 1].min()-1,stop=x_set[:, 1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Bankruptcy")
plt.
plt.show()


from matplotlib.colors import ListedColormap
x_set,y_set=X_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:, 0].min()-1,stop=x_set[:, 0].max()+1,step=0.01),
                  np.arange(start=x_set[:, 1].min()-1,stop=x_set[:, 1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Bankruptcy")
plt.show()