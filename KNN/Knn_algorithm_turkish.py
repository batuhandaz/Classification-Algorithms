#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:23:46 2021
@author: batuhandaz
"""
# Kütüphaneler tanımlanır
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Veri seti içeriye aktarılır
dataset = pd.read_csv('/home/batuhan/İndirilenler/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Veri setini eğitim ve test seti olarak ayırma 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Eğitim setine sınıflandırıcı ekleme 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)
# Test sonuçlarını karşılaştırma
y_pred = classifier.predict(X_test)
# Confusion Matrisi oluşturma
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Eğitim seti sonuçlarını görselleştirme
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'black')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'white'))(i), label = j)
plt.title('KNN')
plt.xlabel('Yaş')
plt.ylabel('Tahmini Maaş')
plt.legend()
plt.show()