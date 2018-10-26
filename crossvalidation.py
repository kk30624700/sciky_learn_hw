'''
Created on 2018年10月13日

@author: emti
'''
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X=iris.data
y=iris.target

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
#     scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross Validated Accuracy')
plt.show()