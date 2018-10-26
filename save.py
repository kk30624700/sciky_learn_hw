'''
Created on 2018年10月14日

@author: emti
'''
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)


from sklearn.externals import joblib
# Save
joblib.dump(clf, 'save/clf.pkl')
# restore
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))