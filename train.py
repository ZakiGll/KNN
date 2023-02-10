import arff, numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from knn import knn


dataset = arff.load(open('diabetes.arff', 'r'))
data = np.array(dataset['data'])

X = data[:, :-1]
y = data[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = knn(10)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

acc = np.sum( prediction == y_test) / len(y_test)
print(acc)
