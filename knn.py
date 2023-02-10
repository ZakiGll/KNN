import numpy as np
from collections import Counter


class knn:

    def disEuclid(self, x1, x2):
        distance = 0
        for i in range(len(x1) - 1):
            distance += np.sqrt(np.sum((x1[i].astype(np.cfloat) - x2[i].astype(np.cfloat)) ** 2))
        return distance

    def disManhattan(self, x1, x2):
        distance = 0
        for i in range(len(x1) - 1):
            distance += np.sum(np.abs(x1[i].astype(np.cfloat) - x2[i].astype(np.cfloat)))
        return distance

    def __init__(self, K=3):
        self.k = K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        prediction = [self.__predict(x) for x in X]
        return prediction

    def __predict(self, x):
        # calculer la distance
        distances = [self.disEuclid(x, x_train) for x_train in self.X_train]

        # trouver les plus proches exemples
        k_indices = np.argsort(distances)[:self.k]
        k_proches = [self.y_train[i] for i in k_indices]

        # class majoritaire
        class_majoritaire = Counter(k_proches).most_common()

        return class_majoritaire[0][0]
