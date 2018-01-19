from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
iris.keys()
X = np.array(iris['data'])

train = X
y = np.array(iris['target'])
for i in range(10):
	rand = random.randint(0, len(train))
	if i == 0:
		test_values = np.array([train[rand]])
		test_target = np.array([y[rand]])
	else:
		test_values = np.vstack((test_values, train[rand]))
		test_target = np.vstack((test_target, y[rand]))
	train = np.delete(train, (rand), axis=0)
	y = np.delete(y, (rand), axis=0)
neigh = KNeighborsClassifier().fit(train, y)

for i in range(len(test_values)):
	print("Predicted: ", neigh.predict([test_values[i]])[0], "Actual: ", test_target[i][0], "Probability: ",  neigh.predict_proba([test_values[i]]))
