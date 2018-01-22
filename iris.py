from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Limits Number of Rows
pd.set_option('display.max_rows', 200) 

# Loads Up Data
iris = load_iris()
iris.keys()
X = np.array(iris['data'])
df = pd.DataFrame(X)
df.columns = iris['feature_names']
target = pd.Series(iris['target'])
df['target'] = target.values

# Classifier
km = KMeans(n_clusters=3, max_iter=1000)
km.fit(X)
prediction = pd.Series(km.labels_)
df['KMeans Prediction'] = prediction.values
print(df)

# The Table prints oddly but it shows the prediction next to the actual value and the rest of the data above that
# (Remember the classigier doesn't know which class is which as unsupervised, thats why the labels may be a different number)
