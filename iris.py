from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 200)

iris = load_iris()
iris.keys()
X = iris['data']
df = pd.DataFrame(X)
df.columns = iris['feature_names']
target = pd.Series(iris['target'])
df['target'] = target.values
km = KMeans(n_clusters=3, max_iter=1000)
km.fit(X)
prediction = pd.Series(km.labels_)
df['KMeans Prediction'] = prediction.values

print(df)
