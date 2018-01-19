import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as np

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

print(len(digits.data))

x = np.array([0, 1, 2, 3, 10, 11, 10, 14])
y = np.array([3, 4, 5, 4, 10, 11, 12, 12])

X = ([[0, 3], [1, 4], [2, 5], [3, 4], [10, 10], [11, 11], [10, 12], [14, 12]])


# clf.fit(x, y)

plt.scatter(x, y, color='black')
plt.show()
exit()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
