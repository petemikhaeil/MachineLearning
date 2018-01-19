import numpy as np
import random
from sklearn import datasets as ds
from sklearn import linear_model
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


x = [i for i in range(17)]

y = [0, 4, 8.5, 11.5, 12.5, 14, 14, 14.5, 14, 14, 14, 14, 14, 14]
y = [0, 5, 16, 18, 21, 25, 27.5, 28, 29, 32, 30, 28, 31, 29, 30, 29.5, 30]
plt.scatter(x, y, color='black')
plt.show()

plt.scatter(x, y, color='black')
pl1 = np.polyfit(x, y, 1)
pl2 = np.polyfit(x, y, 3)
pl3 = np.polyfit(x, y, 13)
plt.plot(x, np.polyval(pl1, x), color='blue', label='Order 1')
plt.legend()
plt.show()

plt.scatter(x, y, color='black')
plt.plot(x, np.polyval(pl2, x), color='orange', label='Order 3')
plt.legend()
plt.show()

plt.scatter(x, y, color='black')
plt.plot(x, np.polyval(pl3, x), color='purple', label='Order 13')
plt.legend()
plt.show()

