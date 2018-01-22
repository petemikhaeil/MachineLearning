import numpy as np
import random
from sklearn import datasets as ds
from sklearn import linear_model
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Random Data I made up to give a nice shape
x = [i for i in range(17)]
y = [0, 5, 16, 18, 21, 25, 27.5, 28, 29, 32, 30, 28, 31, 29, 30, 29.5, 30]

# Plot these points
plt.scatter(x, y, color='black')
plt.show()

# Have to replot the points everytime
plt.scatter(x, y, color='black')
# Classifier (Linear as seen by the 1)
pl1 = np.polyfit(x, y, 1)
plt.plot(x, np.polyval(pl1, x), color='blue', label='Order 1')
plt.legend()
plt.show()

plt.scatter(x, y, color='black')
# Classifier (Order 3)
pl2 = np.polyfit(x, y, 3)
plt.plot(x, np.polyval(pl2, x), color='orange', label='Order 3')
plt.legend()
plt.show()

plt.scatter(x, y, color='black')
# Classifier Order 13
pl3 = np.polyfit(x, y, 13)
plt.plot(x, np.polyval(pl3, x), color='purple', label='Order 13')
plt.legend()
plt.show()

