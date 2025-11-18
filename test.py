import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from scipy import misc
np.random.seed(0)

Z = np.array([[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]])

def softmax(Z):
    e_Z = Z
    A = e_Z / e_Z.sum(axis=0)
    return e_Z.sum(axis=0)

print(Z)
print(softmax(Z))