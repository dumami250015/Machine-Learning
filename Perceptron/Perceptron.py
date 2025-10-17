import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)

def sg(w, x):
    return np.sign(np.dot(w.T, x))

def check_converged(w, X, y):
    return np.array_equal(sg(w, X), y)

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        rd_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, rd_id[i]].reshape(d, 1)
            yi = y[0, rd_id[i]]
            if (sg(w[-1], xi) != yi):
                w_new = w[-1] + yi * xi
                w.append(w_new)
                mis_points.append(rd_id[i])
            
        if check_converged(w[-1], X, y):
            break
    return (w, mis_points)

w_init = np.random.randn(X.shape[0], 1)
(w, mis_points) = perceptron(X, y, w_init)
print(w)
print('Number of updates:', len(w) - 1)
print('Misclassified points indices:', mis_points)

x_0 = np.linspace(0, 1, 3, endpoint=True)
y_0 = w[-1][0][0] + w[-1][1][0] * x_0
plt.plot(x_0, y_0, 'y-', linewidth=2)
plt.plot(X[1, 0:N], X[2, 0:N], 'bo')
plt.plot(X[1, N:2 * N], X[2, N:2 * N], 'ro')
