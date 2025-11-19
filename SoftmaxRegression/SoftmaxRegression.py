import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

def convert_labels(y, C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A

def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y * np.log(A))

def grad(X, Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)

# check grad by numerical_grad
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i, j] = (cost(X, Y, W_p) - cost(X, Y, W_n)) / (2 * eps)
    return g

def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 100000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20

    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(W[-1].T.dot(xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1

            if (count % check_w_after) == 0:
                if (np.linalg.norm(W_new - W[-check_w_after]) < tol):
                    return W
            W.append(W_new)
    return W

def pred(W, X):
    A = softmax(W.T.dot(X))
    return np.argmax(A, axis=0)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0).T 
# extended data
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3
eta = .05

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

W_init = np.random.randn(X.shape[0], C)
W = softmax_regression(X, original_label, W_init, eta)
print(W[-1])

# display
# 1. Setup Plot
plt.figure(figsize=(10, 8))

# 2. Plot the Data Points (X0, X1, X2) with different colors and markers
# We match the markers from your example image
plt.plot(X0[:, 0], X0[:, 1], 'rs', markersize=6, label='Class 0 (Squares)', markeredgecolor='k', alpha=0.7)
plt.plot(X2[:, 0], X2[:, 1], 'b^', markersize=6, label='Class 2 (Triangles)', markeredgecolor='k', alpha=0.7)
plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=6, label='Class 1 (Circles)', markeredgecolor='k', alpha=0.7)


# 3. Plot Decision Regions (Meshgrid background)
# Determine grid range
x_min, x_max = X[1, :].min() - 1, X[1, :].max() + 1
y_min, y_max = X[2, :].min() - 1, X[2, :].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict for every point on the grid
grid_data = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()].T
Z = pred(W[-1], grid_data)
Z = Z.reshape(xx.shape)

# Plot the filled regions
# 'brg' matches blue, red, green. We can use a custom map to match your image
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FFDDDD', '#DDDDFF', '#DDFFDD']) # light red, light blue, light green
# Note: The order depends on your labels. Let's try to match the plot.
# Class 0 = red, Class 1 = green, Class 2 = blue
cmap_visual = ListedColormap(['#FDBDBD', '#D4FDBE', '#BFDBFE']) # light red, light green, light blue
plt.contourf(xx, yy, Z, alpha=0.7, cmap=cmap_visual)
 

# 4. Final Formatting
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Softmax Regression Results (Simple Plot)')
plt.legend()

# Turn off the x/y ticks to match the example
plt.xticks([])
plt.yticks([])

plt.show()