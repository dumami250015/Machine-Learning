import numpy as np
import matplotlib.pyplot as plt

def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    x_i = np.array([1, X[true_i][0]])
    y_i = y[true_i]
    a = np.dot(x_i, w) - y_i
    return (a * x_i).reshape(2, 1)

def SGD(w_init, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            w_new = w[-1] - eta * sgrad(w[-1], i, rd_id)
            w.append(w_new)
            if count % iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3:
                    return w, count
                w_last_check = w_this_check
    return w, count

X = np.random.rand(1000, 1)
y = 4 + 3 * X + 0.2 * np.random.randn(1000, 1)
one = np.ones((X.shape[0], 1))

w, iter = SGD(np.array([[0], [0]]), 0.1)
x_0 = np.linspace(0, 1, 2, endpoint=True)
y_0 = w[-1][0][0] + w[-1][1][0] * x_0

plt.plot(X.T, y.T, 'b.')
plt.plot(x_0, y_0, 'r-', linewidth=2)
plt.axis([0, 1, 0, 10])
print('w =', w[-1].T, 'iter =', iter)
plt.show()