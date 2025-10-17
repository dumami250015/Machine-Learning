import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(1000, 1)
y = 4 + 3 * X + 0.2 * np.random.randn(1000, 1)

def sgrad(w, batch):
    X_batch = np.array(X[batch[:]][0])
    one = np.ones((len(batch), 1))
    X_batch = np.concatenate((one, X_batch), axis=1)
    y_batch = np.array(y[batch[:]])
    a = np.dot(X_batch, w) - y_batch
    grad = np.dot(X_batch.T, a)
    return grad

def MBGD(w_init, eta, batch_size):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        batch = []  
        for i in range(N):
            batch_num = (i + 1) // batch_size + (1 if (i + 1) % batch_size != 0 else 0)