import math
import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 2 * x + 5 * np.cos(x) 

def cost(x):
    return x ** 2 + 5 * np.sin(x)

def myGD(eta, x0):
    x = [x0]
    for it in range(100):
        newX = x[-1] - eta * grad(x[-1])
        if abs(grad(newX)) < 1e-3:
            break
        x.append(newX)
    return (x, it)

(x1, it1) = myGD(0.1, -5)
(x2, it2) = myGD(0.1, 2.5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))