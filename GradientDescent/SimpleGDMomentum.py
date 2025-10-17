import math
import numpy as np

def grad(x):
    return 2 * x + 10 * np.cos(x)

def GD_momentum(theta_init, eta, gamma):
    theta = [theta_init]
    v_old = 0
    for it in range(200):
        v_new = gamma * v_old + eta * grad(theta[-1])
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        v_old = v_new
        if abs(grad(theta[-1])) < 1e-3:
            break
    return (theta, it)

def NAG(theta_init, eta, gamma):
    theta = [theta_init]
    v_old = 0
    for it in range(200):
        v_new = gamma * v_old + eta * grad(theta[-1] - gamma * v_old)
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        v_old = v_new
        if abs(grad(theta[-1])) < 1e-3:
            break
    return (theta, it)

(theta, it1) = GD_momentum(-6, 0.1, 0.9)
(theta_NAG, it_NAG) = NAG(-6, 0.05, 0.9)
print('Theta: ', theta_NAG)
print('Obtained after %d interations'%(it_NAG))
# print('Solution of GD Momentum is theta = %f, obtained after %d interations' %(theta[-1], it1))
# print('Solution of NAG is theta = %f, obtained after %d interations' %(theta_NAG[-1], it_NAG))