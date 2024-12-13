from random import randint
import matplotlib.pyplot as plt
import numpy as np

def view(center, cluster):
    plt.clf()
    color = ['g^', 'ro', 'bs']
    for i in range(k):
        px = [p[0] for p in cluster[i]]
        py = [p[1] for p in cluster[i]]
        plt.plot(px, py, color[i])
    for i in range(k):
        plt.plot(center[i][0], center[i][1], 'yo', markersize = 20, alpha = 0.9)
    plt.show()

def cal(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

n, P, k, rand = 1000, [], 3, 100

for i in range(n):
    while True:
        x, y = randint(1, rand), randint(1, rand)
        if [x, y] not in P:
            P.append([x, y])
            break

center = [[randint(1, rand), randint(1, rand)] for i in range(k)]
centerTmp = []

while True:
    global cluster
    cluster = [[], [], []]
    for i in range(n):
        tmp = [[cal(P[i], center[j]), j] for j in range(k)]
        cluster[min(tmp, key = lambda x: x[0])[1]].append(P[i])
    view(center, cluster)
    for i, eachCluster in enumerate(cluster):
        sumx, sumy = 0, 0
        for p in eachCluster:
            sumx += p[0]
        for p in eachCluster:
            sumy += p[1]
        center[i] = [sumx // len(eachCluster), sumy // len(eachCluster)]
    if center == centerTmp:
        break
    else:
        centerTmp = center[:]

view(center, cluster)