import pandas as pd
from random import randint

df = pd.read_csv("Iris1.csv", header = 0)
data = df.values.tolist()
data_test, data_train = [], []

for x in data:
    data_test.append(x) if randint(1, 10) == 5 else data_train.append(x)

def cal(x, y):
    m = 0
    for i in range(len(x) - 1): 
        m += abs(x[i] - y[i])
    return m

k = 10
for x in data_test:
    tmp = []
    for y in data_train: 
        tmp.append([cal(x, y), y[4]])
    tmp.sort(key = lambda x: x[0])
    name = ("Iris-virginica", "Iris-versicolor", "Iris-setosa")
    d = [0, 0, 0]
    for i in range(k):
        if tmp[i][1] == "Iris-virginica":
            d[0] += 1
        elif tmp[i][1] == "Iris-versicolor":
            d[1] += 1
        else:
            d[2] += 1
    maxn = 0
    pos = 0
    for i in range(3):
        if d[i] > maxn:
            maxn = d[i]
            pos = i
    print(name[pos] == x[4], x[4], name[pos])