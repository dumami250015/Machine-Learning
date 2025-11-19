import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from mnist import MNIST
import matplotlib.pyplot as plt

mntrain = MNIST('./dataset/MNIST/training/')
mntrain.load_training()
Xtrain = np.asarray(mntrain.train_images)/255.0
ytrain = np.array(mntrain.train_labels.tolist())

mntest = MNIST('./dataset/MNIST/testing/')
mntest.load_testing()
Xtest = np.asarray(mntest.test_images)/255.0
ytest = np.array(mntest.test_labels.tolist())

# train
logreg = linear_model.LogisticRegression(C=1e5, solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(Xtrain, ytrain)

# test
y_pred = logreg.predict(Xtest)
print("Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist())))