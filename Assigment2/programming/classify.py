from load_mnist import *
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

X_train, y_train = load_mnist('training' , )
X_test, y_test = load_mnist('testing'  ,   )

#Scaling the data
X_train = X_train / 255.
X_test = X_test/255.
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
mlp.fit(X_train, y_train)
print("Test set score: %f" % mlp.score(X_test, y_test))
loss_values=mlp.loss_curve_
plt.title('Training loss curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(loss_values)
plt.show()