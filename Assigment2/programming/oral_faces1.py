import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import math
# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']
trainX_check= np.reshape(trainX,(240,112,92))
plt.imshow(trainX_check[0])
plt.show()
trainX = trainX/255.
testX = testX/255.
mlp = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(150, 100),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=50000, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.1, verbose=10, warm_start=False)

mlp.fit(trainX, trainY)
predicted_y = mlp.predict(testX)
print('{0:0.02f}'.format(  np.mean(predicted_y==testY)*100), "of test examples classified correctly.")
loss_values=mlp.loss_curve_
plt.title('Training loss curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(loss_values)
plt.show()