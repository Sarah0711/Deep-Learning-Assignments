from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.metrics

# Load data - two class
X_train, y_train = load_mnist('training')
X_test_orig, y_test = load_mnist('testing')

np.random.seed(5)
#Total classes 10
i=0
X_train_final = []
y_train_final = np.zeros(1000)
count =0
while(i<10):
    indices_for_class = np.argwhere(y_train==i).flatten()
    indices_for_data = np.random.choice(indices_for_class,100)
    y_train_final[count:count+100] = i
    count = count +100
    X_train_final.extend([X_train[j] for j in indices_for_data])
    i=i+1



# Reshape the image data into rows
X_train = np.reshape(np.array(X_train_final), (np.array(X_train_final).shape[0], -1))
X_test = np.reshape(X_test_orig, (X_test_orig.shape[0], -1))


# Compute distances:
dists = mlBasics.compute_euclidean_distances(X_train, X_test)

y_test_pred ,predictions= mlBasics.predict_labels(dists, y_train_final)

fig=plt.figure(figsize=(5, 20))
columns = 2
rows = 10
image_number=1
for i in range(10):
    fig.add_subplot(rows, columns, image_number)
    image_number = image_number+1
    plt.imshow(X_test_orig[i])
    predicted = predictions[i]
    for j in range(1):
        index = predicted[j]
        fig.add_subplot(rows, columns, image_number)
        image_number = image_number+1
        plt.imshow(X_train_final[index])
plt.show()
print('{0:0.02f}'.format(  np.mean(y_test_pred==y_test)*100), "of test examples classified correctly with k =1.")
print(sklearn.metrics.confusion_matrix(y_test, y_test_pred, labels=None, sample_weight=None))

y_test_pred,predictions = mlBasics.predict_labels(dists, y_train_final,5 )
fig=plt.figure(figsize=(20, 20))
columns = 6
rows = 10
image_number=1
for i in range(10):
    fig.add_subplot(rows, columns, image_number)
    image_number = image_number + 1
    plt.imshow(X_test_orig[i])
    predicted = predictions[i]
    for j in range(5):
        index = predicted[j]
        fig.add_subplot(rows, columns, image_number)
        image_number = image_number + 1
        plt.imshow(X_train_final[index])

plt.show()

print('{0:0.02f}'.format(  np.mean(y_test_pred==y_test)*100), "of test examples classified correctly with k =5.")
print(sklearn.metrics.confusion_matrix(y_test, y_test_pred, labels=None, sample_weight=None))

folds = 5
ks = np.arange(15)+1
folds_indices = np.arange(X_train.shape[0])
np.random.shuffle(folds_indices)
folds_indices = folds_indices.reshape(folds,-1)
accuracies = np.zeros(ks.shape[0])
for k in ks:
    for i in range(folds):
        indices = (folds_indices[~np.isin(np.arange(len(folds_indices)), [i])].flatten())
        dists = mlBasics.compute_euclidean_distances(X_train[indices], X_train[folds_indices[i]])
        y_test_pred, predictions = mlBasics.predict_labels(dists, y_train_final[indices],k=k)
        accuracies[k-1] += np.mean(y_test_pred == y_train_final[folds_indices[i]])
    accuracies[k-1] /= folds

best_k = np.argmax(accuracies)+1
print("Best K is "+str(best_k))
plt.plot(ks,accuracies)
plt.show()
