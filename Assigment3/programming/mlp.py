import numpy as np
from tensorflow import keras
fashion_mnist = keras.datasets

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']
trainX_check= np.reshape(trainX,(240,112,92))
plt.imshow(trainX_check[0])
plt.show()
import torch



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.out = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.out(x),dim=1)
        return x

dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, trainX[0].size, 512, 20

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in,dtype=dtype,device=device)
y = torch.randn(N, D_out,dtype=dtype,device=device)
x = torch.from_numpy(trainX).float().to(device)
y = torch.from_numpy(trainY)
y=y.type(torch.LongTensor).to(device)





# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
model.cuda()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x).squeeze()

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 1000 == 99:
        print(loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_x = torch.from_numpy(testX).float().to(device)
test_y = torch.from_numpy(testY).to(device)


correct = 0
total = 0
with torch.no_grad():
    outputs = model(test_x)
    _, predicted = torch.max(outputs.data, 1)
    total += test_y.shape[0]
    correct += (predicted == test_y).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))