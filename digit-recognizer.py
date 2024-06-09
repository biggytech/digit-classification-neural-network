# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:28:05.487649Z","iopub.execute_input":"2024-06-09T10:28:05.488496Z","iopub.status.idle":"2024-06-09T10:28:05.494346Z","shell.execute_reply.started":"2024-06-09T10:28:05.488453Z","shell.execute_reply":"2024-06-09T10:28:05.492718Z"}}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:28:05.496789Z","iopub.execute_input":"2024-06-09T10:28:05.497216Z","iopub.status.idle":"2024-06-09T10:28:08.770974Z","shell.execute_reply.started":"2024-06-09T10:28:05.497181Z","shell.execute_reply":"2024-06-09T10:28:08.769374Z"}}
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:28:08.773047Z","iopub.execute_input":"2024-06-09T10:28:08.773583Z","iopub.status.idle":"2024-06-09T10:28:08.797794Z","shell.execute_reply.started":"2024-06-09T10:28:08.773542Z","shell.execute_reply":"2024-06-09T10:28:08.795881Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:28:08.800975Z","iopub.execute_input":"2024-06-09T10:28:08.801491Z","iopub.status.idle":"2024-06-09T10:28:09.680048Z","shell.execute_reply.started":"2024-06-09T10:28:08.801441Z","shell.execute_reply":"2024-06-09T10:28:09.678959Z"}}
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255. # !

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. # !

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:28:09.681489Z","iopub.execute_input":"2024-06-09T10:28:09.681833Z","iopub.status.idle":"2024-06-09T10:28:09.697570Z","shell.execute_reply.started":"2024-06-09T10:28:09.681804Z","shell.execute_reply":"2024-06-09T10:28:09.696147Z"}}
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0
    
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1/ m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# %% [code]
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# %% [code]
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:33:05.977599Z","iopub.execute_input":"2024-06-09T10:33:05.978066Z","iopub.status.idle":"2024-06-09T10:33:05.988689Z","shell.execute_reply.started":"2024-06-09T10:33:05.978033Z","shell.execute_reply":"2024-06-09T10:33:05.987165Z"}}
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:35:23.936692Z","iopub.execute_input":"2024-06-09T10:35:23.937108Z","iopub.status.idle":"2024-06-09T10:35:24.202465Z","shell.execute_reply.started":"2024-06-09T10:35:23.937076Z","shell.execute_reply":"2024-06-09T10:35:24.201333Z"}}
test_prediction(2, W1, b1, W2, b2)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-09T10:36:30.940116Z","iopub.execute_input":"2024-06-09T10:36:30.941520Z","iopub.status.idle":"2024-06-09T10:36:30.978960Z","shell.execute_reply.started":"2024-06-09T10:36:30.941471Z","shell.execute_reply":"2024-06-09T10:36:30.977136Z"}}
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
