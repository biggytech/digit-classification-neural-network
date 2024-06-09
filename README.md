# Digit classification neural network
Uses [`numpy`](https://numpy.org/doc/stable/index.html) and math equations only!

## Creds
[Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang)

## Kaggle link
[Digit recognizer](https://www.kaggle.com/code/heroice/digit-recognizer)

## The Goal
Given an image with handwritten digit, predict which digit the image represents.

## Database used
MNIST database

## The Math
### Input
- 28 x 28 px images = 784 px
- black & white images (0 to 255 each pixel)
- `m` of these images

### The Network
We represent this input as matrix, where each row has 784 columns,
and we have `m` of such rows. Each row represents an image.

Then we're transponing this matrix so each column is an image. And there will be 784 rows corresponding to each pixel.

There are 3 layers in the neural network:
1. The 0 layer has 784 nodes - input layer. Each pixel maps to a node.
2. The first (hidden) layer has 10 nodes.
3. Second (output) layer has 10 units - corresponding to one digit that can be predicted.

### The Training
#### 1. Forward Propagation
Run an image throught this network, from this network you compute
what your outout is going to be. Going from the end to the input backwards.

- `A0 = X` (input layer)
- `Z1` = unactivated first layer (with weight and bias applied) = `W1 * A0 + b1`
- `A1 = ReLU(Z1)` with applied activation function (in order to remove linearity) - ReLU function used
- `Z2 = W2 * A1 + b2`
- `A2 = softmax(Z2)` with another activation function resulting in list of probabilities. Resulting prediction.

#### 2. Backwards propagation
In order to learn right `W*` weights and `b*` biases values.

- `dZ2` (error of the second layer) = `A2 - Y` (`Y` - actual correct label of the image)
- `dW2` - average of the absolute error = `1/m * dZ2 * A1^T`
- `db2 = 1/m * ∑dZ2`
- `dZ1 = W2^T * dZ2 * g'(Z1)` - note `g'` (derivative) to "undo" g function from predictions
- `dW1 = 1/m * dZ1 * X^T`
- `db1 = 1/m * ∑dZ1`

#### 3. Update weights and biases according to calculated values
- `W1 = W1 - αdW1`
- `b1 = b1 - αdb1`
- `W2 = W2 - αdW2`
- `b2 = b2 - αdb2`
where `α` - learning rate set by us

#### 4. Repeat 🔄
