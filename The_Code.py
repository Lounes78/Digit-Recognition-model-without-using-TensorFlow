import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Data Preprocessing/Preparation
data_train = pd.read_csv('data/mnist_train.csv')
data_dev   = pd.read_csv('data/mnist_test.csv')


data_train = np.array(data_train) #each line represent an image / the first column contains the label / 60000*785
m, n = data_train.shape
np.random.shuffle(data_train) #Shuffling the data to avoid biases in the model

data_dev = np.array(data_dev)
data_dev = data_dev.T

#developpement/validation data
Y_dev = data_dev[0] #label
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

#Training data
data_train = data_train.T
Y_train = data_train[0] #label
X_train = data_train[1:n] 
X_train = X_train / 255.
_, m_train = X_train.shape



# Coding the NN

def init_params():
	W1 = np.random.rand(10, 784) - 0.5  #Centering around 0 , See after Xavier/Glorot initialization
	b1 = np.random.rand(10, 1) - 0.5
	W2 = np.random.rand(10, 10) - 0.5
	b2 = np.random.rand(10, 1) - 0.5 
	return W1, b1, W2, b2


def ReLU(Z):
	return np.maximum(Z, 0)

def softmax(Z): #Gives us prob
	A = np.exp(Z) / sum(np.exp(Z))
	return A


def forward_prop(W1, b1, W2, b2, X):
	Z1 = W1.dot(X) + b1
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = softmax(Z2)
	return Z1, A1, Z2, A2


def ReLU_deriv(Z):
	return Z > 0  # Quite elegant


def one_hot(Y): # Need a matrix 10*m one hot encoded
	one_hot_Y = np.zeros((Y.size, Y.max()+1)) #We want 10 hyperclasses Y.size is the number of images 
	one_hot_Y[np.arange(Y.size), Y] = 1
	one_hot_Y = one_hot_Y.T
	return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
	one_hot_Y = one_hot(Y)
	dZ2 = A2 - one_hot_Y
	dW2 = 1 / m * dZ2.dot(A1.T)
	db2 = 1 / m * np.sum(dZ2)
	dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
	dW1 = 1 / m * dZ1.dot(X.T)
	db1 = 1 / m * np.sum(dZ1)
	return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	W1 = W1 - alpha * dW1	# use learning rate schedules to improve training.
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2
	return W1, b1, W2, b2


def get_predictions(A2):
	return np.argmax(A2, 0)  #Returns the indice of the max probability aka the prediction 

def get_accuracy(predictions, Y):
	return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
	W1, b1, W2, b2 = init_params()
	for i in range (iterations):
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) 
		W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		if i % 10 == 0:
			print("iteration: ", i)
			print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
	return W1, b1, W2, b2


# Training
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 200)




def make_prediction(X, W1, W2, b1, b2):
	_, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
	prediction = get_predictions(A2)
	return prediction   # si ça reçoie une seule image ça renvoie 1 thing sinon a list of predictions waw


def test_prediction(index, W1, W2, b1, b2):
	prediction = make_prediction(X_train[:, index, None], W1, W2, b1, b2) #Gives the prediction
	true_val = Y_train[index] # The label
	print("Prediction: ", prediction)
	print("Label: ", true_val)

	current_image = X_train[:, index, None] #None  is used to ensure that the selected has the right shape and can be properly reshaped and displayed as an image
	current_image = current_image.reshape((28, 28)) * 255
	plt.gray()
	plt.imshow(current_image)
	plt.show()


# Let's try it out
test_prediction(0, W1, W2, b1, b2)
test_prediction(1, W1, W2, b1, b2)
test_prediction(2, W1, W2, b1, b2)
test_prediction(3, W1, W2, b1, b2)
test_prediction(4, W1, W2, b1, b2)
test_prediction(5214, W1, W2, b1, b2)


# Accuracy for the dev set
def test_on_dev_set(X_dev, Y_dev, W1, W2, b1, b2):
	predictions = make_prediction(X_dev, W1, W2, b1, b2)
	accuracy = get_accuracy(predictions, Y_dev)

	print("Accuracy: ", accuracy)

test_on_dev_set(X_dev, Y_dev, W1, W2, b1, b2)
