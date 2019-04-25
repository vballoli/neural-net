import numpy as np
from nn import NeuralNetwork
from layer import Layer
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix

alpha = 1

# layers = []
# layers.append(Layer(2, 4, Layer.LINEAR))
# layers.append(Layer(4, 1, Layer.SIGMOID))
# n = NeuralNetwork(alpha, layers)

# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([0, 1, 1, 0])

layers = []
layers.append(Layer(2500, 100, Layer.TANH))
layers.append(Layer(100, 50, Layer.SIGMOID))
layers.append(Layer(50, 1, Layer.SIGMOID))
n = NeuralNetwork(alpha, layers)

path = '/Volumes/mp/Datasets/ml/train/'

X = []
y = []

for image_path in os.listdir(path):
    img = np.array(Image.open(path+image_path).convert('L').resize((50, 50)))
    img = img.reshape((1, 2500))
    X.append(img)
    if image_path[0:3] == 'cat':
        y.append(0)
    else:
        y.append(1)

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[2]))
y = y.reshape((y.shape[0], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

for _ in range(50):
    n.feed_forward(X_train)
    n.back_propogation(y_train)
    n.update_weights()

output = X_test
for layer in layers:
    output = layer.derivative_activation(np.dot(output, layer.weights))

output = np.where(output>0.5, 1, 0)
print(mean_squared_error(output, y_test)**2)
print("Confusion matrix: ", confusion_matrix(y_test, output))