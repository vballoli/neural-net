import numpy as np
from nn import NeuralNetwork
from layer import Layer
from PIL import Image
import os

alpha = 1

# layers = []
# layers.append(Layer(2, 4, Layer.LINEAR))
# layers.append(Layer(4, 1, Layer.SIGMOID))
# n = NeuralNetwork(alpha, layers)

# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([0, 1, 1, 0])

layers = []
layers.append(Layer(2500, 100, Layer.LINEAR))
layers.append(Layer(100, 50, Layer.SIGMOID))
layers.append(Layer(50, 4, Layer.SIGMOID))
layers.append(Layer(4, 1, Layer.SIGMOID))
n = NeuralNetwork(alpha, layers)

path = './images/'

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
print(X.shape)
X = X.reshape((X.shape[0], X.shape[2]))
print(X.shape)
y = y.reshape((y.shape[0], 1))
print(y.shape)

for _ in range(100):
#    for i in range(X.shape[0]):
    n.feed_forward(X)
    n.back_propogation(y)
    n.update_weights()

print(layers[len(layers)-1].y)