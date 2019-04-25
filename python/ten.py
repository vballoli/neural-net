import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Dense(2500, input_shape=(2500,), activation='linear'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([0, 1, 1, 0])

# layers = []
# layers.append(Layer(2500, 100, Layer.LINEAR))
# layers.append(Layer(100, 50, Layer.SIGMOID))
# layers.append(Layer(50, 4, Layer.SIGMOID))
# layers.append(Layer(4, 1, Layer.SIGMOID))
# n = NeuralNetwork(alpha, layers)

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
print(X.shape)
X = X.reshape((X.shape[0], X.shape[2]))
print(X.shape)
y = y.reshape((y.shape[0], 1))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)


# model.compile(loss='mean_squared_error', optimizer='RMSProp')
# model.fit(X, y, epochs=100)
# print(model.predict())