
import requests, gzip, os, hashlib
import numpy as np

import smearn

def fetch(url):
    fp = os.path.join("data", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

Y_onehot = np.zeros(Y.shape + (10,))
Y_onehot[np.arange(Y.shape[0]), Y] = 1.0

def test_model_accuracy(model, X_test, Y_test):
    Y_pred = model.evaluate(X_test).argmax(axis=-2).reshape((-1,))
    return np.sum(np.where(Y_pred == Y_test, 1, 0)) / Y_test.shape[0]


print("Example 1. Simple model with two hidden layers")

X_i = smearn.layers.Input(28*28)
layer = X_i
layer = smearn.layers.Dense(layer, 30)
layer = smearn.layers.ReLU(layer)
layer = smearn.layers.Dense(layer, 10)
layer = smearn.layers.Softmax(layer)

model = smearn.Model(input=X_i, output=layer, loss=smearn.layers.CrossEntropy, optimizer=smearn.optimization.Adam(0.001))

model.train(data=X, labels=Y_onehot, lr=0.01, batch_size=16, epochs=10, validation_split=64, early_stopping=2)

print("Test results accuracy: {}%\n".format(100 * test_model_accuracy(model, X_test, Y_test)))


print("Example 2. Bigger model with different regularizations at various layers and a linearly decaying learning rate")

X_i = smearn.layers.Input(28*28)
layer = X_i
layer = smearn.layers.Dropout(layer, 0.8)
layer = smearn.layers.Dense(layer, 40)
layer = smearn.layers.ReLU(layer)
layer = smearn.layers.Dense(layer, 60, regularization=smearn.regularization.L2(0.01))
layer = smearn.layers.ReLU(layer)
layer = smearn.layers.Dense(layer, 40, regularization=smearn.regularization.L1(0.01))
layer = smearn.layers.ReLU(layer)
layer = smearn.layers.Dense(layer, 10)
layer = smearn.layers.Softmax(layer)

model = smearn.Model(input=X_i, output=layer, loss=smearn.layers.CrossEntropy, optimizer=smearn.optimization.Adam(smearn.optimization.LinearDecrease(1e-3, 1e-5, 10)))



model.train(data=X, labels=Y_onehot, lr=0.01, batch_size=256, epochs=10, validation_split=64, early_stopping=3)


print("Test results accuracy: {}%\n".format(100 * test_model_accuracy(model, X_test, Y_test)))


print("Example 3. Bagging ensemble")

models = []

for i in range(6):
    X_i = smearn.layers.Input(28*28)
    layer = X_i
    layer = smearn.layers.Dense(layer, 20)
    layer = smearn.layers.ReLU(layer)
    layer = smearn.layers.Dense(layer, 10)
    layer = smearn.layers.Softmax(layer)

    models.append(smearn.Model(input=X_i, output=layer, loss=smearn.layers.CrossEntropy, optimizer=smearn.optimization.SGD(0.001)))

model = smearn.regularization.BaggingEnseble(models)

model.train(data=X[:6000], labels=Y_onehot[:6000], lr=0.01, batch_size=8, epochs=10, early_stopping=2, validation_split=64)


Y_pred = model.evaluate(X_test).argmax(axis=-2).reshape((-1,))
print("Test results accuracy: {}%".format(100 * np.sum(np.where(Y_pred == Y_test, 1, 0)) / Y_test.shape[0]))

for i, m in enumerate(models):
    Y_pred = m.evaluate(X_test).argmax(axis=-2).reshape((-1,))
    print("Individual model #{} accuracy: {}%".format(i+1, 100 * np.sum(np.where(Y_pred == Y_test, 1, 0)) / Y_test.shape[0]))