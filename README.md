# Library for convolutional neural networks

Note that this is in fact *not* an optimized, usable tensorflow alternative.
This is a project, I've made in winter 2022, because I wanted to learn.
It is very slow and not a novum, but it works and back then, it was a huge achievement for me :)

### Example usage

```
# X being the training data converted to sensible np matrices
# Y being the training data labels, a list of vectors containing the desired output values (0 - 1)
net = NeuralNetwork([
    Convolutional(1, (28, 28), 4, 32),
    ELU(),
    Pooling(2),
    Dropout(0.2),
    Convolutional(32, (13, 13), 3, 64),
    Pooling(2),
    ELU(),
    Dropout(0.2),
    Reshape((64, 6, 6), (2304, 1)),
    Dense(2304, 128),
    LeakyReLU(),
    Dense(128, 10),
    Softmax()
], CategoricalCrossEntropy())
X, Y = your_training_data_receiver()
net.train(X, Y, epochs=5, batch_size=64, shuffle=True, lr=0.1)
```
