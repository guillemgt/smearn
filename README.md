# smearn

`smearn` is a small deep learning library written in python and using `numpy` that I made to learn the fundamentals of deep learning.
It is based on a symbolic treatment of the computations that a neural network performs that allows for learning using gradient methods and backpropagation.

This repository includes the package as well as a usage example, which uses some of the features of the library to create models for handwritten digit classification using the MNIST dataset.

At the moment, the library can only be used to create feedforward neural networks for supervised learning, and I have no intention of expanding its functionality to handle recurrent neural networks or unsupervised learning. However, with how the library has been set up, expanding it to handle these extra functionalities would be straightforward.
Moreover, needless to say, there are many better alternatives.

The documentation for the project can be found [here](https://guillemgt.github.io/smearn/).