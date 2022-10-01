'''
`smearn` is a small deep learning library I made to learn the fundamentals of deep learning.

At the moment it can only be used to create feedforward neural networks for supervised learning, and I have no intention of expanding its functionality to handle recurrent neural networks or unsupervised learning: the goal of this project was for me to learn how to set up an easily-expandable framework for deep learning, and I think that, with how it has been set up, expanding it to handle these extra functionalities would be easy enough that I would get very little learning value from it.
Moreover, needless to say, there are many better alternatives.

The module is based on a symbolic treatment of the computations that a neural network performs that allows for learning using gradient methods and backpropagation. This basis is included in the `smearn.symbolic` submodule, which along with `smearn.models` is imported into the `main` smearn module.
All the computations are done using `numpy`.
'''

from .symbolic import *
from .models import *

from . import layers
from . import optimization
from . import regularization