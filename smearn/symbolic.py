'''
The module `smearn.symbolic` contains the basics of symbolic manipulation on which `smearn` are based.
It contains the basics of computation graphs for forward and backward propagation for computing values and gradients.
'''

import numpy as np

#
# Exceptions
#

class ShapeError(Exception):
    '''
    An exception type to be passed when there is an error regarding the shapes of tensors -- for example, if trying to add two tensors of different (non-broadcastable) shapes.
    '''
    pass


#
# Symbolic manipulation
#

class Symbol:
    '''
    A `smearn.symbolic.Symbol` represents a node in a computation graph.
    '''

    def __init__(self, shape=None, parents=[], value=None, constant=False, regularization=None, value_initializers=None):
        '''
        `shape` is the shape of the tensor that the node will output

        `value` will hold its value of the tensor (after being computed if the symbol it represents is not a constant). Note that the shape of `value` may (and usually will) be different from `shape`, this is because values may be computed for batches of more than one element at a time.

        `constant` determines whether the symbol represents a constant (which will not be optimized for during training).

        `parents` is a list of other symbols that will be used to compute the value of the current symbol.

        Any class inheriting from this one and representing a specific operation should overload the `_op_f` and `_op_b` methods.
        `_op_f` must return the value of the symbol (computed using the values of its parents).
        `_op_b` must take as arguments an integer i and the gradient tensor of some other symbol with respect to the current one and return the gradient of that symbol with respect to the i-th parent of the current one.
        '''
        self.shape = shape
        self.value = value
        self.constant = constant
        self.parents = parents
        for parent in parents:
            parent.children.append(self)
        self.children = []
        self.gradient = None
        self.regularization = regularization
        self.value_initializers = value_initializers

    def set_value(self, value):
        if not self.constant:
            raise Exception("Setting the value of a constant symbol is not allowed.")

        s1, t1 = remove_trailing_ones(self.shape)
        s2, t2 = remove_trailing_ones(value.shape)

        if len(s1) > 0 and s1 != s2[-len(s1):]:
            raise ShapeError("Shape {} is not broadcastable to shape {}".format(self.shape, value.shape))

        self.value = value.reshape(value.shape + (t1 - t2) * (1,)) if t1 >= t2 else value.reshape(value.shape[:t1-t2])

        

    def compute_value(self):
        if self.value is not None:
            return

        for parent in self.parents:
            parent.compute_value()

        self.value = self._op_f()

    def compute_gradients(self, batch_shape):
        if self.gradient is not None:
            return

        self.gradient = np.zeros(batch_shape + self.shape)

        for child in self.children:
            child.compute_gradients(batch_shape)
            v = child._op_b(child.parents.index(self), child.gradient)
            self.gradient += v

    def propagate_gradients(self, batch_shape=None):
        if batch_shape is None:
            self.gradient = np.ones(self.value.shape)
            batch_shape = self.value.shape[:-len(self.shape)]
        else:
            self.compute_gradients(batch_shape)

        for parent in self.parents:
            parent.propagate_gradients(batch_shape)

    def propagate_learning(self, optimizer):
        if not self.constant and len(self.parents) == 0:
            v = np.mean(self.gradient, axis=tuple([*range(len(self.gradient.shape)-2)]))
            optimizer.apply_gradient(self)

        for parent in self.parents:
            parent.propagate_learning(optimizer)

    def initialize_optimizer_for_parents(self, optimizer):
        if not self.constant and len(self.parents) == 0:
            optimizer.initialize_symbol(self)

        for parent in self.parents:
            parent.initialize_optimizer_for_parents(optimizer)

    def reset_values_and_gradients(self, train=True):
        if not len(self.parents) == 0:
            self.value = None
        elif self.value_initializers is not None:
            self.value = self.value_initializers[0 if train else 1](self.shape)
        
        self.gradient = None

        for parent in self.parents:
            parent.reset_values_and_gradients()

    def _op_f(self):
        return self.value

    def _op_b(self, idx, childs_gradient):
        return childs_gradient


#
# Helper functions
#

def ensure_tuple(d):
    '''
    This function returns the input if it is a tuple, and a tuple containing the input if the input is an integer.
    '''
    if type(d) is tuple:
        return d

    if type(d) is not int:
        raise Exception("Expected a tuple or an integer")

    return (d,)

def remove_trailing_ones(shape):
    '''
    Removes the trailing ones from a tuple.
    '''
    i = 0
    while len(shape) > 0 and shape[-1] == 1:
        shape = shape[:-1]
        i += 1
    return shape, i

def np_parallel_transpose(A):
    '''
    Given a stack of matrices, this function returns the stack of matrix transposes.
    '''
    return np.einsum("...ij->...ji", A)

