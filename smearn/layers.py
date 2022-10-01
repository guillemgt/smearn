'''
The module `smearn.layers` contains many of the most common operations used in computation graphs for neural networks.
'''

from smearn.regularization import Regularizer
from .symbolic import *

class Input(Symbol):
    ''' A `smearn.symbolic.Symbol` representing any input to a `smearn.symbolic.Model`. '''
    def __init__(self, shape: tuple, pad_one_dimension: bool = True):
        ''' Initializes an input of shape `shape`'''
        shape = ensure_tuple(shape)
        super().__init__(shape=shape + ((1,) if pad_one_dimension else ()), value=np.zeros(shape), constant=True)

    def _op_f(self):
        pass

    def _op_b(self, idx, childs_gradient):
        pass


class ReLU(Symbol):
    ''' A `smearn.symbolic.Symbol` representing a rectified linear unit. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.ReLU must be a Symbol")
        super().__init__(shape=X.shape, parents=[X])

    def _op_f(self):
        return np.clip(self.parents[0].value, a_min=0, a_max=None)

    def _op_b(self, idx, childs_gradient):
        return np.multiply(np.where(self.parents[idx].value > 0.0, 1, 0), childs_gradient)


class Add(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the addition of two tensors. '''

    def __init__(self, X, Y):
        ''''''
        if not isinstance(X, Symbol) or not isinstance(Y, Symbol):
            raise Exception("The arguments X and Y to layers.Add must be Symbols")
            
        if X.shape != Y.shape:
            raise ShapeError("Can't add symbols of shapes {} and {}".format(X.shape, Y.shape))

        super().__init__(shape=X.shape, parents=[X, Y])

    def _op_f(self):
        return np.add(self.parents[0].value, self.parents[1].value)

    def _op_b(self, idx, childs_gradient):
        return childs_gradient


class Subtract(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the subtraction of two tensors.. '''

    def __init__(self, X, Y):
        ''''''
        if not isinstance(X, Symbol) or not isinstance(Y, Symbol):
            raise Exception("The arguments X and Y to layers.Subtract must be Symbols")
            
        if X.shape != Y.shape:
            raise ShapeError("Can't subtract symbols of shapes {} and {}".format(X.shape, Y.shape))

        super().__init__(shape=X.shape, parents=[X, Y])

    def _op_f(self):
        return np.subtract(self.parents[0].value, self.parents[1].value)

    def _op_b(self, idx, childs_gradient):
        return childs_gradient if idx == 0 else -childs_gradient


class HadamardProduct(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the Hadamard (i.e. elementwise) product of two tensors. '''

    def __init__(self, X, Y):
        ''''''
        if not isinstance(X, Symbol) or not isinstance(Y, Symbol):
            raise Exception("The arguments X and Y to layers.HadamardProduct must be Symbols")
            
        if X.shape != Y.shape:
            raise ShapeError("Can't take Hadamard product of symbols of shapes {} and {}".format(X.shape, Y.shape))

        super().__init__(shape=X.shape, parents=[X, Y])

    def _op_f(self):
        return np.multiply(self.parents[0].value, self.parents[1].value)

    def _op_b(self, idx, childs_gradient):
        return np.multiply(childs_gradient, self.parents[1-idx].value)




class MatrixProduct(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the matrix of two (stacks of) matrices.

    If its first argument has shape `(...,m,n)` and its second argument has shape `(...,n,l)` where the two `...` are the same, then the resulting symbol will have shape `(...,m,l)`.
    '''

    def __init__(self, X, Y):
        ''''''
        if not isinstance(X, Symbol) or not isinstance(Y, Symbol):
            raise Exception("The arguments X and Y to layers.MatrixProduct must be Symbols")
            
        if X.shape[-1] != Y.shape[-2]:
            raise ShapeError("Can't take Hadamard product of symbols of shapes {} and {}".format(X.shape, Y.shape))

        super().__init__(shape=X.shape[:-1] + Y.shape[:-2] + (Y.shape[-1],), parents=[X, Y])

    def _op_f(self):
        return np.matmul(self.parents[0].value, self.parents[1].value)

    def _op_b(self, idx, childs_gradient):
        if idx == 0:
            return np.matmul(childs_gradient, np_parallel_transpose(self.parents[1].value))
        if idx == 1:
            return np.matmul(np_parallel_transpose(self.parents[0].value), childs_gradient)



class SelfDot(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the dot product of a column vector with itself '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.SelfDot must be a Symbol")

        if X.shape[-1] != 1:
            raise ShapeError("layers.SelfDot can only be applied to a (stack of) column vector(s)")

        super().__init__(shape=X.shape[:-2] + (1,1), parents=[X])

    def _op_f(self):
        return np.matmul(np_parallel_transpose(self.parents[0].value), self.parents[0].value)

    def _op_b(self, idx, childs_gradient):
        return np.matmul(childs_gradient, 2.0 * self.parents[0].value)

def Dense(X, output_shape, regularization=None):
    ''' Returns a `smearn.symbolic.Symbol` representing a dense layer in a neural network, i.e. a general affine transformation.
    
    `X` is the input `smearn.symbolic.Symbol`.

    `output_shape` is the shape of the output of the layer

    `regularization` is the regularization term associated to the weights of the linear transformation associated to the later. It can be one of `None`, `smearn.regularization.L1` or `smearn.regularization.L2`.
    '''

    if not isinstance(X, Symbol):
        raise Exception("The argument X to layers.Dense must be a Symbol")

    if regularization is not None and not isinstance(regularization, Regularizer) and not callable(regularization):
        raise Exception("The argument regularization to layers.Dense must be None, a Regularizer, or a function")

    output_shape = ensure_tuple(output_shape)

    while len(output_shape) <= 1:
        output_shape += (1,)

    W_shape = output_shape[:-1] + X.shape[:-1]

    W = Symbol(shape=W_shape, value=np.random.normal(0.0, 0.05, size=W_shape), regularization=regularization)
    b = Symbol(shape=output_shape, value=np.random.normal(0.1, 0.05, size=output_shape))

    return Add(MatrixProduct(W, X), b)



class Log(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the (elementwise) logarithm of a tensor. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.Log must be a Symbol")

        super().__init__(shape=X.shape, parents=[X])

    def _op_f(self):
        return np.log(1e-12 + self.parents[0].value)

    def _op_b(self, idx, childs_gradient):
        return np.divide(childs_gradient, 1e-12 + self.parents[0].value)


class Negate(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the negation of a tensor. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.Negate must be a Symbol")

        super().__init__(shape=X.shape, parents=[X])

    def _op_f(self):
        return -self.parents[0].value

    def _op_b(self, idx, childs_gradient):
        return-childs_gradient

class OneMinus(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the one minus a tensor. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.OneMinus must be a Symbol")

        super().__init__(shape=X.shape, parents=[X])

    def _op_f(self):
        return 1.0-self.parents[0].value

    def _op_b(self, idx, childs_gradient):
        return-childs_gradient


class Sigmoid(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the (elementwise) logistic sigmoid of a tensor. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.Sigmoid must be a Symbol")

        super().__init__(shape=X.shape, parents=[X])
        pass

    def _op_f(self):
        return 1.0 / (1.0 + np.exp(-self.parents[0].value))

    def _op_b(self, idx, childs_gradient):
        return np.matmul(childs_gradient, np.multiply(self.value, 1.0 - self.value))


class Softmax(Symbol):
    ''' A `smearn.symbolic.Symbol` representing the softmax of a tensor along its third-to-last axis. '''

    def __init__(self, X):
        ''''''
        if not isinstance(X, Symbol):
            raise Exception("The argument X to layers.Softmax must be a Symbol")

        super().__init__(shape=X.shape, parents=[X])
        pass

    def _op_f(self):
        arg = self.parents[0].value
        max = arg.max(axis=-2).reshape(arg.shape[:-2] + (1,1))
        #max = 0.0
        #print(arg)
        exps = np.exp(arg - max)
        return exps / exps.sum(axis=-2).reshape(arg.shape[:-2] + (1,1))

    def _op_b(self, idx, childs_gradient):
        s = self.value
        t = np.transpose(s, axes=[*range(len(s.shape)-2)] + [len(s.shape)-1, len(s.shape)-2])
        s_diagonal = np.multiply(np.eye(s.shape[-2]), s)
        return np.matmul(np.subtract(s_diagonal, np.matmul(s, t)), childs_gradient) #todo multiplication in the wrong order?


def CrossEntropy(X, Y):
    ''' Returns a `smearn.symbolic.Symbol` representing the cross-entropy of two tensors. '''

    return Negate(HadamardProduct(X, Log(Y)))

def BinaryCrossEntropy(X, Y):
    ''' Returns a `smearn.symbolic.Symbol` representing the binary cross-entropy of two tensors. '''
    return Negate(Add(HadamardProduct(X, Log(Y)), HadamardProduct(OneMinus(X), Log(OneMinus(Y)))))

def MeanSquareError(X, Y):
    ''' Returns a `smearn.symbolic.Symbol` representing the square error (i.e. self-dot product of the difference of) of two tensors. '''
    return SelfDot(Subtract(X, Y))




def Dropout(X, p):
    ''' Returns a `smearn.symbolic.Symbol` adding dropout regularization to another Symbol. '''
    def _dropout_init_train(shape):
        return np.random.choice([0.0, 1.0], size=shape, p=[1-p, p])
    def _dropout_init_test(shape):
        return np.ones(shape)
    mask = Symbol(shape=X.shape, value_initializers=[_dropout_init_train, _dropout_init_test])
    return HadamardProduct(X, mask)