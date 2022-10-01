'''
The `smearn.optimization` module includes some gradient-based optimization techniques for neural networks (namely, stochastic gradient descent, stochastic gradient descent with momentum, AdaGrad, RMSProp and Adam), as well as a system for scheduling hyperparameter changes such as learning rate scheduling.
'''

from .symbolic import *

# Schedulers

def wrap_in_scheduler(x):
    return x if isinstance(x, Scheduler) else Scheduler(x)

class Scheduler:
    '''
    The base class for schedulers. The `update` method is called after each epoch during training, and this method updates its attribute `value`, whose value will be used as the learning rate during the next epoch.
    '''
    def __init__(self, value):
        self.value = value

    def update(self, step):
        pass

class LinearDecrease(Scheduler):
    '''
    Scheduler for a linear decrease.
    '''
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.value = self.start

    def update(self, step):
        if step > self.steps:
            self.value = self.end
        else:
            t = step / self.steps
            self.value = (1-t)*self.start + t*self.end


# Gradient-based optimization methods

class SGD:
    '''
    SGD optimizer
    '''
    def __init__(self, lr=0.001):
        self.lr = wrap_in_scheduler(lr)

    def initialize_symbol(self, symbol):
        pass

    def apply_gradient(self, symbol):
        symbol.value = symbol.value - self.lr.value * np.mean(symbol.gradient, axis=0)

    def update_hyperparameters(self, epoch):
        self.lr.update(epoch)

        
class SGDMomentum:
    '''
    SGD with momentum optimizer
    '''
    def __init__(self, lr=0.001, beta=0.5):
        self.lr = wrap_in_scheduler(lr)
        self.lr.value *= (1-beta)
        self.beta = wrap_in_scheduler(beta)
        self.momentums = {}

    def initialize_symbol(self, symbol):
        if self.momentums.get(symbol) is not None:
            return False
        self.momentums[symbol] = np.zeros(symbol.shape)
        return True

    def apply_gradient(self, symbol):
        self.momentums[symbol] = self.beta.value * self.momentums[symbol] - self.lr.value * np.mean(symbol.gradient, axis=0)
        symbol.value = symbol.value + self.momentums[symbol]

    def update_hyperparameters(self, epoch):
        self.lr.update(epoch)
        self.beta.update(epoch)

class AdaGrad:
    '''
    AdaGrad optimizer
    '''
    def __init__(self, lr=0.001, delta=1e-7):
        self.lr = wrap_in_scheduler(lr)
        self.delta = wrap_in_scheduler(delta)
        self.accumulated_square_gradients = {}

    def initialize_symbol(self, symbol):
        if self.accumulated_square_gradients.get(symbol) is not None:
            return False
        self.accumulated_square_gradients[symbol] = np.zeros(symbol.shape)
        return True

    def apply_gradient(self, symbol):
        gradient = np.mean(symbol.gradient, axis=0)
        self.accumulated_square_gradients[symbol] += np.multiply(gradient, gradient)
        symbol.value = symbol.value - self.lr.value / (self.delta.value + np.sqrt(self.accumulated_square_gradients[symbol])) * gradient

    def update_hyperparameters(self, epoch):
        self.lr.update(epoch)
        self.delta.update(epoch)

class RMSProp:
    '''
    RMSProp optimizer
    '''
    def __init__(self, lr=0.001, beta=0.5, delta=1e-6):
        self.lr = wrap_in_scheduler(lr)
        self.beta = wrap_in_scheduler(beta)
        self.delta = wrap_in_scheduler(delta)
        self.accumulations = {}

    def initialize_symbol(self, symbol):
        if self.accumulations.get(symbol) is not None:
            return False
        self.accumulations[symbol] = np.zeros(symbol.shape)
        return True

    def apply_gradient(self, symbol):
        gradient = np.mean(symbol.gradient, axis=0)
        self.accumulations[symbol] = self.beta.value * self.accumulations[symbol] + (1 - self.beta.value) * np.multiply(gradient, gradient)
        symbol.value = symbol.value - self.lr.value / (np.sqrt(self.delta.value + self.accumulations[symbol])) * gradient

    def update_hyperparameters(self, epoch):
        self.lr.update(epoch)
        self.beta.update(epoch)
        self.delta.update(epoch)

class Adam:
    '''
    Adam optimizer
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, delta=1e-8):
        self.lr = wrap_in_scheduler(lr)
        self.beta_1 = wrap_in_scheduler(beta_1)
        self.beta_2 = wrap_in_scheduler(beta_2)
        self.delta = wrap_in_scheduler(delta)
        self.first_moments = {}
        self.second_moments = {}
        self.beta_t = {}

    def initialize_symbol(self, symbol):
        if self.first_moments.get(symbol) is not None:
            return False
        self.first_moments[symbol] = np.zeros(symbol.shape)
        self.second_moments[symbol] = np.zeros(symbol.shape)
        self.beta_t[symbol] = [1,1]
        return True

    def apply_gradient(self, symbol):
        gradient = np.mean(symbol.gradient, axis=0)
        self.beta_t[symbol][0] *= self.beta_1.value
        self.beta_t[symbol][1] *= self.beta_2.value
        self.first_moments[symbol] = self.beta_1.value * self.first_moments[symbol] + (1 - self.beta_1.value) * gradient
        self.second_moments[symbol] = self.beta_2.value * self.second_moments[symbol] + (1 - self.beta_2.value) * np.multiply(gradient, gradient)
        unbiased_first_moment = self.first_moments[symbol] / (1 - self.beta_t[symbol][0])
        unbiased_second_moment = self.second_moments[symbol] / (1 - self.beta_t[symbol][1])
        symbol.value = symbol.value - self.lr.value * unbiased_first_moment / (self.delta.value + np.sqrt(unbiased_second_moment))

    def update_hyperparameters(self, epoch):
        self.lr.update(epoch)
        self.beta_1.update(epoch)
        self.beta_2.update(epoch)
        self.delta.update(epoch)
        print(self.lr.value)