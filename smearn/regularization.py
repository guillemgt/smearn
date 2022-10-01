'''
The `smearn.regularization` module includes a few regularization techniques, such as L1 and L2 regularization and bagging ensembles. The `smearn.layers` module also includes dropout layers.
'''


from .symbolic import *

class Regularizer:
    '''
    Regularizer base class.
    A class that inherits from it to be used as a regularizer while training a `smearn.models.Model` should overload the method `gradient` to send the gradient of the regularizer function with respect to the `weights` argument.
    See `smearn.regularization.L1` or `smearn.regularization.L2` for an example.
    '''

    def gradient(weights):
        return np.zeros(weights.value.shape)

class L2(Regularizer):
    '''
    L2 regularization
    '''
    def __init__(self, alpha):
        self.alpha = alpha

    def gradient(self, weights):
        return self.alpha * weights

class L1(Regularizer):
    '''
    L1 regularization
    '''
    def __init__(self, alpha):
        self.alpha = alpha

    def gradient(self, weights):
        return self.alpha * np.sign(weights)

class BaggingEnseble:
    '''
    A class to be used to train and evaluate ensembles of neural networks using the bagging technique.
    '''
    def __init__(self, models):
        for model in models:
            if model.input.shape != models[0].input.shape or model.output.shape != models[0].output.shape:
                raise Exception("All models in an ensemble must have the same input and output shapes")

        self.models = models

    def train(self, data, labels=None, lr=0.001, epochs=1, batch_size=None, validation_data=None, validation_labels=None, validation_split=0, early_stopping=0):
        if validation_split > 0 and validation_data is not None:
            raise Exception("Do not provide a validation split if you are providing validation data")

        if validation_split > 0:
            if validation_split < 1:
                validation_split = validation_split * data.shape[0]
            validation_split = int(validation_split)
            validation_data = data[:validation_split]
            data = data[validation_split:]
            validation_labels = labels[:validation_split]
            labels = labels[validation_split:]

        if early_stopping > 0 and validation_data is None:
            raise Exception("Early stopping is not allowed without providing validation data or a validation split")

        random_indices = np.random.choice(data.shape[0], size=(len(self.models), data.shape[0]))

        consecutive_worsening_epochs = 0
        last_validation_loss = np.infty

        for epoch in range(epochs):
            for i, model in enumerate(self.models):
                random_data = data[random_indices[i]]
                random_labels = labels[random_indices[i]]
                model.train(data=random_data, labels=random_labels, lr=lr, epochs=epoch+1, start_epoch=epoch, batch_size=batch_size)

            # Compute validation loss and check if we should use early stopping
            if validation_data is not None:
                total_loss = 0
                for i, model in enumerate(self.models):
                    model.loss.reset_values_and_gradients(train=False)
                    model.input.set_value(validation_data)
                    model.labels.set_value(validation_labels.reshape(validation_labels.shape + (1,)))
                    model.loss.compute_value()
                    total_loss += np.mean(model.loss.value)

                validation_loss = total_loss / len(self.models)
                print("Mean validation loss: {}".format(validation_loss))

                if early_stopping > 0:
                    if validation_loss > last_validation_loss:
                        consecutive_worsening_epochs += 1
                    else:
                        consecutive_worsening_epochs = 0
                    
                    if consecutive_worsening_epochs >= early_stopping:
                        print("Training stopped because of early stopping")
                        return

                last_validation_loss = validation_loss

    def evaluate(self, data):
        return np.mean(np.array([model.evaluate(data) for model in self.models]), axis=0)