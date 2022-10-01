'''
The `smearn.models` module contains the `smearn.models.Model` class, which provides a wrapper to optimize feedforward neural networks using supervised networks.
'''

import numpy as np
from . import layers
from . import optimization
from .symbolic import *


#
# Models
#

class Model:
    def __init__(self, input=None, output=None, loss=layers.MeanSquareError, optimizer=optimization.SGD()):
        self.input = input
        self.output = output
        self.labels = layers.Input(output.shape, pad_one_dimension=False)
        self.loss = loss(self.labels, self.output)
        self.optimizer = optimizer
        self.loss.initialize_optimizer_for_parents(optimizer)

    def train(self, data, labels=None, lr=0.001, epochs=1, batch_size=None, validation_data=None, validation_labels=None, validation_split=0, early_stopping=0, start_epoch=0):

        # First, handle the some of the logistics of validation splits and early stopping

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

        if batch_size is None:
            batch_size = labels.shape[0]

        batches = labels.shape[0] // batch_size

        consecutive_worsening_epochs = 0
        last_validation_loss = np.infty

        for epoch in range(start_epoch, epochs):
            current_index = 0
            loss = np.zeros(())
            loss_count = 0
            for i in range(batches):
                next_index = min(current_index + batch_size, labels.shape[0])
                batch_data = data[current_index:next_index]
                batch_labels = labels[current_index:next_index]

                # Forward pass

                self.loss.reset_values_and_gradients()
                self.input.set_value(batch_data)
                self.labels.set_value(batch_labels.reshape(batch_labels.shape + (1,)))
                self.loss.compute_value()

                loss += np.mean(self.loss.value)
                loss_count += (next_index - current_index) / batch_size

                # Backward pass
                
                self.loss.propagate_gradients()
                self.loss.propagate_learning(self.optimizer)

                current_index = next_index

            
            print("[Epoch {}]: mean training loss: {}".format(epoch + 1, loss / loss_count))

            # Tell schedulers to update the hyperparameters
            self.optimizer.update_hyperparameters(epoch + 1)

            # Compute validation loss and check if we should use early stopping
            if validation_data is not None:
                self.loss.reset_values_and_gradients(train=False)
                self.input.set_value(validation_data)
                self.labels.set_value(validation_labels.reshape(validation_labels.shape + (1,)))
                self.loss.compute_value()

                validation_loss = np.mean(self.loss.value)
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
        self.output.reset_values_and_gradients(train=False)
        self.input.set_value(data)
        self.output.compute_value()

        return self.output.value