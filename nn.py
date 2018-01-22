from sklearn import  metrics
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import collections
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(y):
    y[y <= 0] = 0.01
    y[y > 0] = 1
    return y

from sklearn.utils.extmath import softmax
#def softmax(z):
    #return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

#Base layer class
class Layer(object):

    #Return iterator over parameters
    def get_params_iter(self):
        return []

    #Return gradient of parameters
    def get_params_grad(self, X, output_grad):
        return []

    def get_output(self, X):
        pass

    #Return gradient at inputs of layer
    def get_input_grad(self, Y, output_grad=None, T=None):

        pass

#Linear transformation
class LinearLayer(Layer):

    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)

    def get_params_iter(self):
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        return X.dot(self.W) + self.b

    def get_params_grad(self, X, output_grad):
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        del output_grad, X
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        return output_grad.dot(self.W.T)

#Apply ReLu function
class ReluLayer(Layer):

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(relu_deriv(Y), output_grad)

#Apply softMax function
class SoftmaxOutputLayer(Layer):


    def get_output(self, X):
        return softmax(X)

    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]


# Forward propagation, returns activations for each layer
def forward_step(input_samples, layers):
    activations = [input_samples]

    X = input_samples
    del input_samples
    for layer in layers:
        #Y = layer.get_output(X)  # Get the output of the current layer
        #activations.append(Y)  # Store the output for future processing
        #X = activations[-1]  # Set the current input as the activations of the previous layer
         Y = layer.get_output(activations[-1])
         activations.append(Y)
    return activations

#Backward propagation, return parameters gradient
def backward_step(activations, targets, layers):
    param_grads = collections.deque()
    output_grad = None

    for layer in reversed(layers):
        Y = activations.pop()
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:
            input_grad = layer.get_input_grad(Y, output_grad)
        X = activations[-1]
        grads = layer.get_params_grad(X, output_grad)
        del X
        param_grads.appendleft(grads)
        del grads
        output_grad = input_grad
    return list(param_grads)


#Update parameters according to given gradient
def update_params(layers, param_grads, learning_rate):
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            param -= learning_rate * grad  # Update each parameter

#Evaluate network results
def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


def neural_network(dataset, hidden_layers):
    print("len(hidden_layers): %d"% len(hidden_layers))
    if(len(hidden_layers) < 1):
        print("Please provide at least one hidden layer dimention")
        exit(1)

    #params:
    batch_size = 16
    max_nb_of_iterations = 5
    learning_rate = 0.1

    data = np.array(dataset.iloc[:, 3:-1])
    target = dataset.iloc[:,-1]
    del dataset

    #Convert target to output softmax layer format
    T = np.array(pd.get_dummies(pd.Series(target)))
    del target

    # Divide the data into a train and test set.
    X_train, X_test, T_train, T_test = train_test_split(
        data, T, test_size=0.4, random_state=42)
    del data
    # Divide the test set into a validation set and final test set.
    X_validation, X_test, T_validation, T_test = train_test_split(
        X_test, T_test, test_size=0.5, random_state=42)

    print("in_dim: %d" % X_train.shape[1])
    print("out_dim %d" % T_train.shape[1])

    #Create layers
    layers = []
    # Add first hidden layer
    hidden_neurons_1 = hidden_layers[0]
    hidden_neurons_last = hidden_layers[-1]
    print("hidden_neurons_1: %d"%hidden_neurons_1)
    print("layer in")
    print("neurons %d , %d" % (X_train.shape[1], hidden_neurons_1))
    layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1))
    layers.append(ReluLayer())
    # Add middle hidden layers
    if (len(hidden_layers) > 1):
        print("Create hidden layers")
        for i in range(1, (len(hidden_layers))):
            print("layer %d" % i)
            print("hidden_neurons %d , %d" % (hidden_layers[i-1], hidden_layers[i]))
            layers.append(LinearLayer(hidden_layers[i-1], hidden_layers[i]))
            layers.append(ReluLayer())
    # Add output layer
    print("layer last")
    print("neurons %d , %d" % (hidden_neurons_last, T_train.shape[1]))
    layers.append(LinearLayer(hidden_neurons_last, T_train.shape[1]))
    layers.append(SoftmaxOutputLayer())

    # Create the minibatches
    nb_of_batches = X_train.shape[0] / batch_size
    X_batch = np.array_split(X_train, nb_of_batches, axis=0)
    Y_batch = np.array_split(T_train, nb_of_batches, axis=0)
    '''
    XT_batches = zip(
        np.array_split(X_train, nb_of_batches, axis=0),
        np.array_split(T_train, nb_of_batches, axis=0))
    '''
    # Perform backpropagation
    #minibatch_costs = []
    training_costs = []
    validation_costs = []

    # Train for the maximum number of iterations
    for iteration in range(max_nb_of_iterations):
        print("Iteration: %d~~~~~~~~~~~~~~~~~~~~~~" % iteration)
        batch_nr = 0
        XT_batches = zip(X_batch, Y_batch)
        for X, T in XT_batches:  # For each minibatch sub-iteration
            batch_nr = batch_nr + 1
            print("Batch nr:%d from %d" % (batch_nr, nb_of_batches))
            activations = forward_step(X, layers)  # Get the activations
            #minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
            #minibatch_costs.append(minibatch_cost)
            #del minibatch_cost
            param_grads = backward_step(activations, T, layers)
            update_params(layers, param_grads, learning_rate)
        # Get full training cost for future analysis (plots)
        activations = forward_step(X_train, layers)
        train_cost = layers[-1].get_cost(activations[-1], T_train)
        training_costs.append(train_cost)
        del train_cost
        # Get full validation cost
        activations = forward_step(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], T_validation)
        validation_costs.append(validation_cost)
        del validation_cost

        if len(validation_costs) > 3:
            # Stop training if the cost on the validation set doesn't decrease
            #  for 3 iterations
            #if [(validation_costs[-1] >= validation_costs[-2]) & (validation_costs[-2] >= validation_costs[-3])]:
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break

    nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    print("Finished at iteration: %d" % nb_of_iterations)
    # Get results of test data
    y_true = np.argmax(T_test, axis=1)  # Get the target outputs
    activations = forward_step(X_test, layers)  # Get activation of test samples
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    '''
    test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
    test_f1 = metrics.f1_score(y_true, y_pred)
    test_recall = metrics.recall_score(y_true, y_pred)
    print('The accuracy on the test set is {:.2f}'.format(test_accuracy))
    print('The f1 on the test set is {:.2f}'.format(test_f1))
    print('The recall on the test set is {:.2f}'.format(test_recall))
    '''
    evaluate(y_true, y_pred)

    print("validation_costs: ")
    print(validation_costs)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.plot(validation_costs)
    plt.ylabel('validation_costs')

    plt.subplot(212)
    plt.plot(training_costs)
    plt.ylabel('training_costs')
    plt.show()

    return
