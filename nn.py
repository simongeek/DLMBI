from sklearn import  metrics
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import collections
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from LinearLayer import LinearLayer
from ReluLayer import ReluLayer
from SoftmaxLayer import SoftmaxOutputLayer

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
def evaluate(test_labels, predictions, target_labels):
    report = classification_report(test_labels, predictions, target_names=target_labels)
    print(report)
    print(type(report))
    print(len(report))
    cm = confusion_matrix(test_labels, predictions)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    np.set_printoptions(precision=2)
    acc = accuracy_score(test_labels, predictions)
    print("Quality numbers")
    #score_matrix = [np.transpose(target_labels), np.transpose(TPR), np.transpose(TNR), np.transpose(ACC), np.transpose(PPV)]
    score_labels = ['', "Sensitivity", 'specificity', 'Accuracy', 'Precision']
    score_matrix = np.concatenate((np.array(target_labels), TPR, TNR, ACC, PPV))
    score_matrix = score_matrix.reshape(5,5)
    score_matrix = score_matrix.transpose()
    print(score_labels)
    print(score_matrix)

    plot.figure()
    plot_confusion_matrix(cm, target_labels, normalize=True)
    plot.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')


def neural_network(dataset, hidden_layers):
    print("len(hidden_layers): %d"% len(hidden_layers))
    if(len(hidden_layers) < 1):
        print("Please provide at least one hidden layer dimention")
        exit(1)

    #params:
    batch_size =64
    max_nb_of_iterations = 300
    learning_rate = 0.01

    #data = np.array(dataset.iloc[:, 3:-1])
    #idx = np.random.randint(2, len(dataset), size=3000)
    data = np.array(dataset.iloc[:, 3:1000])
    target = dataset.iloc[:,-1]
    del dataset

    target_labels = np.unique(target)

    #Convert target to output softmax layer format
    T = np.array(pd.get_dummies(pd.Series(target)))
    del target

    # Divide the data into a train and test set.
    X_train, X_test, T_train, T_test = train_test_split(
        data, T, test_size=0.4, random_state=42)
    del data, T
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
        validation_costs.append((validation_cost))
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

    evaluate(y_true, y_pred, target_labels)
    print(len(y_true))
    print(y_pred)
    print(y_true)

    return
