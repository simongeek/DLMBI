from Layer import Layer
import numpy as np

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(y):
    y[y <= 0] = 0.01
    y[y > 0] = 1
    return y

#Apply ReLu function
class ReluLayer(Layer):

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(relu_deriv(Y), output_grad)