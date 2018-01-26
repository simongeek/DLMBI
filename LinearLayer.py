from Layer import Layer
import numpy as np
import itertools

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