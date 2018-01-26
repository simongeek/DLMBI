from Layer import Layer
import numpy as np
from sklearn.utils.extmath import softmax

#Apply softMax function
class SoftmaxOutputLayer(Layer):


    def get_output(self, X):
        return softmax(X)

    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        return - np.float64(np.multiply(T, np.log(Y)).sum()) / Y.shape[0]
