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