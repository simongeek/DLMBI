import unittest
from TestLinearLayer import LinearLayerTestCase
from TestReluLayer import ReluLayerTestCase
from TestSoftmaxLayer import SoftmaxLayerTestCase

def suite():
    suite = unittest.TestSuite()

    suite.addTest(LinearLayerTestCase('test_get_output'))
    suite.addTest(LinearLayerTestCase('test_get_params_grad'))
    suite.addTest(LinearLayerTestCase('test_get_input_grad'))

    suite.addTest(ReluLayerTestCase('test_get_output'))
    suite.addTest(ReluLayerTestCase('test_get_input_grad'))
    suite.addTest(ReluLayerTestCase('test_relu'))
    suite.addTest(ReluLayerTestCase('test_relu_deriv'))

    suite.addTest(SoftmaxLayerTestCase('test_get_output'))
    suite.addTest(SoftmaxLayerTestCase('test_get_input_grad'))
    suite.addTest(SoftmaxLayerTestCase('test_get_cost'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())