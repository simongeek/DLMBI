import unittest
from numpy.testing import assert_array_equal
import numpy as np
from ReluLayer import ReluLayer, relu_deriv, relu

class ReluLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.reluLayer = ReluLayer()

    def tearDown(self):
        del self.reluLayer

    def test_get_output(self):
        assert_array_equal(self.reluLayer.get_output(np.array([-1,1])), np.array([0,1]))

    def test_get_input_grad(self):
        assert_array_equal(self.reluLayer.get_input_grad(np.array([-1,1]),np.array([0.5, 0.5])), np.array([0, 0.5]))

    def test_relu(self):
        self.assertEqual(relu(-1), 0)
        self.assertEqual(relu(1), 1)

    def test_relu_deriv(self):
        assert_array_equal(relu_deriv(np.array([-1,1])), np.array([0, 1]))