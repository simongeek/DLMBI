import unittest
import numpy as np
from numpy.testing import assert_array_equal
from LinearLayer import LinearLayer

class LinearLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.linearLayer = LinearLayer(2,4)
        self.linearLayer.W = np.array([[0.1, 0.1, 0.1, 0.1], [0.15, 0.15, 0.15, 0.15]])

    def tearDown(self):
        del self.linearLayer

    def test_get_output(self):
        assert_array_equal(self.linearLayer.get_output(np.array([1,2])), np.array([0.4, 0.4, 0.4, 0.4]))

    def test_get_params_grad(self):
        assert_array_equal(self.linearLayer.get_params_grad(np.array([1,2]).reshape((2, 1)), np.array([1,0])), np.array((1,1)))

    def test_get_input_grad(self):
        assert_array_equal(self.linearLayer.get_input_grad([], np.array([0,0,1,0])), np.array([0.1, 0.15]))
