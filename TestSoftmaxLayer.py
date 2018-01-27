import unittest
from numpy.testing import assert_array_equal
import numpy as np
from SoftmaxLayer import SoftmaxOutputLayer

class SoftmaxLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.softmaxLayer = SoftmaxOutputLayer()

    def tearDown(self):
        del self.softmaxLayer

    def test_get_output(self):
        assert_array_equal(self.softmaxLayer.get_output(np.array(0.0).reshape((1,1))), np.array(1.0).reshape((1,1)))

    def test_get_input_grad(self):
        assert_array_equal(self.softmaxLayer.get_input_grad(np.array([-1,1]),np.array([0.5, 0.5])), np.array([-0.75, 0.25]))

    def test_get_cost(self):
        assert_array_equal(self.softmaxLayer.get_cost(np.array([1,1]),np.array([0.5, 0.5])), 0)