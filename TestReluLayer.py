import unittest
from ReluLayer import ReluLayer, relu_deriv, relu

class ReluLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.reluLayer = ReluLayer()

    def tearDown(self):
        self.reluLayer.dispose()

    def test_get_output(self):
        self.assertEqual(self.reluLayer.get_output([-1,1]), [[0,1]])

    def test_get_input_grad(self):
        self.assertEqual(self.reluLayer.get_input_grad([-1,1],[0.5, 0.5]), [0.005, 0.5])

    def test_relu(self):
        self.assertEqual(relu(-1), 0)
        self.assertEqual(relu(1), 1)

    def test_relu_deriv(self):
        self.assertEqual(relu_deriv([-1,1]), [0.01, 1])