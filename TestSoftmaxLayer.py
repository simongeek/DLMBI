import unittest
from SoftmaxLayer import SoftmaxOutputLayer

class ReluLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.softmaxLayer = SoftmaxOutputLayer()

    def tearDown(self):
        self.softmaxLayer.dispose()

    def test_get_output(self):
        self.assertEqual(self.softmaxLayer.get_output(0), 1)

    def test_get_input_grad(self):
        self.assertEqual(self.softmaxLayer.get_input_grad([-1,1],[0.5, 0.5]), [-1.5, 0.5])

    def test_get_cost(self):
        self.assertEqual(self.softmaxLayer.get_cost([1,1],[0.5, 0.5]), 0)