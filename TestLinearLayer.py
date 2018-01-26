import unittest
from LinearLayer import LinearLayer

class LinearLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.linearLayer = LinearLayer(2,4)
        self.linearLayer.W = [[0.1, 0.1, 0.1, 0.1], [0.15, 0.15, 0.15, 0.15]]

    def tearDown(self):
        self.linearLayer.dispose()

    def test_get_output(self):
        self.assertEqual(self.linearLayer.get_output([1,2]), [[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]])

    def test_get_params_grad(self):
        self.assertEqual(self.linearLayer.get_params_grad([1,2], [0,0,1,0]), [0,0,1,0,0,0,2,1,2])

    def test_get_input_grad(self):
        self.assertEqual(self.linearLayer.get_input_grad([], [0,0,1,0]), [[0.15, 0.1], [0.15, 0.1], [0.15, 0.1], [0.15, 0.1]])
