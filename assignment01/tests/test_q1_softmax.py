import unittest
import math

from . import softmax, per_row_softmax


class TestSoftmax(unittests.TestCase):

    def __init__(self):
        super(self).__init__()

    def test_softmax(self):
        x = [1, 2, 3]
        softmax_x = [0.0900305,  0.244728, 0.6652409]
        self.assertListEqual(softmax(x), softmax_x)

    def test_per_row_softmax(self):
        raise NotImplementError


if __main__ == "__main__":
    unittest.main()
