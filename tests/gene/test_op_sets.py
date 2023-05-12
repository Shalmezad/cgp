from cgp.gene import OpSets
import unittest
import numpy as np


class TestOpSets(unittest.TestCase):

    def test_sizes(self):
        # Each op should take 3 [Nx1] arrays
        # and return an [Nx1] array:
        in1 = np.ones(100)
        in2 = np.ones(100)
        in3 = np.ones(100)
        opsets = [
            OpSets.IMPROBED_2022,
            OpSets.GPTP_II
        ]
        for opset_idx, opset in enumerate(opsets):
            for op_idx, op in enumerate(opset):
                # print("Testing: {}-{}".format(opset_idx, op_idx))
                lamb = op[1]
                out = lamb(in1, in2, in3)
                self.assertEqual(out.shape[0], 100)

    def test_safe_division(self):
        in1 = np.ones(100)
        in2 = np.zeros(100)
        in3 = np.zeros(100)
        OpSets.SAFE_DIVISION[1](in1, in2, in3)

    def test_safe_log(self):
        in1 = np.zeros(100)
        in2 = np.zeros(100)
        in3 = np.zeros(100)
        OpSets.SAFE_LOG[1](in1, in2, in3)


if __name__ == '__main__':
    unittest.main()
