import sys
sys.path.append('.')
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
        for opset in opsets:
            for op in opset:
                lamb = op[1]
                out = lamb(in1, in2, in3)
                self.assertEqual(out.shape[0], 100)

if __name__=='__main__':
	unittest.main()