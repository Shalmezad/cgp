from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
class Gene:
    NUM_OPS = 4

    num_inputs: int
    middlenodes: tuple[int]
    output_idxes: list[int]

    def evaluateBatch(self, input: npt.ArrayLike):
        result = [self.evaluateNodeBatch(x, input) for x in self.output_idxes]
        result = np.asarray(result)
        result = np.swapaxes(result, 0,1)
        return result
    
    def evaluateNodeBatch(self, nodeIdx, input):
        if nodeIdx < self.num_inputs:
            return input[:,nodeIdx]
        else:
            nodeIdx = nodeIdx - self.num_inputs
            middleNode = self.middlenodes[nodeIdx]
            return self.evaluateMiddleNodeBatch(middleNode, input)
        
    def evaluateMiddleNodeBatch(self, middleNode, input):
        in1idx, in2idx, in3idx, op = middleNode
        in1 = self.evaluateNodeBatch(in1idx, input)
        in2 = self.evaluateNodeBatch(in2idx, input)
        in3 = self.evaluateNodeBatch(in3idx, input)
        result = 0.0
        if op == 0:
            return np.absolute(in1)
        elif op == 1:
            return np.sqrt(np.absolute(in1))
        elif op == 2:
            return np.power(in1, 2)
        elif op == 3:
            return np.power(in1, 3)
        return result
