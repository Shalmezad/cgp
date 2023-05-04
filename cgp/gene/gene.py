import math

import numpy as np
import numpy.typing as npt


class Gene:
    NUM_OPS = 4

    def __init__(self) -> None:
        self.num_inputs = 0
        self.middlenodes = []
        self.output_idxes = []
        pass

    def evaluateBatch(self, input: npt.ArrayLike):
        # TODO: Come up with better mechanism rather than running on each sample:
        # return np.apply_along_axis(self.evaluateSingle, 1, input)
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
    
    def __str__(self) -> str:
        return "<cgp.gene.gene.Gene middleNodes: {} outputIdx: {}".format(self.middlenodes, self.output_idxes)