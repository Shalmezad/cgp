from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
class Gene:
    NUM_OPS = 16

    num_inputs: int
    middlenodes: tuple[int]
    output_idxes: list[int]

    def evaluate(self, input: npt.ArrayLike):
        result = [self.evaluateNode(x, input) for x in self.output_idxes]
        result = np.asarray(result)
        result = np.swapaxes(result, 0,1)
        return result
    
    def evaluateNode(self, nodeIdx, input):
        if nodeIdx < self.num_inputs:
            return input[:,nodeIdx]
        else:
            nodeIdx = nodeIdx - self.num_inputs
            middleNode = self.middlenodes[nodeIdx]
            return self.evaluateMiddleNode(middleNode, input)
        
    def evaluateMiddleNode(self, middleNode, input):
        in1idx, in2idx, in3idx, op = middleNode
        in1 = self.evaluateNode(in1idx, input)
        in2 = self.evaluateNode(in2idx, input)
        in3 = self.evaluateNode(in3idx, input)
        if op == 0:
            return np.absolute(in1)
        elif op == 1:
            return np.sqrt(np.absolute(in1))
        elif op == 2:
            return np.power(in1, 2)
        elif op == 3:
            return np.power(in1, 3)
        elif op == 4:
            return (2 * np.exp(in1 + 1) - math.pow(math.e,2) - 1) / (math.pow(math.e,2) - 1)
        elif op == 5:
            return np.sin(in1)
        elif op == 6:
            return np.cos(in1)
        elif op == 7:
            return np.tanh(in1)
        elif op == 8:
            return in1 * -1
        elif op == 9:
            return np.where(in1 < 0.0, 0.0, 1.0)
        elif op == 10:
            return np.sqrt((np.power(in1, 2) + np.power(in2,2))/2.0)
        elif op == 11:
            return (in1 + in2) / 2.0
        elif op == 12:
            return (in1 - in2) / 2.0
        elif op == 13:
            return in1 * in2
        elif op == 14:
            return np.maximum(in1, in2)
        elif op == 15:
            return np.minimum(in1, in2)
        raise ValueError("Unknown operator: {}".format(op))
    
    def nodeToHumanFormula(self, nodeIdx):
        if nodeIdx < self.num_inputs:
            return "in{}".format(nodeIdx)
        else:
            nodeIdx = nodeIdx - self.num_inputs
            middleNode = self.middlenodes[nodeIdx]
            in1idx, in2idx, in3idx, op = middleNode
            in1 = self.nodeToHumanFormula(in1idx)
            in2 = self.nodeToHumanFormula(in2idx)
            in3 = self.nodeToHumanFormula(in3idx)
            if op == 0:
                return "|{}|".format(in1)
            elif op == 1:
                return "sqrt(|{}|)".format(in1)
            elif op == 2:
                return "{}^2".format(in1)
            elif op == 3:
                return "{}^3".format(in1)
            elif op == 4:
                return "exp({})".format(in1)
            elif op == 5:
                return "sin({})".format(in1)
            elif op == 6:
                return "cos({})".format(in1)
            elif op == 7:
                return "tanh({})".format(in1)
            elif op == 8:
                return "-{}".format(in1)
            elif op == 9:
                return "step({})".format(in1)
            elif op == 10:
                return "hyp({}, {})".format(in1, in2)
            elif op == 11:
                return "({} + {})/2.0".format(in1, in2)
            elif op == 12:
                return "({} - {})/2.0".format(in1, in2)
            elif op == 13:
                return "{} * {}".format(in1, in2)
            elif op == 14:
                return "max({},{})".format(in1, in2)
            elif op == 15:
                return "max({},{})".format(in1, in2)
            else:
                return "unknown{}".format(op)

    
    def toHumanFormula(self):
        return [self.nodeToHumanFormula(idx) for idx in self.output_idxes]
