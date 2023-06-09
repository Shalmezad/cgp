from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .op_sets import OpSets, OpsetKey


@dataclass(frozen=True)
class Gene:
    num_inputs: int
    middlenodes: list[tuple[int, int, int, int]]
    output_idxes: list[int]
    opset_key: OpsetKey

    def evaluate(
            self,
            input: npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
        if input.ndim != 2 or input.shape[1] != self.num_inputs:
            raise ValueError(
                "Expected shape of (X,{}), received {}".format(
                    self.num_inputs,
                    input.shape))
        result = np.asarray([
            self.evaluateNode(x, input) for x in self.output_idxes])
        result = np.swapaxes(result, 0, 1)
        return result

    def evaluateNode(
            self,
            nodeIdx: int,
            input: npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
        if nodeIdx < self.num_inputs:
            return input[:, nodeIdx]
        else:
            nodeIdx = nodeIdx - self.num_inputs
            middleNode = self.middlenodes[nodeIdx]
            return self.evaluateMiddleNode(middleNode, input)

    def evaluateMiddleNode(
            self,
            middleNode: tuple[int, int, int, int],
            input: npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:

        in1idx, in2idx, in3idx, op_id = middleNode
        in1 = self.evaluateNode(in1idx, input)
        in2 = self.evaluateNode(in2idx, input)
        in3 = self.evaluateNode(in3idx, input)
        ops = OpSets.OPSET_DICT[self.opset_key]

        if op_id > len(ops):
            raise ValueError("Unknown operator: {}".format(op_id))

        op = ops[op_id]
        try:
            result = op[1](in1, in2, in3)
            result[np.isnan(result)] = 0
            return result
        except FloatingPointError as e:
            # numpy runtime warning:
            # We're going to reraise, but also post additional info:
            print("Encountered a warning in Gene#evaluateMiddleNode")
            print("middleNode: {}".format(middleNode))
            print("Input 1: {}".format(in1))
            print("Input 2: {}".format(in2))
            raise e

    def nodeToHumanFormula(self, nodeIdx: int) -> str:
        if nodeIdx < self.num_inputs:
            return "in{}".format(nodeIdx)
        else:
            nodeIdx = nodeIdx - self.num_inputs
            middleNode = self.middlenodes[nodeIdx]
            in1idx, in2idx, in3idx, op_id = middleNode
            in1 = self.nodeToHumanFormula(in1idx)
            in2 = self.nodeToHumanFormula(in2idx)
            in3 = self.nodeToHumanFormula(in3idx)
            ops = OpSets.OPSET_DICT[self.opset_key]

            if op_id > len(ops):
                raise ValueError("Unknown operator: {}".format(op_id))

            op = ops[op_id]
            return op[0](in1, in2, in3)

    def toHumanFormula(self) -> list[str]:
        return [self.nodeToHumanFormula(idx) for idx in self.output_idxes]
