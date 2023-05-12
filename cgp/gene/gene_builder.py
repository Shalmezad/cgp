from dataclasses import dataclass
import random
from .gene import Gene


@dataclass
class GeneBuilderConfig:
    num_inputs: int
    num_middlenodes: int
    num_outputs: int
    ops: list


class GeneBuilder:
    def __init__(self, config: GeneBuilderConfig) -> None:
        self.config = config

    def makeGene(self) -> Gene:
        num_inputs = self.config.num_inputs
        middlenodes = [
            self.makeMiddleNode(x) for x in range(self.config.num_middlenodes)]
        output_idx_range = self.config.num_inputs + self.config.num_middlenodes
        output_idxes = [
            random.randrange(output_idx_range)
            for x in range(self.config.num_outputs)]
        return Gene(num_inputs, middlenodes, output_idxes, self.config.ops)

    def makeMiddleNode(self, middleIdx) -> tuple[int, int, int, int]:
        maxIdx = self.config.num_inputs + middleIdx
        in1idx = random.randrange(maxIdx)
        in2idx = random.randrange(maxIdx)
        in3idx = random.randrange(maxIdx)
        op = random.randrange(len(self.config.ops))
        return (in1idx, in2idx, in3idx, op)
