from dataclasses import dataclass
import random
from .gene import Gene

@dataclass
class GeneMutatorConfig:
    mutation_rate: float

class GeneMutator:
    def __init__(self, config:GeneMutatorConfig) -> None:
        self.config = config

    def mutateGene(self, g):
        g = Gene()
        g.num_inputs = self.config.num_inputs
        g.middlenodes = [self.makeMiddleNode(x) for x in range(self.config.num_middlenodes)]
        g.output_idxes = [random.randrange(self.config.num_inputs + self.config.num_middlenodes) for x in range(self.config.num_outputs)]
        return g
    
    def makeMiddleNode(self, middleIdx):
        maxIdx = self.config.num_inputs + middleIdx
        in1idx = random.randrange(maxIdx)
        in2idx = random.randrange(maxIdx)
        in3idx = random.randrange(maxIdx)
        op = random.randrange(Gene.NUM_OPS)
        return (in1idx, in2idx, in3idx, op)