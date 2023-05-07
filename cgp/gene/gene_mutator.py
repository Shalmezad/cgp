from dataclasses import dataclass
import random
from .gene import Gene

@dataclass
class GeneMutatorConfig:
    mutation_rate: float

class GeneMutator:
    def __init__(self, config:GeneMutatorConfig) -> None:
        self.config = config

    def mutateGene(self, g: Gene) -> Gene:
        num_inputs = g.num_inputs
        middlenodes = []
        for idx, middlenode in enumerate(g.middlenodes):
            maxIdx = num_inputs + idx
            in1idx, in2idx, in3idx, op = middlenode
            if random.random() < self.config.mutation_rate:
                in1idx = random.randrange(maxIdx)
            if random.random() < self.config.mutation_rate:
                in2idx = random.randrange(maxIdx)
            if random.random() < self.config.mutation_rate:
                in3idx = random.randrange(maxIdx)
            if random.random() < self.config.mutation_rate:
                op = random.randrange(len(g.ops))
            middlenodes.append((in1idx, in2idx, in3idx, op))
        output_idxes = []
        for idx in g.output_idxes:
            if random.random() < self.config.mutation_rate:
                output_idxes.append(random.randrange(num_inputs + len(middlenodes)))
            else:
                output_idxes.append(idx)
        return Gene(num_inputs, middlenodes, output_idxes, g.ops)
