from dataclasses import dataclass
import random
from .gene import Gene
from .gene_mutator_base import GeneMutatorBase
from .op_sets import OpSets


@dataclass
class PointMutatorConfig:
    mutation_rate: float


class PointMutator(GeneMutatorBase):
    def __init__(self, config: PointMutatorConfig) -> None:
        self.config = config

    def mutateGene(self, g: Gene) -> Gene:
        num_inputs = g.num_inputs
        middlenodes = []
        ops = OpSets.OPSET_DICT[g.opset_key]
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
                op = random.randrange(len(ops))
            middlenodes.append((in1idx, in2idx, in3idx, op))
        output_idxes = []
        for idx in g.output_idxes:
            if random.random() < self.config.mutation_rate:
                output_idx_range = num_inputs + len(middlenodes)
                output_idxes.append(random.randrange(output_idx_range))
            else:
                output_idxes.append(idx)
        return Gene(num_inputs, middlenodes, output_idxes, g.opset_key)
