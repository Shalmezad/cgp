import random
from .gene import Gene
from .gene_mutator_base import GeneMutatorBase
from .op_sets import OpSets


class GoldmanMutator(GeneMutatorBase):

    def mutateGene(self, g: Gene) -> Gene:
        middlenodes = []
        for idx, middlenode in enumerate(g.middlenodes):
            in1idx, in2idx, in3idx, op = middlenode
            middlenodes.append((in1idx, in2idx, in3idx, op))
        new_gene = Gene(
            g.num_inputs,
            middlenodes,
            g.output_idxes,
            g.opset_key
        )

        # Mutate until something used changes (ie: the formula changes):
        while g.toHumanFormula() == new_gene.toHumanFormula():
            # Are we changing a middle node or output node:
            if random.random() < 0.8:
                # Middle node
                middlenodes = new_gene.middlenodes
                idx = random.randrange(len(middlenodes))
                maxIdx = g.num_inputs + idx
                in1idx, in2idx, in3idx, op = middlenodes[idx]
                ops = OpSets.OPSET_DICT[g.opset_key]

                which_val = random.choices(['1', '2', '3', 'op'])
                if which_val == '1':
                    in1idx = random.randrange(maxIdx)
                elif which_val == '2':
                    in2idx = random.randrange(maxIdx)
                elif which_val == '3':
                    in3idx = random.randrange(maxIdx)
                else:
                    op = random.randrange(len(ops))
                middlenodes[idx] = (in1idx, in2idx, in3idx, op)
                new_gene = Gene(
                    new_gene.num_inputs,
                    middlenodes,
                    new_gene.output_idxes,
                    new_gene.opset_key
                )
            else:
                # Output node
                output_idxes = new_gene.output_idxes
                idx = random.randrange(len(output_idxes))
                output_idx_range = new_gene.num_inputs + len(
                    new_gene.middlenodes)
                output_idxes[idx] = random.randrange(output_idx_range)
                new_gene = Gene(
                    new_gene.num_inputs,
                    new_gene.middlenodes,
                    output_idxes,
                    new_gene.opset_key
                )
        return new_gene
