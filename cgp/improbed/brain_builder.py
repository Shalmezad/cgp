from dataclasses import dataclass

from cgp.gene import GeneBuilder, GeneBuilderConfig, OpSets
from .brain import Brain


@dataclass
class BrainBuilderConfig:
    num_inputs: int
    num_outputs: int
    ops: list


class BrainBuilder:
    def __init__(self) -> None:
        op_set = OpSets.IMPROBED_2022
        soma_builder_config = GeneBuilderConfig(
            10,
            200,
            4,
            op_set
        )
        self.soma_builder = GeneBuilder(soma_builder_config)
        dendrite_builder_config = GeneBuilderConfig(
            9,
            200,
            4,
            op_set
        )
        self.dendrite_builder = GeneBuilder(dendrite_builder_config)

    def build(self) -> Brain:
        soma_program = self.soma_builder.makeGene()
        dendrite_program = self.dendrite_builder.makeGene()
        brain = Brain()
        brain.somaProgram = soma_program
        brain.dendriteProgram = dendrite_program
        return brain
