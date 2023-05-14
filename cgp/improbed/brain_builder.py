import random

from cgp.gene import GeneBuilder, GeneBuilderConfig, OpSets
from .brain import Brain
from .config import Config
from .dendrite import Dendrite
from .neuron import Neuron
from .point2d import Point2d


class BrainBuilder:
    def __init__(self, config: Config) -> None:
        op_set = OpSets.IMPROBED_2022
        soma_builder_config = GeneBuilderConfig(
            8,
            200,
            4,
            op_set
        )
        self.soma_builder = GeneBuilder(soma_builder_config)
        dendrite_builder_config = GeneBuilderConfig(
            8,
            200,
            4,
            op_set
        )
        self.dendrite_builder = GeneBuilder(dendrite_builder_config)
        self.config = config

    def build(self) -> Brain:
        soma_program = self.soma_builder.makeGene()
        dendrite_program = self.dendrite_builder.makeGene()

        initial_neurons = []
        for _ in range(self.config.initial_non_output_neurons):
            dendrites = []
            for _ in range(self.config.initial_num_dendrites):
                # Dendrites initialized with random
                #   health, weight, and position
                dendrite = Dendrite(
                    random.random() * 2 - 1,
                    random.random() * 2 - 1,
                    Point2d(random.random(), random.random())
                )
                dendrites.append(dendrite)
            # Neurons are *initialized* with random values
            neuron = Neuron(
                random.random() * 2 - 1,
                Point2d(random.random() * 2 - 1, random.random() * 2 - 1),
                random.random() * 2 - 1,
                dendrites,
                0
            )
            initial_neurons.append(neuron)

        brain = Brain(
            soma_program,
            dendrite_program,
            initial_neurons,
            self.config
        )
        return brain
