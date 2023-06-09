import random

from cgp.gene import GeneBuilder, GeneBuilderConfig, OpsetKey
from .brain import Brain
from .config import Config
from .dendrite import Dendrite
from .neuron import Neuron
from .point2d import Point2d


class BrainBuilder:
    def __init__(self, config: Config) -> None:
        op_set_key = OpsetKey.IMPROBED_2022_OPSET_KEY
        soma_builder_config = GeneBuilderConfig(
            9,
            200,
            4,
            op_set_key
        )
        self.soma_builder = GeneBuilder(soma_builder_config)
        dendrite_builder_config = GeneBuilderConfig(
            9,
            200,
            4,
            op_set_key
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

        for i in range(len(self.config.num_outputs)):
            num_out_problem_i = self.config.num_outputs[i]
            for _ in range(num_out_problem_i):
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
                    i+1  # Has to be +1 as indexes start at 0
                )
                initial_neurons.append(neuron)

        input_locations = []
        for input_count in self.config.num_inputs:
            for _ in range(input_count):
                # Make a random input point:
                p = Point2d(
                    random.random() * -1.0,
                    random.random() * 2 - 1)
                input_locations.append(p)

        brain = Brain(
            soma_program,
            dendrite_program,
            initial_neurons,
            input_locations,
            self.config
        )
        return brain
