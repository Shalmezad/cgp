from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cgp.util import MathUtil
from cgp.gene import Gene
from .ann import ANN
from .config import Config
from .dendrite import Dendrite
from .neuron import Neuron
from .point2d import Point2d


@dataclass(frozen=True)
class Brain:
    # The program for updating soma
    somaProgram: Gene
    # The program for updating dendrites
    dendriteProgram: Gene

    neurons: list[Neuron]

    inputLocations: list[Point2d]

    config: Config

    def update(self) -> Brain:
        newNeurons = []
        nonOutputNeurons = []
        outputNeurons = []
        for neuron in self.neurons:
            if neuron.out == 0:
                nonOutputNeurons.append(neuron)
            else:
                outputNeurons.append(neuron)
        # Process non-output neurons:
        max_non_output_neurons = (self.config.max_num_neurons -
                                  len(outputNeurons))
        for neuron in nonOutputNeurons:
            health, positionX, positionY, bias = self.runSoma(neuron)
            updatedNeuron = self.runAllDendrites(neuron,
                                                 Point2d(positionX, positionY),
                                                 health,
                                                 bias)
            if (updatedNeuron.health >
                    self.config.neuron_health_death_threshold):
                # Neuron survives
                newNeurons.append(updatedNeuron)
                if len(newNeurons) >= max_non_output_neurons:
                    break
            if (updatedNeuron.health >
                    self.config.neuron_health_birth_threshold):
                # Neuron replicates
                replicatedNeuron = self.createNewNeuron(updatedNeuron)
                newNeurons.append(replicatedNeuron)
                if len(newNeurons) >= max_non_output_neurons:
                    break
        for outputNeuron in outputNeurons:
            health, positionX, positionY, bias = self.runSoma(neuron)
            updatedNeuron = self.runAllDendrites(neuron,
                                                 Point2d(positionX, positionY),
                                                 health,
                                                 bias)
            newNeurons.append(updatedNeuron)
        # Build the new brain:
        return Brain(
            self.somaProgram,
            self.dendriteProgram,
            newNeurons,
            self.inputLocations,
            self.config
        )

    def runSoma(self, neuron: Neuron) -> tuple[float, float, float, float]:
        somaProgramInputs = neuron.programInputs()
        somaProgramOutputs = self.somaProgram.evaluate(somaProgramInputs)
        updatedNeuron = self.updateNeuron(neuron, somaProgramOutputs[0])
        return updatedNeuron

    def updateNeuron(self,
                     neuron: Neuron,
                     somaProgramOutputs: npt.NDArray[np.float64]
                     ) -> tuple[float, float, float, float]:
        parentHealth = neuron.health
        parentPositionX = neuron.position.x
        parentPositionY = neuron.position.y
        parentBias = neuron.bias
        health = somaProgramOutputs[0]
        positionX = somaProgramOutputs[1]
        positionY = somaProgramOutputs[2]
        bias = somaProgramOutputs[3]
        # Calculate our increment
        healthIncrement = 0.1
        positionIncrement = 0.1
        biasIncrement = 0.1
        # Apply the increment
        health = parentHealth + MathUtil.sign(health) * healthIncrement
        positionX = parentPositionX + (
            MathUtil.sign(positionX) * positionIncrement)
        positionY = parentPositionY + (
            MathUtil.sign(positionY) * positionIncrement)
        bias = parentBias + MathUtil.sign(bias) * biasIncrement
        health = MathUtil.clamp(health, -1.0, 1.0)
        positionX = MathUtil.clamp(positionX, -1.0, 1.0)
        positionY = MathUtil.clamp(positionY, -1.0, 1.0)
        bias = MathUtil.clamp(bias, -1.0, 1.0)
        return health, positionX, positionY, bias

    def runAllDendrites(self,
                        neuron: Neuron,
                        newSomaPosition: Point2d,
                        newSomaHealth: float,
                        newSomaBias: float) -> Neuron:
        # We're going to rearrange this slightly
        # 1: Loop through all existing dendrites, and update:
        new_dendrites = []
        base_dendrites = neuron.dendrites
        if neuron.health > self.config.dendrite_health_birth_threshold:
            base_dendrites.append(self.generateDendrite(neuron))
        for dendrite in base_dendrites:
            inputs = []
            inputs.append(neuron.health)
            inputs.append(neuron.position.x)
            inputs.append(neuron.position.y)
            inputs.append(neuron.bias)
            inputs.append(dendrite.health)
            inputs.append(dendrite.weight)
            inputs.append(dendrite.position.x)
            inputs.append(dendrite.position.y)
            dendrite_program_outputs = self.dendriteProgram.evaluate(
                np.asarray(inputs).reshape((1, -1)))
            updated_dendrite = self.runDendrite(neuron,
                                                dendrite,
                                                dendrite_program_outputs[0])
            if (updated_dendrite.health >
                    self.config.dendrite_health_death_threshold):
                new_dendrites.append(updated_dendrite)
                if len(new_dendrites) >= self.config.max_num_dendrites:
                    break
        if len(new_dendrites) == 0:
            new_dendrites.append(neuron.dendrites[0])
        return Neuron(newSomaHealth,
                      newSomaPosition,
                      newSomaBias,
                      new_dendrites,
                      neuron.out)

    def generateDendrite(self, neuron: Neuron) -> Dendrite:
        # New dendrite is given weight/health of 1.0
        # And x/y of 0.8 of parent
        initialPosition = Point2d(neuron.position.x * 0.8,
                                  neuron.position.y * 0.8)
        return Dendrite(
            1.0,
            1.0,
            initialPosition
        )

    def runDendrite(
            self,
            neuron: Neuron,
            dendrite: Dendrite,
            dendriteOutputs: npt.NDArray[np.float64]) -> Dendrite:
        parentHealth = dendrite.health
        parentPositionX = dendrite.position.x
        parentPositionY = dendrite.position.y
        parentWeight = dendrite.weight
        health = dendriteOutputs[0]
        weight = dendriteOutputs[1]
        positionX = dendriteOutputs[2]
        positionY = dendriteOutputs[3]
        # Calculate our increment
        healthIncrement = 0.1
        weightIcrement = 0.1
        positionIncrement = 0.1

        health = parentHealth + MathUtil.sign(health) * healthIncrement
        weight = parentWeight + MathUtil.sign(weight) * weightIcrement
        positionX = parentPositionX + (
            MathUtil.sign(positionX) * positionIncrement)
        positionY = parentPositionY + (
            MathUtil.sign(positionY) * positionIncrement)
        health = MathUtil.clamp(health, -1.0, 1.0)
        weight = MathUtil.clamp(weight, -1.0, 1.0)
        positionX = MathUtil.clamp(positionX, -1.0, 1.0)
        positionY = MathUtil.clamp(positionY, -1.0, 1.0)
        return Dendrite(
            health,
            weight,
            Point2d(positionX, positionY)
        )

    def createNewNeuron(self, parentNeuron: Neuron) -> Neuron:
        # New neurons are given health 1, bias 0:
        newPosition = parentNeuron.position
        newDendrites = []
        for _ in range(self.config.initial_num_dendrites):
            dendrite = Dendrite(
                1.0,
                0.0,
                Point2d(0.0, 0.0)
            )
            newDendrites.append(dendrite)
        return Neuron(
            1.0,
            newPosition,
            0.0,
            newDendrites,
            parentNeuron.out
        )

    def extractANN(self, problem, outputAddress):
        numberInputs = 3
        nonOutputNeur = []
        nonOutputNeuronAddress = []
        outputNeurons = []
        outputNeuronAddress = []
        phenotype_isOut = []
        phenotype_bias = []
        phenotype_address = []
        for i in range(len(self.neurons)):
            address = i + numberInputs
            neuron = self.neurons[i]
            if neuron.out > 0:
                outputNeurons.append(neuron)
                outputNeuronAddress.append(address)
            else:
                nonOutputNeur.append(neuron)
                nonOutputNeuronAddress.append(address)
        for i in range(len(nonOutputNeur)):
            neuron = nonOutputNeur[i]
            phenotype_isOut.append(0)
            phenotype_bias.append(neuron.bias)
            phenotype_address.append(nonOutputNeuronAddress[i])
            # neuronPosition = nonOutputNeur[i].position
            for j in range(len(neuron.dendrites)):
                dendrite = neuron.dendrites[j]
                dendPos = dendrite.position
                next

    def getClosest(self, numNonOutNeur, nonOutNeur, isOut, dendPos):
        pass
