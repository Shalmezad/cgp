import math

from cgp.util import MathUtil
from cgp.gene import Gene
from .neuron import Neuron


class Brain:
    # The program for updating soma
    somaProgram: Gene
    # The program for updating dendrites
    dendriteProgram: Gene

    neurons: list[Neuron]

    def update(self):
        # newNeurons = []
        nonOutputNeurons = []
        outputNeurons = []
        for neuron in self.neurons:
            if neuron.out == 0:
                nonOutputNeurons.append(neuron)
            else:
                outputNeurons.append(neuron)
        # Process non-output neurons:
        for neuron in nonOutputNeurons:
            updatedNeuronVars = self.runSoma(neuron)
            updatedNeuron = self.runAllDendrites(neuron, updatedNeuronVars)

    def runSoma(self, neuron):
        somaProgramInputs = neuron.programInputs()
        somaProgramOutputs = self.somaProgram.evaluate(somaProgramInputs)
        updatedNeuron = self.updateNeuron(neuron, somaProgramOutputs)
        return updatedNeuron
    
    def updateNeuron(self, neuron, somaProgramOutputs):
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
        health = parentHealth + math.sign(health) * healthIncrement
        positionX = parentPositionX + math.sign(positionX) * positionIncrement
        positionY = parentPositionY + math.sign(positionY) * positionIncrement
        bias = parentBias + math.sign(bias) * biasIncrement
        health = MathUtil.clamp(health, -1.0, 1.0)
        positionX = MathUtil.clamp(positionX, -1.0, 1.0)
        positionY = MathUtil.clamp(positionY, -1.0, 1.0)
        bias = MathUtil.clamp(bias, -1.0, 1.0)
        return health, positionX, positionY, bias