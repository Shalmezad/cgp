from dataclasses import dataclass

import numpy as np


@dataclass
class ANN:
    connectionAddresses: list[list[int]]
    weights: list[list[float]]
    isOut: list[int]
    bias: list[float]
    address: list[int]
    numConnectionAddress: list[int]
    outputAddresses: list[int]
    inputIdxes: list[int]

    def forward(self, input):
        result = np.asarray([
            self.evaluateLayer(x, input) for x in self.outputAddresses])
        result = np.swapaxes(result, 0, 1)
        return result

    def evaluateLayer(self, address, input):
        if address < len(self.inputIdxes):
            inputIdx = self.inputIdxes[address]
            if inputIdx == -1:
                return np.zeros_like(input[:, 0])
            else:
                return input[:, inputIdx]
        # Ok, it's a node
        # *in theory*, this should just be:
        idx = self.address.index(address)
        connections = self.connectionAddresses[idx]
        if address in connections:
            print("Address: {}".format(address))
            print("Connections: {}".format(connections))
            raise ValueError("Address in self.connections")
        weights = np.asarray(self.weights[idx]).reshape((1, -1))
        values = np.asarray([self.evaluateLayer(x, input) for x in connections])
        base_values_no_sum = values * weights.T
        base_value = np.sum(base_values_no_sum, axis=0)
        result = base_value + self.bias[idx]
        result = np.tanh(result)
        return result
