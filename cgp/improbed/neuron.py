from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .point2d import Point2d
from .dendrite import Dendrite


@dataclass(frozen=True)
class Neuron:
    health: float
    position: Point2d
    bias: float
    dendrites: list[Dendrite]
    # Denotes which problem this neuron is for
    # 0 if it's a non-output,
    # 1-N if it's an output neuron
    out: int

    def programInputs(self) -> npt.NDArray[np.float64]:
        inputs = []
        inputs.append(self.health)
        inputs.append(self.position.x)
        inputs.append(self.position.y)
        inputs.append(self.bias)
        avgPosition = self.getAvgDendritePosition()
        inputs.append(avgPosition.x)
        inputs.append(avgPosition.y)
        inputs.append(self.getAvgDendriteWeight())
        inputs.append(self.getAvgDendriteHealth())
        return np.asarray(inputs).reshape((1, -1))
    
    def getAvgDendritePosition(self) -> Point2d:
        totalX = 0.0
        totalY = 0.0

        for dendrite in self.dendrites:
            totalX += dendrite.position.x
            totalY += dendrite.position.y
        return Point2d(totalX / len(self.dendrites),
                       totalY / len(self.dendrites))

    def getAvgDendriteWeight(self) -> float:
        totalWeight = 0.0

        for dendrite in self.dendrites:
            totalWeight += dendrite.weight
        return totalWeight / len(self.dendrites)

    def getAvgDendriteHealth(self) -> float:
        totalHeight = 0.0

        for dendrite in self.dendrites:
            totalHeight += dendrite.health
        return totalHeight / len(self.dendrites)
