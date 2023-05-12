from .point2d import Point2d
from .dendrite import Dendrite

import numpy as np
import numpy.typing as npt


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
        inputs.append(self.getAvgDendritePosition())
        inputs.append(self.getAvgDendriteWeight())
        inputs.append(self.getAvgDendriteHealth())
        return np.asarray(inputs)
