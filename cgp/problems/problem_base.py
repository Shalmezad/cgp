from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ProblemBase(ABC):

    @abstractmethod
    def numInputs(self) -> int:
        pass

    @abstractmethod
    def numOutputs(self) -> int:
        pass

    @abstractmethod
    def trainingSet(self) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:
        """
        Returns a pair of input / expected_output for training
        """
        pass

    @abstractmethod
    def validationSet(self) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:
        """
        Returns a pair of input / expected_output for validation
        """
        pass

    @abstractmethod
    def measureFitness(self,
                       expected_output: npt.NDArray[np.float64],
                       actual_output: npt.NDArray[np.float64]
                       ) -> float:
        """
        Returns a fitness measure
        """
        pass
