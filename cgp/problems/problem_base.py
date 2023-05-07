from abc import ABC, abstractmethod

class ProblemBase(ABC):

    @abstractmethod
    def numInputs(self):
        pass

    @abstractmethod
    def numOutputs(self):
        pass

    @abstractmethod
    def trainingSet(self):
        """
        Returns a pair of input / expected_output for training
        """
        pass

    @abstractmethod
    def validationSet(self):
        """
        Returns a pair of input / expected_output for validation
        """
        pass

    @abstractmethod
    def measureFitness(self, expected_output, actual_output):
        """
        Returns a fitness measure
        """
        pass
