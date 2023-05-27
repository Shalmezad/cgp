from abc import ABC, abstractmethod
from .gene import Gene


class GeneMutatorBase(ABC):

    @abstractmethod
    def mutateGene(self, g: Gene) -> Gene:
        pass
