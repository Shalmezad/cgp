from cgp.gene import Gene
from .point2d import Point2d

class Brain:
    # The program for updating soma
    somaProgram: Gene    
    # The program for updating dendrites
    dendriteProgram: Gene
    # Input locations:
    # NOTE: This is going to be a FLAT array
    # It is up to the caller to determine which indexes correspond to which programs
    inputLocations: list[Point2d]
    # Output locations:
    outputLocations: list[Point2d]