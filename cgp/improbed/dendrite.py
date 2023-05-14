from dataclasses import dataclass

from .point2d import Point2d


@dataclass(frozen=True)
class Dendrite:
    health: float
    weight: float
    position: Point2d
