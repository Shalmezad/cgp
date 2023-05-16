from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Point2d:
    x: float
    y: float

    def distanceTo(self, p: Point2d) -> float:
        return math.sqrt(self.squaredDistanceTo(p))

    def squaredDistanceTo(self, p: Point2d) -> float:
        return math.pow(self.x - p.x, 2) + math.pow(self.y - p.y, 2)
