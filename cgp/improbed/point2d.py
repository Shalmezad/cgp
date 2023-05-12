from dataclasses import dataclass


@dataclass(frozen=True)
class Point2d:
    x: float
    y: float
