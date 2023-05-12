from dataclasses import dataclass


@dataclass
class Point2d(frozen=True):
    x: float
    y: float
