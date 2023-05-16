from dataclasses import dataclass


@dataclass
class ANN:
    connectionAddresses: list[list[int]]
    weights: list[float]
    isOut: list[int]
    bias: list[float]
    address: list[int]
    numConnectionAddress: list[int]
