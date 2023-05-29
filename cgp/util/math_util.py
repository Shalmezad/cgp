import math


class MathUtil:
    @staticmethod
    def clamp(val: float, min_val: float, max_val: float) -> float:
        return max(min(val, max_val), min_val)

    @staticmethod
    def sign(val: float) -> float:
        if val > 0.0:
            return 1.0
        elif val < 0.0:
            return -1.0
        else:
            return 0.0

    @staticmethod
    def sig(val: float) -> float:
        return 1.0 / (1.0 + math.exp(-val))
