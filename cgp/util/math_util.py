class MathUtil:
    def clamp(val, min_val, max_val):
        return max(min(val, max_val), min_val)
    
    def sign(val):
        if val > 0.0:
            return 1.0
        elif val < 0.0:
            return -1.0
        else:
            return 0.0
