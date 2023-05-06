import math
import numpy as np

class OpSets:
    # IMPROBED: Multiple Problem-Solving Brian via Evolved Developmental Programs
    # Julian Francis Miller 
    # Artificial Life 27: 300–335 (2022) https://doi.org/10.1162/artl_a_00346
    IMPROBED_2022 = [
        (   # 0 abs
            lambda in1, in2, in3: "|{}|".format(in1),
            lambda in1, in2, in3: np.absolute(in1)
        ),
        (   # 1 sqrt
            lambda in1, in2, in3: "sqrt(|{}|)".format(in1),
            lambda in1, in2, in3: np.sqrt(np.absolute(in1))
        ),
        (   # 2 sqr
            lambda in1, in2, in3: "{}^2".format(in1),
            lambda in1, in2, in3: np.power(in1, 2)
        ),
        (   # 3 cube
            lambda in1, in2, in3: "{}^3".format(in1),
            lambda in1, in2, in3: np.power(in1, 3)
        ),
        (   # 4 exp
            lambda in1, in2, in3: "exp({})".format(in1),
            lambda in1, in2, in3: (2 * np.exp(in1 + 1) - math.pow(math.e,2) - 1) / (math.pow(math.e,2) - 1)
        ),
        (   # 5 sin
            lambda in1, in2, in3: "sin({})".format(in1),
            lambda in1, in2, in3: np.sin(in1)
        ),
        (   # 6 cos
            lambda in1, in2, in3: "cos({})".format(in1),
            lambda in1, in2, in3: np.cos(in1)
        ),
        (   # 7 tanh
            lambda in1, in2, in3: "tanh({})".format(in1),
            lambda in1, in2, in3: np.tanh(in1)
        ),
        (   # 8 inv
            lambda in1, in2, in3: "inv({})".format(in1),
            lambda in1, in2, in3: in1 * -1
        ),
        (   # 9 step
            lambda in1, in2, in3: "step({})".format(in1),
            lambda in1, in2, in3: np.where(in1 < 0.0, 0.0, 1.0)
        ),
        (   # 10 hyp
            lambda in1, in2, in3: "hyp({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.sqrt((np.power(in1, 2) + np.power(in2,2))/2.0)
        ),
        (   # 11 add
            lambda in1, in2, in3: "add({}, {})".format(in1, in2),
            lambda in1, in2, in3: (in1 + in2) / 2.0
        ),
        (   # 12 sub
            lambda in1, in2, in3: "sub({}, {})".format(in1, in2),
            lambda in1, in2, in3: (in1 - in2) / 2.0
        ),
        (   # 13 mult
            lambda in1, in2, in3: "mult({}, {})".format(in1, in2),
            lambda in1, in2, in3: in1 * in2
        ),
        (   # 14 max
            lambda in1, in2, in3: "max({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.maximum(in1, in2)
        ),
        (   # 15 min
            lambda in1, in2, in3: "min({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.minimum(in1, in2)
        ),
        (   # 16 and
            lambda in1, in2, in3: "and({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.where((in1 > 0.0) & (in2 > 0.0), 1.0, -1.0)
        ),
    ]