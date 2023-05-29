from enum import Enum
import math
from typing import Callable

import numpy as np
import numpy.typing as npt


class OpsetKey(Enum):
    IMPROBED_2022_OPSET_KEY = 'IMPROBED_2022_OPSET_KEY'
    GPTP_II_OPSET_KEY = 'GPTP_II_OPSET_KEY'


class OpSets:
    OpName = Callable[
        [
            str,
            str,
            str
        ],
        str]
    OpFunction = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]
        ],
        npt.NDArray[np.float64]]
    NamedOp = tuple[OpName, OpFunction]

    _SAFE_DIVISION: NamedOp = (   # /
        lambda in1, in2, in3: "/({}, {})".format(in1, in2),
        lambda in1, in2, in3: np.divide(
            in1, in2, out=in1, where=in2 != 0)
    )
    _SAFE_LOG: NamedOp = (   # log
        lambda in1, in2, in3: "log({})".format(in1),
        lambda in1, in2, in3: np.log(  # type: ignore
            np.absolute(in1), out=np.zeros_like(in1), where=in1 != 0)
    )


    # IMPROBED: Multiple Problem-Solving Brian
    #   via Evolved Developmental Programs
    # Julian Francis Miller
    # Artificial Life 27: 300–335 (2022) https://doi.org/10.1162/artl_a_00346
    _IMPROBED_2022: list[NamedOp] = [
        (   # 0 abs
            lambda in1, in2, in3: "|{}|".format(in1),
            lambda in1, in2, in3: np.absolute(in1)
        ),
        (   # 1 sqrt
            lambda in1, in2, in3: "sqrt(|{}|)".format(in1),
            lambda in1, in2, in3: np.sqrt(  # type: ignore
                np.absolute(in1))
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
            lambda in1, in2, in3: (
                (  # type: ignore
                    2 * np.exp(in1 + 1) - math.pow(math.e, 2) - 1) /
                (math.pow(math.e, 2) - 1))
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
            lambda in1, in2, in3: np.sqrt(  # type: ignore
                (np.power(in1, 2) + np.power(in2, 2))/2.0)
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
            lambda in1, in2, in3: np.where(
                (in1 > 0.0) & (in2 > 0.0), 1.0, -1.0)
        ),
        (   # 17 or
            lambda in1, in2, in3: "or({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.where(
                (in1 > 0.0) | (in2 > 0.0), 1.0, -1.0)
        ),
        (   # 18 rmux
            lambda in1, in2, in3: "rmux({}, {}, {})".format(in1, in2, in3),
            lambda in1, in2, in3: np.where(in3 > 0.0, in1, in2)
        ),
        (   # 19 imult
            lambda in1, in2, in3: "imult({}, {})".format(in1, in2),
            lambda in1, in2, in3: in1 * in2 * -1
        ),
        (   # 20 xor
            lambda in1, in2, in3: "xor({}, {})".format(in1, in2),
            lambda in1, in2, in3: np.where(
                ((in1 > 0.0) & (in2 > 0.0)) | ((in1 < 0.0) & (in2 < 0.0)),
                -1.0, 1.0)
        ),
        (   # 21 istep
            lambda in1, in2, in3: "istep({})".format(in1),
            lambda in1, in2, in3: np.where(in1 < 1.0, 0.0, -1.0)
        ),
    ]

    # GPTP II
    # Cartesian Genetic Programming and the Post Docking Filtering Problem
    # A. Beatriz Garmendia-Doval, Julian F. Miller, S. David Morley
    _GPTP_II: list[NamedOp] = [
        (   # +
            lambda in1, in2, in3: "+({}, {})".format(in1, in2),
            lambda in1, in2, in3: in1 + in2
        ),
        (   # -
            lambda in1, in2, in3: "-({}, {})".format(in1, in2),
            lambda in1, in2, in3: in1 - in2
        ),
        (   # *
            lambda in1, in2, in3: "*({}, {})".format(in1, in2),
            lambda in1, in2, in3: in1 * in2
        ),
        _SAFE_DIVISION,
        _SAFE_LOG,
        (   # exp
            lambda in1, in2, in3: "exp({})".format(in1),
            lambda in1, in2, in3: (
                np.where(in1 > 200, math.exp(200), 0.0) +  # type: ignore
                np.exp(
                    in1,
                    out=np.zeros_like(in1),
                    where=(in1 <= 200) & (in1 > -200))
            )
        ),
        (   # if
            lambda in1, in2, in3: "if({}, {}, {})".format(in1, in2, in3),
            lambda in1, in2, in3: np.where(in1 > 0, in2, in3)
        ),
    ]

    OPSET_DICT = {
        OpsetKey.IMPROBED_2022_OPSET_KEY: _IMPROBED_2022,
        OpsetKey.GPTP_II_OPSET_KEY: _GPTP_II
    }
