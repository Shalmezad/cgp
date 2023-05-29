from cgp.util import MathUtil
import unittest


class TestMathUtil(unittest.TestCase):

    def test_clamp(self) -> None:
        # Test clamping within range (should not clamp)
        inRange = MathUtil.clamp(5, 0, 10)
        self.assertEqual(inRange, 5)
        # Test over range:
        overRange = MathUtil.clamp(11, 0, 10)
        self.assertEqual(overRange, 10)
        # Test under range:
        underRange = MathUtil.clamp(-1, 0, 10)
        self.assertEqual(underRange, 0)

    def test_sign(self) -> None:
        # Test positive:
        self.assertEqual(MathUtil.sign(10), 1)
        # Test Negative:
        self.assertEqual(MathUtil.sign(-10), -1)
        # Test Zero:
        self.assertEqual(MathUtil.sign(0), 0)


if __name__ == '__main__':
    unittest.main()
