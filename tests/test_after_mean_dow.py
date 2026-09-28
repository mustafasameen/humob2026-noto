"""The weekday-adjusted after-anchor mean is `anchor` with a zero before-anchor weight."""
import numpy as np

from humob26.anchors import gap_weight


def test_after_shape_puts_no_weight_on_the_before_anchor():
    for u in (0.0, 0.3, 1.0):
        assert gap_weight("after", u) == 0.0
    u = np.linspace(0, 1, 5)
    assert np.all(gap_weight("after", u) == 0.0)
    assert gap_weight("linear", 0.0) == 1.0          # the other shapes are unchanged
