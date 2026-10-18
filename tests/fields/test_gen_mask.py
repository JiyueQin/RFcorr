"""Tests for rc.gen_mask. Run with: python -m pytest tests/fields/test_gen_mask.py"""
import numpy as np
import pytest

import rfcorr as rc


def test_gen_mask_ellipse():
    mask = rc.gen_mask((40, 60), axes=(1, 0.5))
    assert mask.shape == (40, 60) and mask.dtype == bool
    # centered and symmetric
    np.testing.assert_array_equal(mask, mask[::-1, ::-1])
    # area close to pi * a * b with semi-axes 20 and 15
    assert mask.sum() == pytest.approx(np.pi * 20 * 15, rel=0.05)
    # inscribed ellipse touches the middle of each side
    full = rc.gen_mask((41, 41))
    assert full[0, 20] and full[20, 0] and not full[0, 0]
    assert rc.gen_mask((9, 9, 9), axes=0.5).sum() < rc.gen_mask((9, 9, 9)).sum()
    with pytest.raises(ValueError):
        rc.gen_mask((10, 10), axes=(1, 1, 1))
    with pytest.raises(ValueError):
        rc.gen_mask((10, 10), shape="square")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
