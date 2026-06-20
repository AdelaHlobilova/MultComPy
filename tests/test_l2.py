from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MultComPy as mcp


def _bool_stripes(shape, axis):
    """Build a deterministic binary medium with 2-phase stripes."""
    coords = np.indices(shape)[axis]
    return (coords % 2) == 0


def _single_voxel(shape):
    medium = np.zeros(shape, dtype=bool)
    medium[(0,) * len(shape)] = True
    return medium


def _expected_shape(shape):
    return tuple(2 * s - 1 for s in shape)


def _expected_center(shape):
    return tuple(s - 1 for s in shape)


def _expected_constant_l2(shape, fill_value, phase):
    return np.full(_expected_shape(shape), float(fill_value == phase))


def _expected_stripes_l2(shape, axis):
    """
    Exact lineal-path profile for cyclic 2-phase stripes.

    Along the striped axis, the valid segment lengths are determined by the
    cyclic run length of the True phase. The result is then broadcast across
    the remaining axes because the medium is constant there.
    """
    expected = np.zeros(_expected_shape(shape), dtype=float)
    n = shape[axis]
    center = n - 1

    profile = np.zeros(2 * n - 1, dtype=float)
    profile[center] = np.ceil(n / 2) / n
    if n % 2 == 1 and n > 1:
        profile[center - 1] = 1.0 / n
        profile[center + 1] = 1.0 / n

    reshape = [1] * len(shape)
    reshape[axis] = 2 * n - 1
    expected = np.broadcast_to(profile.reshape(reshape), expected.shape).astype(float)
    return expected


def _expected_single_voxel_l2(shape):
    expected = np.zeros(_expected_shape(shape), dtype=float)
    expected[_expected_center(shape)] = 1.0 / np.prod(shape)
    return expected


def _l2_direct(medium, maxsize, phase=True):
    return mcp.L2_direct_computation(medium, maxsize, phase=phase, step=1, method="py")


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("fill_value", [True, False])
@pytest.mark.parametrize("phase", [True, False])
def test_l2_constant_fields_match_exact_solution(shape, fill_value, phase):
    """Uniform media have an exact closed-form L2: all ones or all zeros."""
    medium = np.full(shape, fill_value, dtype=bool)
    expected = _expected_constant_l2(shape, fill_value, phase)

    l2 = _l2_direct(medium, shape, phase=phase)
    np.testing.assert_allclose(l2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((3, 4), 1),
        ((3, 3, 4), 0),
    ],
)
def test_l2_anisotropic_stripes_have_exact_closed_form(shape, axis):
    """
    Stripe-like media admit an exact lineal-path solution: a full central slice
    survives, and every other displacement is impossible.
    """
    medium = _bool_stripes(shape, axis=axis)
    expected = _expected_stripes_l2(shape, axis=axis)

    l2 = _l2_direct(medium, shape, phase=True)
    np.testing.assert_allclose(l2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4),
        (3, 3, 4),
    ],
)
def test_l2_single_voxel_matches_exact_volume_fraction(shape):
    """A single occupied cell gives a delta at zero displacement."""
    medium = _single_voxel(shape)
    expected = _expected_single_voxel_l2(shape)

    l2 = _l2_direct(medium, shape, phase=True)
    np.testing.assert_allclose(l2, expected, atol=1e-12, rtol=1e-12)
