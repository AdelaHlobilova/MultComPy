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


def _checkerboard(shape):
    """Build a 2D checkerboard medium with exactly half the pixels occupied."""
    coords = np.indices(shape)
    return (np.sum(coords, axis=0) % 2) == 0


def _single_voxel(shape):
    medium = np.zeros(shape, dtype=bool)
    medium[(0,) * len(shape)] = True
    return medium


def _expected_shape(shape):
    return tuple(2 * s - 1 for s in shape)


def _expected_center(shape):
    return tuple(s - 1 for s in shape)


def _expected_constant_c2(shape, fill_value):
    return np.full(_expected_shape(shape), float(fill_value))


def _expected_stripes_c2(shape, axis):
    expected = np.zeros(_expected_shape(shape), dtype=float)
    slc = [slice(None)] * len(shape)
    slc[axis] = shape[axis] - 1
    expected[tuple(slc)] = 0.5
    return expected


def _expected_checkerboard_c2(shape):
    expected = np.zeros(_expected_shape(shape), dtype=float)
    expected[_expected_center(shape)] = 0.5
    return expected


def _expected_single_voxel_c2(shape):
    expected = np.zeros(_expected_shape(shape), dtype=float)
    expected[_expected_center(shape)] = 1.0 / np.prod(shape)
    return expected


def _c2_fft(medium, version):
    return mcp.C2_Discrete_Fourier_transform(medium, larger=True, version=version)


def test_c2_single_cell_domain_matches_exact_solution():
    """A 1x1 domain is the smallest exact C2 sanity check."""
    shape = (1, 1)

    for fill_value in (True, False):
        medium = np.full(shape, fill_value, dtype=bool)
        expected = np.full(shape, float(fill_value))

        for version in (0, 1, 2, 3):
            c2 = _c2_fft(medium, version=version)
            np.testing.assert_allclose(c2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail(
    reason="Current C2 implementation cannot yet assemble the full 2*n-1 descriptor for nontrivial shapes.",
    raises=ValueError,
)
@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((4, 6), True),
        ((4, 6), False),
        ((4, 4, 6), True),
        ((4, 4, 6), False),
    ],
)
def test_c2_constant_fields_match_exact_solution(shape, fill_value):
    """Uniform media have an exact closed-form C2: all ones or all zeros."""
    medium = np.full(shape, fill_value, dtype=bool)
    expected = _expected_constant_c2(shape, fill_value)

    for version in (0, 1, 2, 3):
        c2 = _c2_fft(medium, version=version)
        np.testing.assert_allclose(c2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail(
    reason="Current C2 implementation cannot yet assemble the full 2*n-1 descriptor for nontrivial shapes.",
    raises=ValueError,
)
@pytest.mark.parametrize(
    "shape, axis",
    [
        ((4, 6), 1),
        ((4, 4, 6), 0),
    ],
)
def test_c2_anisotropic_stripes_have_exact_closed_form(shape, axis):
    """Stripe-like media give an exact hyperplane pattern for the same-cluster probability."""
    medium = _bool_stripes(shape, axis=axis)
    expected = _expected_stripes_c2(shape, axis=axis)

    for version in (0, 1, 2, 3):
        c2 = _c2_fft(medium, version=version)
        np.testing.assert_allclose(c2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail(
    reason="Current C2 implementation cannot yet assemble the full 2*n-1 descriptor for nontrivial shapes.",
    raises=ValueError,
)
def test_c2_checkerboard_isolated_clusters_have_exact_delta():
    """A checkerboard has isolated clusters, so only the zero lag survives."""
    shape = (6, 6)
    medium = _checkerboard(shape)
    expected = _expected_checkerboard_c2(shape)

    for version in (0, 1, 2, 3):
        c2 = _c2_fft(medium, version=version)
        np.testing.assert_allclose(c2, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail(
    reason="Current C2 implementation cannot yet assemble the full 2*n-1 descriptor for nontrivial shapes.",
    raises=ValueError,
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 6),
        (4, 4, 6),
    ],
)
def test_c2_single_voxel_matches_exact_volume_fraction(shape):
    """A single occupied cell provides an exact patch test for the zero lag."""
    medium = _single_voxel(shape)
    expected = _expected_single_voxel_c2(shape)

    for version in (0, 1, 2, 3):
        c2 = _c2_fft(medium, version=version)
        np.testing.assert_allclose(c2, expected, atol=1e-12, rtol=1e-12)
