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


def _alternating_parity_profile(length):
    return np.array([1.0 if shift % 2 == 0 else 0.0 for shift in range(length)])


def _expected_parity_correlation(shape, axis):
    """Return the exact autocorrelation for a stripe medium."""
    line = _alternating_parity_profile(shape[axis])
    reshape = [1] * len(shape)
    reshape[axis] = shape[axis]
    expected = line.reshape(reshape)
    return np.broadcast_to(expected, shape).astype(float) * 0.5


def _deterministic_medium(shape):
    """Return a nontrivial boolean medium for backend-equivalence tests."""
    coords = np.indices(shape)
    weights = np.arange(1, len(shape) + 1).reshape((-1,) + (1,) * len(shape))
    pattern = np.sum(weights * coords, axis=0)
    return (pattern % 5) < 2


def _single_voxel(shape):
    medium = np.zeros(shape, dtype=bool)
    medium[(0,) * len(shape)] = True
    return medium


def _checkerboard(shape):
    """Build a 2D checkerboard medium with exactly half the pixels occupied."""
    coords = np.indices(shape)
    return (np.sum(coords, axis=0) % 2) == 0


def _cyclic_overlap_1d(length, period, shift):
    """Exact overlap of one contiguous interval on a cyclic 1D domain."""
    if shift < 0 or shift >= period:
        raise ValueError("shift must be within the cyclic domain")

    if length > period // 2:
        raise ValueError("This helper assumes the block is not larger than half the domain")

    if shift < length:
        return length - shift
    if shift >= period - length:
        return length - (period - shift)
    return 0


def _solid_block_expected(domain_shape, block_shape):
    """Exact cyclic autocorrelation for a solid rectangular block at the origin."""
    expected = np.zeros(domain_shape, dtype=float)
    area = np.prod(domain_shape)

    for i in range(domain_shape[0]):
        overlap_y = _cyclic_overlap_1d(block_shape[0], domain_shape[0], i)
        for j in range(domain_shape[1]):
            overlap_x = _cyclic_overlap_1d(block_shape[1], domain_shape[1], j)
            expected[i, j] = (overlap_y * overlap_x) / area

    return expected


def _s2_fft(arr1, arr2=None, larger=False, version=1):
    if version == 0:
        if arr2 is None:
            arr2 = arr1
        return mcp.S2_Discrete_Fourier_transform(arr1, arr2, larger=larger, version=0)

    return mcp.S2_Discrete_Fourier_transform(arr1, arr2, larger=larger, version=version)


@pytest.mark.parametrize(
    "shape, fill_value",
    [
        ((4, 6), True),
        ((4, 6), False),
        ((4, 4, 6), True),
        ((4, 4, 6), False),
    ],
)
def test_s2_constant_fields_match_exact_solution(shape, fill_value):
    """Uniform media have an exact closed-form S2: all ones or all zeros."""
    medium = np.full(shape, fill_value, dtype=bool)
    expected = np.full(shape, float(fill_value))

    direct = mcp.S2_direct_computation(medium, medium, larger=False)
    np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        np.testing.assert_allclose(fft, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((4, 6), 1),
        ((4, 4, 6), 0),
    ],
)
def test_s2_anisotropic_stripes_have_exact_closed_form(shape, axis):
    """Stripe-like media give an exact parity pattern that is easy to verify."""
    medium = _bool_stripes(shape, axis=axis)
    expected = _expected_parity_correlation(shape, axis=axis)

    direct = mcp.S2_direct_computation(medium, medium, larger=False)
    np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        np.testing.assert_allclose(fft, expected, atol=1e-12, rtol=1e-12)


def test_s2_checkerboard_matches_exact_parity_pattern():
    """A checkerboard exercises coupled shifts in both spatial directions."""
    shape = (6, 6)
    medium = _checkerboard(shape)
    expected = np.fromfunction(
        lambda i, j: np.where(((i + j) % 2) == 0, 0.5, 0.0),
        shape,
        dtype=int,
    ).astype(float)

    direct = mcp.S2_direct_computation(medium, medium, larger=False)
    np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        np.testing.assert_allclose(fft, expected, atol=1e-12, rtol=1e-12)


def test_s2_single_solid_block_matches_exact_cyclic_overlap():
    """
    A solid block has an exact piecewise-linear autocorrelation under periodic
    boundary conditions.
    """
    domain_shape = (10, 10)
    block_shape = (3, 4)
    medium = np.zeros(domain_shape, dtype=bool)
    medium[: block_shape[0], : block_shape[1]] = True
    expected = _solid_block_expected(domain_shape, block_shape)

    direct = mcp.S2_direct_computation(medium, medium, larger=False)
    np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        np.testing.assert_allclose(fft, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 6),
        (4, 4, 6),
    ],
)
@pytest.mark.parametrize("larger", [False, True])
def test_s2_direct_and_fft_match_on_nontrivial_media(shape, larger):
    """
    Backend parity test for regression coverage.

    The medium is deterministic, binary, and non-symmetric so the test exercises
    a realistic mixed phase distribution instead of a trivial uniform field.
    """
    medium = _deterministic_medium(shape)
    direct = mcp.S2_direct_computation(medium, medium, larger=larger)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=larger, version=version)
        assert fft.shape == direct.shape
        np.testing.assert_allclose(fft, direct, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 5), marks=pytest.mark.xfail(reason="Current FFT backend truncates odd-sized last dimensions.")),
        (5, 4),
        pytest.param((3, 5, 7), marks=pytest.mark.xfail(reason="Current FFT backend truncates odd-sized last dimensions.")),
    ],
)
def test_s2_fft_shape_matches_direct_for_odd_shapes(shape):
    """Document the current odd-size FFT regression without breaking the suite."""
    medium = _deterministic_medium(shape)
    direct = mcp.S2_direct_computation(medium, medium, larger=False)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        assert fft.shape == direct.shape


@pytest.mark.parametrize(
    "shape",
    [
        (4, 6),
        (4, 4, 6),
    ],
)
def test_s2_single_voxel_matches_exact_volume_fraction(shape):
    """A single occupied cell provides an exact patch test for the zero lag."""
    medium = _single_voxel(shape)
    expected = np.zeros(shape, dtype=float)
    expected[(0,) * len(shape)] = 1.0 / np.prod(shape)

    direct = mcp.S2_direct_computation(medium, medium, larger=False)
    np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        np.testing.assert_allclose(fft, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((4, 6), 1),
        ((4, 4, 6), 0),
    ],
)
def test_s2_auto_plus_complement_equals_volume_fraction(shape, axis):
    """
    Auto-correlation with a phase and cross-correlation with its complement
    must sum to the phase volume fraction for every lag.
    """
    medium = _bool_stripes(shape, axis=axis)
    complement = np.logical_not(medium)
    phi = np.mean(medium.astype(float))

    auto_direct = mcp.S2_direct_computation(medium, medium, larger=False)
    cross_direct = mcp.S2_direct_computation(medium, complement, larger=False)
    np.testing.assert_allclose(auto_direct + cross_direct, phi, atol=1e-12, rtol=1e-12)

    for version in (0, 1, 2, 3):
        auto_fft = _s2_fft(medium, medium if version == 0 else None, larger=False, version=version)
        cross_fft = _s2_fft(medium, complement, larger=False, version=version)
        np.testing.assert_allclose(auto_fft + cross_fft, phi, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [
    (6, 6),       # 2D test
    (4, 4, 6)     # 3D test
])
def test_s2_fft_matches_brute_force_on_small_matrices(shape):
    """Verify that the fastest FFT implementation (version 3) matches the brute force computation for 2D and 3D."""
    np.random.seed(42)
    small_medium = np.random.choice([True, False], size=shape, p=[0.3, 0.7])

    # Brute force
    s2_brute = mcp.S2_direct_computation(small_medium, small_medium, larger=True)

    # FFT (version 3)
    s2_fft = mcp.S2_Discrete_Fourier_transform(small_medium, small_medium, larger=True, version=3)

    # Compare results
    assert s2_brute.shape == s2_fft.shape
    np.testing.assert_allclose(s2_fft, s2_brute, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(10, 10), (6, 6, 6)])
def test_s2_peak_location_depends_on_larger_parameter(shape):
    """Check that the 'larger' parameter correctly shifts the origin (maximum value) of the S2 matrix."""
    np.random.seed(42)
    # Generate a random array to ensure we have a clear volume fraction
    medium = np.random.choice([True, False], size=shape, p=[0.4, 0.6])
    
    # 1. Test for larger=False (maximum should be at index 0,0 / 0,0,0)
    s2_small = mcp.S2_Discrete_Fourier_transform(medium, larger=False, version=3)
    peak_idx_small = np.unravel_index(np.argmax(s2_small), s2_small.shape)
    expected_small_origin = (0,) * len(shape)
    
    assert peak_idx_small == expected_small_origin
    
    # 2. Test for larger=True (maximum should be at the center)
    s2_large = mcp.S2_Discrete_Fourier_transform(medium, larger=True, version=3)
    peak_idx_large = np.unravel_index(np.argmax(s2_large), s2_large.shape)
    # Center is the original size minus 1
    expected_large_origin = tuple(s - 1 for s in shape)
    
    assert peak_idx_large == expected_large_origin
