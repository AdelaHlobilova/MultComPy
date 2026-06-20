from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MultComPy as mcp


EXAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "2DcemPaste_clinkerVol_0.05_size_100x100.0.img.png"

EXPECTED_S2_1D_VALUES = np.array(
    [
        0.0489,
        0.04125,
        0.03503333333333334,
        0.029625,
        0.02375,
        0.01845714285714286,
        0.014879999999999999,
        0.01149,
        0.008433333333333334,
        0.0056058823529411776,
        0.0038607142857142863,
        0.002922222222222223,
        0.00216764705882353,
        0.0014522727272727283,
        0.000977272727272728,
        0.0008642857142857152,
        0.0009696428571428578,
        0.0012821428571428578,
        0.0015392857142857148,
        0.0018879310344827591,
        0.0023142857142857153,
        0.0024611111111111114,
        0.002864285714285715,
        0.0030250000000000008,
        0.003145833333333334,
        0.003140476190476191,
        0.0029268292682926834,
        0.0026375,
        0.002269565217391304,
        0.0018651162790697673,
        0.0015019999999999996,
        0.0012427083333333335,
        0.0010723404255319146,
        0.0008653846153846155,
        0.000899107142857143,
        0.0008535714285714287,
        0.0010043859649122812,
        0.0012008928571428576,
        0.0014758064516129036,
        0.0016991525423728816,
        0.0018757575757575758,
        0.001941935483870968,
        0.002164393939393939,
        0.0022869565217391303,
        0.002362121212121212,
        0.0024916666666666668,
        0.0023376811594202898,
        0.002406578947368421,
        0.002274342105263158,
        0.002066025641025641,
    ],
    dtype=float,
)

EXPECTED_S2_1D_R = np.arange(50, dtype=int)

EXPECTED_C2_1D_VALUES = np.array(
    [
        0.0489,
        0.040925,
        0.03456666666666667,
        0.029224999999999998,
        0.02355625,
        0.018457142857142856,
        0.014879999999999999,
        0.011489999999999998,
        0.008433333333333333,
        0.005588235294117647,
        0.003764285714285714,
        0.0027499999999999994,
        0.0018294117647058821,
        0.001,
        0.00034545454545454555,
        9.52380952380978e-06,
        -8.907014919616726e-21,
        -1.322293455079979e-19,
        -7.893537578662928e-20,
        -4.605146262469231e-20,
        1.7585730135930555e-20,
        2.54430244022849e-19,
        3.962680087715268e-19,
        3.8565780252944304e-19,
        2.419104826543331e-19,
        1.0165364233189098e-19,
        1.6762223742803038e-19,
        1.1112090171849063e-19,
        2.8061791675151704e-20,
        -4.3520599642654417e-20,
        -9.262638018344368e-20,
        5.188473588944518e-21,
        -8.094443248291743e-20,
        -1.3332988810171832e-19,
        -1.6953888252362362e-19,
        -7.706715563756674e-21,
        1.4889818012159218e-19,
        1.0916823969680432e-19,
        4.52982692272701e-20,
        1.4203014633654248e-19,
        1.560448238519793e-20,
        -4.035023906124226e-20,
        -8.347026234004886e-20,
        2.1624594103029713e-20,
        3.697997973556631e-20,
        2.14338434639393e-20,
        -2.0260994751229403e-21,
        3.358179258507944e-20,
        5.479351636261525e-20,
        7.553619548504828e-20,
        5.0691217386088805e-20,
        1.1434608477282036e-19,
        1.0361263828910357e-19,
        1.152343735659455e-19,
        9.367791550917645e-20,
        1.0031561426393472e-19,
        1.2121680182470458e-19,
        7.447647615941816e-20,
        -3.822446915315988e-21,
        5.492097017402726e-20,
        4.881352708753237e-20,
        1.3474250165137455e-19,
        6.272466702736626e-20,
        8.408355745868012e-20,
        1.1971722453261917e-19,
        5.064307385223474e-20,
        -7.97378890473699e-20,
        -5.640255075458613e-20,
        -6.824163736980512e-20,
        3.7130901323808136e-20,
        -5.221209166892437e-20,
        -8.956239806707223e-21,
        8.069365001300978e-21,
        3.218258706324844e-20,
        6.367625521727295e-20,
        -2.813472665276588e-20,
        2.9271715759273076e-20,
        3.2174558952274385e-20,
        9.190064048659009e-20,
        2.7523367100729256e-21,
        -3.3104819053920126e-20,
        -6.423854700280657e-20,
        -6.510384961235079e-20,
        -7.917037396031833e-20,
        -6.12224103253029e-20,
        2.0699549815510844e-20,
        1.838235294117652e-05,
        6.666666666666675e-05,
        0.00014253731343283585,
        0.00022465753424657532,
        0.0003446043165467626,
        0.0004916666666666667,
        0.0007035211267605634,
        0.0009216216216216214,
        0.0012475862068965517,
        0.00155,
        0.0018763333333333332,
        0.0022091503267973856,
        0.0023409638554216863,
        0.002900337837837838,
    ],
    dtype=float,
)

EXPECTED_C2_1D_R = np.arange(100, dtype=int)


def _load_example_medium(threshold=200):
    """Load the bundled example image and binarize it exactly like the examples."""
    with Image.open(EXAMPLE_IMAGE) as img_file:
        img_file_bw = img_file.convert("L").point(lambda x: 255 if x > threshold else 0, mode="1")
        return np.array(img_file_bw)


def test_example_image_binarization_has_expected_volume_fraction():
    """The bundled image is the same one used in the example scripts."""
    medium = _load_example_medium()

    assert medium.shape == (100, 100)
    assert medium.dtype == np.bool_
    np.testing.assert_allclose(medium.mean(), 0.0489, atol=1e-12, rtol=1e-12)


def test_example_image_s2_pipeline_matches_across_all_versions():
    """Mirror the example S2 script and keep all four FFT variants in lockstep."""
    medium = _load_example_medium()
    expected_vf = medium.mean()
    reference = None

    for version in (0, 1, 2, 3):
        s2 = mcp.S2_Discrete_Fourier_transform(medium, medium if version == 0 else None, version=version)
        assert s2.shape == (199, 199)
        np.testing.assert_allclose(s2[s2.shape[0] // 2, s2.shape[1] // 2], expected_vf, atol=1e-12, rtol=1e-12)

        transformed = mcp.transform_ND_to_1D(s2, rmax=medium.shape[0] // 2)
        assert len(transformed) == 2
        assert len(transformed[0]) == 50
        assert len(transformed[1]) == 50

        if reference is None:
            reference = transformed
        else:
            np.testing.assert_allclose(transformed[0], reference[0], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(transformed[1], reference[1], atol=1e-12, rtol=1e-12)


def test_example_image_s2_matches_stored_reference_curve():
    """The bundled example should keep matching the checked-in 1D S2 profile."""
    medium = _load_example_medium()
    s2 = mcp.S2_Discrete_Fourier_transform(medium, medium)
    transformed = mcp.transform_ND_to_1D(s2, rmax=medium.shape[0] // 2)

    np.testing.assert_allclose(transformed[0], EXPECTED_S2_1D_VALUES, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(transformed[1], EXPECTED_S2_1D_R)


def test_example_image_c2_pipeline_matches_across_all_versions():
    """Mirror the example C2 script and keep all four FFT variants in lockstep."""
    medium = _load_example_medium()
    expected_vf = medium.mean()
    reference = None

    for version in (0, 1, 2, 3):
        c2 = mcp.C2_Discrete_Fourier_transform(medium, version=version)
        assert c2.shape == (199, 199)
        np.testing.assert_allclose(c2[c2.shape[0] // 2, c2.shape[1] // 2], expected_vf, atol=1e-12, rtol=1e-12)

        transformed = mcp.transform_ND_to_1D(c2, scale=False)
        assert len(transformed) == 2
        assert len(transformed[0]) == 100
        assert len(transformed[1]) == 100

        if reference is None:
            reference = transformed
        else:
            np.testing.assert_allclose(transformed[0], reference[0], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(transformed[1], reference[1], atol=1e-12, rtol=1e-12)


def test_example_image_c2_matches_stored_reference_curve():
    """The bundled example should keep matching the checked-in 1D C2 profile."""
    medium = _load_example_medium()
    c2 = mcp.C2_Discrete_Fourier_transform(medium)
    transformed = mcp.transform_ND_to_1D(c2, scale=False)

    np.testing.assert_allclose(transformed[0], EXPECTED_C2_1D_VALUES, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(transformed[1], EXPECTED_C2_1D_R)
