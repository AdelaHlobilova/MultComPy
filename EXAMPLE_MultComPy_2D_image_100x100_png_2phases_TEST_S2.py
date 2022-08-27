# -*- coding: utf-8 -*-

"""
Test script for image processing and development of image processing library.

@authors:   Adela Hlobilova, adela.hlobilova@gmail.com
            Michal Hlobil, michalhlobil@seznam.cz
"""

import MultComPy as mcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time
import scipy.ndimage as spim
from PIL import Image, ImageFilter 
import matplotlib.pyplot as plt
import sys


img_name = "2DcemPaste_clinkerVol_0.05_size_100x100.0.img.png"

# =============================================================================
# image initialization, convert image file to BW numpy array
# =============================================================================

img_file = Image.open(img_name)
col, row = img_file.size


thresh = 200
def fn(x): return 255 if x > thresh else 0

img_file_BW = img_file.convert('L').point(fn, mode='1')
img_file_BW.save('test.png')

img_array = np.array(img_file_BW)
phase = True

S1 = np.sum(img_array)/np.prod(np.array(img_array.shape))
print("Volume fraction of phase {} is {:.4f}.".format(phase, S1))


# =============================================================================
# Graptical representation of the results
# =============================================================================


def image_init():
    """
    Initialize the image with the identical settings for all images.

    Returns
    -------
    ax : AxesSubplot object

    """
    fig, ax = plt.subplots()
    plt.axhline(y=S1, color='b', linestyle='--')
    plt.axhline(y=S1**2, color='b', linestyle='--')

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.grid(which='major', color='red', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(which='minor', color='black', linestyle=':', linewidth=0.5)
    ax.yaxis.grid(which='major', color='red', linestyle='--', linewidth=0.5)
    ax.yaxis.grid(which='minor', color='black', linestyle=':', linewidth=0.5)
    plt.xlabel('r')
    plt.ylabel('Probability')

    return ax


def image_final():
    """
    Show the image together with the legend.

    Returns
    -------
    None.

    """
    ax.legend(loc='upper center')
    plt.show()


# =============================================================================
# Auto-correlation function (Two-point probability function, S2)
# =============================================================================

print("Computing S2 from MultComPy.py v1 (np fftn)....")
tic = time.time()
S2 = mcp.S2_Discrete_Fourier_transform(img_array, img_array)
## or alternatively by direct computation; this output will be equal to the output
## via Discrete Fourier transform, but it will be much slower: 
# S2_alt = mcp.S2_direct_computation(img_array, img_array)

S2_tr = mcp.transform_ND_to_1D(S2, rmax=int(len(img_array)/2))
toc = time.time() - tic
print("**S2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

print("Computing S2 from MultComPy.py v2 (np fftn switch)....")
tic = time.time()
S2 = mcp.S2_Discrete_Fourier_transform(img_array, version=1)

S2_tr2 = mcp.transform_ND_to_1D(S2, rmax=int(len(img_array)/2))
toc = time.time() - tic
print("**S2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

print("Computing S2 from MultComPy.py v2 (scipy fftn switch)....")
tic = time.time()
S2 = mcp.S2_Discrete_Fourier_transform(img_array, version=2)

S2_tr3 = mcp.transform_ND_to_1D(S2, rmax=int(len(img_array)/2))
toc = time.time() - tic
print("**S2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

print("Computing S2 from MultComPy.py v3 (scipy fftn switch, shorter code)....")
tic = time.time()
S2 = mcp.S2_Discrete_Fourier_transform(img_array, version=3)

S2_tr4 = mcp.transform_ND_to_1D(S2, rmax=int(len(img_array)/2))
toc = time.time() - tic
print("**S2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

#  graphical representation of the results
ax = image_init()
ax.plot(S2_tr[1], S2_tr[0], '-o', color='y',
        markersize=5, label="S2 MultComPy, original variant numpy.fftn", linewidth=1)
ax.plot(S2_tr2[1], S2_tr2[0], '-o', color='r',
        markersize=5, label="S2 MultComPy, switch for autocorr.fun, numpy.fftn", linewidth=1)
ax.plot(S2_tr3[1], S2_tr3[0], '-o', color='g',
        markersize=5, label="S2 MultComPy, scipy.fftn", linewidth=1)
ax.plot(S2_tr4[1], S2_tr4[0], '-o', color='b',
        markersize=5, label="S2 MultComPy, scipy.fftn shorter code", linewidth=1)

image_final()


print("**imageProcess_new.py done!")
