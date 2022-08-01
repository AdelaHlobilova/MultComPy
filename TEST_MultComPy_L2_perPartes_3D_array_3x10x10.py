# -*- coding: utf-8 -*-

"""
Test script for image processing and development of image processing library.

@author: Adela Hlobilova, ITAM of the CAS, adela.hlobilova@gmail.com
"""

#import deskriptory as dp
import MultComPy as mcp

import random
from PIL import Image, ImageFilter 
import numpy as np
import numpy.matlib as npm
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from collections import Counter
import time
import GooseEYE
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, platform
import ctypes, ctypes.util
import copy
from matplotlib.image import NonUniformImage

import porespy as ps
ps.visualization.set_mpl_style()


img_array = np.array([[[0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,1,0],
                       [0,0,0,0,0,0,0,0,1,0],
                       [0,0,0,0,1,1,1,1,1,0],
                       [0,0,0,0,0,0,0,0,0,0]],
                      [[0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,1,1,1,1,0,1,1,1,1],
                       [0,1,1,1,1,0,1,1,1,1],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,1,1,0],
                       [0,0,0,0,0,0,0,1,1,0],
                       [0,0,0,0,1,1,1,1,1,0],
                       [0,0,0,0,1,1,1,1,1,0]],
                      [[0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,0,1,1,0,0,0,1,1,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,1,1],
                       [0,0,0,0,0,0,0,0,1,1],
                       [0,0,0,0,0,0,0,0,1,1],
                       [0,0,0,0,1,1,1,1,1,1]]])


# =============================================================================
# =============================================================================
# # # # # # # # # # # # # # # # # # T E S T S # # # # # # # # # # # # # # # # #
# =============================================================================
# =============================================================================

S1 = np.sum(img_array)/np.prod(np.array(img_array.shape))
print("Volume fraction is {}".format(S1))


# =============================================================================
# Graptical representation of the results
# =============================================================================


def image_init(label):
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
    ax.set_title(label)

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
# Lineal path function (L2)
# =============================================================================
print("Computing L2 from deskriptory.py (Python) ....")
tic = time.time()
newdep = int(img_array.shape[0]/2)+1
newrow = int(img_array.shape[1]/2)+1
newcol = int(img_array.shape[2]/2)+1
# L2 = mcp.L2_direct_computation(img_array, img_array.shape) # evaluates whole lineal path
L2 = mcp.L2_direct_computation(img_array, (newdep, newrow, newcol)) # evaluates half of the lineal path
L2_tr2 = mcp.transform_ND_to_1D(L2)
toc = time.time()-tic
print("**L2 evaluation from deskriptory.py done! eval. time = {}".format(toc))

print("Computing L2 from deskriptory.py (C) ....")
tic = time.time()
# newrow = int(img_array.shape[0]/2)+1
# newcol = int(img_array.shape[1]/2)+1
# L22 = mcp.L2_direct_computation(img_array, img_array.shape, method="dll") # evaluates whole lineal path
L22 = mcp.L2_direct_computation(img_array, (newdep, newrow, newcol), method="dll") # evaluates half of the lineal path
L22_tr2 = mcp.transform_ND_to_1D(L22)
toc = time.time()-tic
print("**L2 evaluation from deskriptory.py done! eval. time = {}".format(toc))

print("Computing L2 from GooseEYE ....")
tic = time.time()
L2_GE = GooseEYE.L(img_array.shape, img_array) # evaluates half of the lineal path
# newshape = tuple((np.array(img_array.shape)*2-1).astype(int))
# L2_GE = GooseEYE.L(newshape, img_array) # evaluates whole lineal path
toc = time.time()-tic
L2GE_tr = mcp.transform_ND_to_1D(L2_GE)
print("**L2 evaluation from GooseEYE done! eval. time = {}".format(toc))

print("Computing L2 from MultComPy.py (dll, per partes) ....")
tic = time.time()

## Define as many mcp.L2_direct_computation_dll as necessary for your media 
#  (with different start_dep and stop_dep). Each 3D image has three dimensions:
#  number of depths, rows, and columns. The parallelization is made through
#  depths, therefore, the first L2_direct_computation_dll run starts with 
#  start_dep equal to 0 and the last L2_direct_computation_dll ends with stop_dep
#  equal to number of depths. The difference between stop_dep and start_dep in
#  one function is not necessary equal to one; it can be equal to as many depths
#  that are possible to evaluate within the given computer time (you have to test it
#  on your own).

## This version evaluates the whole lineal path
L222_1 = mcp.L2_direct_computation_dll(img_array, *img_array.shape, phase=True, step=1, 
                              progress_flag=1, start_dep=0, stop_dep=1)
L222_2 = mcp.L2_direct_computation_dll(img_array, *img_array.shape, phase=True, step=1, 
                              progress_flag=1, start_dep=1, stop_dep=2)
L222_3 = mcp.L2_direct_computation_dll(img_array, *img_array.shape, phase=True, step=1, 
                              progress_flag=1, start_dep=2, stop_dep=3)
L222 = mcp.collect_partial_frequency_matrices_and_transform_to_L2(*img_array.shape, [0,1,2], [1,2,3])

L222_tr = mcp.transform_ND_to_1D(L222)
toc = time.time()-tic
print("**L2 evaluation from MultComPy.py (dll, per partes) done! eval. time = {}".format(toc))



# graphical representation of the results
ax = image_init("Lineal path function")

ax.plot(L2GE_tr[1], L2GE_tr[0], '-', color='r',
        label="L2 GooseEYE", linewidth=5)

# ax.plot(L2_tr[1], L2_tr[0], '-*', color='y', markersize=5,
#         label="L2 MultComPy (cut)", linewidth=1)

ax.plot(L2_tr2[1], L2_tr2[0], '-s', color='y', markersize=5,
        label="L2 MultComPy (dll)", linewidth=1)

ax.plot(L222_tr[1], L222_tr[0], '-s', color='b', markersize=5,
        label="L2 MultComPy (dll, per partes)", linewidth=1)


image_final()



print("**imageProcess_new.py done!")