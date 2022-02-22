# -*- coding: utf-8 -*-

"""
Test script for image processing and development of image processing library.

@authors:   Adela Hlobilova, ITAM of the CAS, adela.hlobilova@gmail.com
            Michal Hlobil, ITAM of the CAS, michalhlobil@seznam.cz
"""

import MultComPy as mcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time
import scipy.ndimage as spim
from PIL import Image, ImageFilter 
import matplotlib.pyplot as plt


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

print("Computing S2 from MultComPy.py ....")
tic = time.time()
S2 = mcp.S2_Discrete_Fourier_transform(img_array, img_array)
S2_tr = mcp.transform_ND_to_1D(S2, rmax=int(len(img_array)/2))
toc = time.time() - tic
print("**S2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))


#  graphical representation of the results
ax = image_init()
ax.plot(S2_tr[1], S2_tr[0], '-o', color='y',
        markersize=5, label="S2 MultComPy", linewidth=1)

image_final()


# =============================================================================
# Surface correlation functions
# =============================================================================

print("Computing surface corr. function from MultComPy.py ....")
tic = time.time()
edges, erode = mcp.find_edges(img_array)

# surface - void correlation function
print("... surface - void correlation function ...")
S2_sv = mcp.S2_Discrete_Fourier_transform(edges, ~img_array)
S2_sv_tr = mcp.transform_ND_to_1D(S2_sv)

# surface - surface correlation function
print("... surface - surface correlation function ...")
S2_ss = mcp.S2_Discrete_Fourier_transform(edges, edges)
S2_ss_tr = mcp.transform_ND_to_1D(S2_ss)
toc = time.time()-tic
print("**SCF evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

# graphical representation of the results
ax = image_init()

ax.plot(S2_sv_tr[1], S2_sv_tr[0], '-o', color='y', markersize=5,
        label="SCF surf-void MultComPy", linewidth=1)

ax.plot(S2_ss_tr[1], S2_ss_tr[0], '-o', color='m', markersize=5,
        label="SCF_surf-surf MultComPy", linewidth=1)

image_final()


# =============================================================================
# Two-point cluster function (C2)
# =============================================================================

print("Computing C2 from MultComPy.py ....")
tic = time.time()
C2 = mcp.C2_Discrete_Fourier_transform(img_array)
C2_tr = mcp.transform_ND_to_1D(C2, scale=False)
toc = time.time() - tic
print("**C2 evaluation from MultComPy done! Elapsed time is {:.4f} seconds."
      .format(toc))

# graphical representation of the results
ax = image_init()
ax.plot(C2_tr[1], C2_tr[0], '-o', color='y', markersize=5,
        label="C2 MultComPy", linewidth=1)

image_final()


# =============================================================================
# real surface area
# =============================================================================

tic = time.time()
ssa_sa = mcp.real_surface_stereological_approach(img_array)
print('''Real surface area by stereological approach: {:.4f}. Elapsed \
time is {:.4f} seconds.'''.format(ssa_sa, time.time()-tic))

tic = time.time()
ssa_e = mcp.real_surface_extrapolation(img_array)
print('''Real surface area by iterative approach is {:.4f}. Elapsed time\
is {:.4f} seconds.'''.format(ssa_e, time.time()-tic))

tic = time.time()
ssa_e2 = mcp.real_surface_differentiation_S2(img_array)
print('''Real surface area by differentiation of S2 approach is {:.4f}. \
Elapsed time is {:.4f} seconds.'''.format(ssa_e2, time.time()-tic))

print("**Real surface area evaluation from from MultComPy.py done!")


# =============================================================================
# Lineal path function (L2)
# =============================================================================


print("Computing L2 from MultComPy.py ....")
tic = time.time()
newrow = int(img_array.shape[0]/2)+1
newcol = int(img_array.shape[1]/2)+1

L2_1 = mcp.L2_direct_computation(img_array, (newrow, newcol), step=1,
                                  method="py")
L2_tr1 = mcp.transform_ND_to_1D(L2_1, rmax=newrow)
toc = time.time()-tic
print("**L2 evaluation from MultComPy.py done! eval. time = {}".format(toc))

# print("Computing L2 from lineal_path_c.dll ....")
# tic = time.time()
# newrow = int(img_array.shape[0]/2)+1
# newcol = int(img_array.shape[1]/2)+1
# # newdep, newrow, newcol = img_array.shape
# # L2 = mcp.L2_direct_computation_dll(img_array, newdep, newrow, newcol)
# L2 = mcp.L2_direct_computation(img_array, (newrow, newcol), step=1,
#                                 method="dll")
# L2_tr2 = mcp.transform_ND_to_1D(L2, rmax=newdep)
# toc = time.time()-tic
# print("L2 evaluation from lineal_path_c.dll done! eval. time = {}".format(toc))


# graphical representation of the results
ax = image_init()

ax.plot(L2_tr1[1], L2_tr1[0], '-o', color='y', markersize=5,
        label="L2 MultComPy (py)", linewidth=1)

# ax.plot(L2_tr2[1], L2_tr2[0], '-s', color='y', markersize=5,
#         label="L2 MultComPy (dll)", linewidth=1)

image_final()


# =============================================================================
# CHORD LENGTH DENSITY FUNCTION (CLD)
# =============================================================================

print("Computing Chord length density function from MultComPy.py ....")
tic = time.time()
CLD_ortho = mcp.chordLengthDensityFunction_orthogonal(img_array,phase=True)
toc = time.time()-tic
print("Computing Chord length density function from MultComPy.py done! Eval. time = {}".format(toc))

fig, ax = plt.subplots()
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

x = list(range(1,len(CLD_ortho[0])+1))
ax.bar(x, CLD_ortho[0], color='r',label="CLD MultComPy, ax 0",linewidth=2, alpha=0.5)
x = list(range(1,len(CLD_ortho[1])+1))
ax.bar(x, CLD_ortho[1], color='g',label="CLD MultComPy, ax 1",linewidth=2, alpha=0.5)

ax.legend(loc='upper center')
plt.show()

print("**imageProcess_new.py done!")
