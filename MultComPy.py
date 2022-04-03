# -*- coding: utf-8 -*-

"""
Statistical descriptors and functions for image analysis.
MultComPy (Multiphase composite in Python) contains functions for evaluating
statistical descriptors using image analysis from realistic paste microstructures 
obtained from computed microtomography.
Created on Sun Feb 14 09:08:28 2021
Last edit on Mon Feb 02 21:45 2022
@authors:   Adela Hlobilova, adela.hlobilova@gmail.com
            Michal Hlobil, michalhlobil@seznam.cz
"""

from PIL import Image
import numpy as np
import time
import scipy.ndimage as spim
import scipy.spatial as sptl
from sys import stdout
import ctypes
import ctypes.util
import copy
import sys
import matplotlib.pyplot as plt
import os
import pathlib
import platform

##############################################################################


def my_shift(x):
    """
    Shift the zero-frequency component to the centre of the spectrum.
    This function works similarly to the fftshift from numpy.fft, but it
    enlarges the descriptor from [M,N,O] to [2*M-1,2*N-1,2*O-1]. The maximum
    value of the descriptor (volume fraction of the selected phase) is shifted
    to the middle of the descriptor instead of x[0,0,0] as is in the original
    variable x.
    Parameters
    ----------
    x : float NumPy array
        Input array. Contains only half-space of the statistical descriptor.
    Returns
    -------
    retval : float NumPy array
        The shifted array.
    """
    nd = len(x.shape)
    sz = np.array(x.shape)
    idx = []
    for i in range(0, nd):
        idx.append(list(range(1, sz[i]))+list(range(0, sz[i])))
    retval = x[np.ix_(*idx)]

    return retval


##############################################################################


def S2_direct_computation(img_array1, img_array2, larger=True):
    """
    Two-point probability function by brute force evaluation.
    Two-point probability function (correlation function) represents
    a probability, that two randomly chosen points x1 and x2 (pixels for 2D,
    voxels for 3D) belong to the prescribed phase. This implementation works
    for strictly statistically homogeneous materials. For the auto-correlation
    function, img_array1 is identical to img_array2, whereas for different
    images, e.g., if img_array2 is a logical negation of img_array1 (for
     two-phase medium), this algorithm evaluates the cross-correlation
    function. The prescribed phase is always True (logical) for both images.
    This implementation is brute force, therefore, it is slower than evaluation
    using Fast Fourier transformation.
    Parameters
    ----------
    img_array1 : Boolean NumPy array
        2D or 3D logical array containing information about the examined phase
        as True
    img_array2 : Boolean NumPy array
        2D or 3D logical array with the same shape as img_array2. If img_array1
        and img_array2 are identical, the algorithm evaluates the
        auto-correlation function. Otherwise, the result of this function is
        the cross-correlation function (a necessary condition is that
        img_array2 is opposite to img_array1 for a two-phase medium).
    larger : Boolean, optional
        The original size of the two-point probability function is larger than
        the original shape of the image (larger=True). If larger is set to
        False, the output of this function is only a section of the two-point
        probability function. Subsequently, the maximum value of the two-point
        probability function is located in S2_mat[0,0,0]. This value is the
        one-point probability function equal to the volume fraction of the
        selected phase if img_array1 and img_array2 are identical. If larger
        is set to True, the S2_mat contains the whole statistical descriptor
        and the origin of the coordinate system, i.e. the one-point probability
        function, is positioned in the middle of the output.
    Returns
    -------
    S2_mat : float NumPy array
        Two-point matrix probability function corresponding to the number of
        dimensions of the original image (for the shape of S2_mat, please,
        refer to the optional parameter larger). If variable larger is set to
        False, each element [m,n,o] of S2_mat represents the probability, that
        endpoints of a segment [0,0,0] and [m,n,o] translated along with the
        medium lie in the same phase for the auto-correlation function or in
        the different phases for the cross-correlation function, and,
        therefore, it is a measure, how the endpoints are correlated in the
        same phase or the different phases, respectively. For conversion from
        the two-point matrix probability function to the two-point probability
        function, refer to transform_ND_to_1D() (which only works for the
        statistically homogeneous and isotropic media).
    """
    if img_array1.shape != img_array2.shape:
        print('''Images must have identical dimensions (depths, rows, columns).
              Terminating without evaluations.''')
        return "Error"

    original_ndim = img_array1.ndim

    if original_ndim == 2:
        img_array1 = np.expand_dims(img_array1, axis=0)
        img_array2 = np.expand_dims(img_array2, axis=0)

    dep, row, col = img_array1.shape
    S2_mat = np.zeros((dep, 2*row-1, 2*col-1))
    numvecs = dep*(2*row-1)*(2*col-1)
    vecs = np.zeros((numvecs, 3), dtype=int)

    count = 0
    for i in range(0, dep):
        for j in range(-row+1, row):
            for k in range(-col+1, col):
                vecs[count, 0] = i
                vecs[count, 1] = j
                vecs[count, 2] = k
                count += 1

    for i in range(0, dep):
        for j in range(row, 2*row):
            for k in range(col, 2*col):
                if img_array1[i % dep, j % row, k % col] == True:
                    for m in range(0, numvecs):
                        if (img_array2[(i+vecs[m, 0]) % dep,
                                       (j+vecs[m, 1]) % row,
                                       (k+vecs[m, 2]) % col] == True):
                            S2_mat[vecs[m, 0],
                                   vecs[m, 1]+row-1,
                                   vecs[m, 2]+col-1] += 1
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            stdout.write("\r%d out of %d, %d %% done, local time: %s" %
                         (i*row*col+(j-row)*col+(k-col)+1,
                          dep*row*col,
                          (i*row*col+(j-row)*col+(k-col)+1)/(dep*row*col)*100,
                          current_time))
    print("\n")
    S2_mat = S2_mat / (row*col*dep)
    S2_mat = S2_mat[:, row-1:, col-1:]

    if original_ndim == 2:
        S2_mat = np.squeeze(S2_mat, axis=0)

    if larger:
        S2_mat = my_shift(S2_mat)

    return S2_mat

##############################################################################


def S2_Discrete_Fourier_transform(img_array1, img_array2, larger=True):
    """
    Two-point probability function evaluated by discrete Fourier transform.
    Two-point probability function (correlation function) represents
    a probability, that two randomly chosen points x1 and x2 (pixels for 2D,
    voxels for 3D) belong to the prescribed phase. This implementation works
    for strictly statistically homogeneous materials. For the auto-correlation
    function, img_array1 is identical to img_array2, whereas for different
    images, e.g., if img_array2 is a logical negation of img_array1 (for
    two-phase medium), this algorithm evaluates the cross-correlation function.
    The prescribed phase is always True (logical) for both images. This
    implementation uses Fast Fourier transformation, therefore, it is faster
    than evaluation using brute force (S2_direct_computation()).
    Parameters
    ----------
    img_array1 : Boolean NumPy array
        2D or 3D logical array containing information about the examined
        phase as True
    img_array2 : Boolean NumPy array
        2D or 3D logical array with the same shape as img_array2. If img_array1
        and img_array2 are identical, the algorithm evaluates the
        auto-correlation function. Otherwise, the result of this function is
        the cross-correlation function (a necessary condition is that
        img_array2 is opposite to img_array1 for a two-phase medium).
    larger : Boolean, optional
        The original size of the two-point probability function is larger than
        the original shape of the image (larger=True). If larger is set to
        False, the output of this function is only a section of the two-point
        probability function. Subsequently, the maximum value of the two-point
        probability function is located in S2_mat[0,0,0]. This value is the
        one-point probability function equal to the volume fraction of the
        selected phase if img_array1 and img_array2 are identical. If larger
        is set to True, the S2_mat contains the whole statistical descriptor
        and the origin of the coordinate system, i.e. the one-point probability
        function, is positioned in the middle of the output.
    Returns
    -------
    S2_mat : float NumPy array
        Two-point matrix probability function corresponding to the number of
        dimensions of the original image (for the shape of S2_mat, please,
        refer to the optional parameter larger). If variable larger is set to
        False, each element [m,n,o] of S2_mat represents the probability, that
        endpoints of a segment [0,0,0] and [m,n,o] translated along with the
        medium lie in the same phase for the auto-correlation function or in
        the different phases for the cross-correlation function, and,
        therefore, it is a measure, how the endpoints are correlated in the
        same phase or the different phases, respectively. For conversion from
        the two-point matrix probability function to the two-point probability
        function, refer to transform_ND_to_1D() (which only works for the
        statistically homogeneous and isotropic media).
    """
    if img_array1.shape != img_array2.shape:
        print('''Images must have identical dimensions (depths, rows, columns).
              Terminating without evaluations.''')
        return "Error"

    cum = np.prod(np.array(img_array1.shape))

    C = np.fft.fftn(img_array1)
    D = np.fft.fftn(img_array2)
    E = C * np.conj(D)

    test = np.max(np.max(np.imag(E)))
    if test > 0.01:
        print("warning: max. abs. value of imag. part in FFTN is :", test)

    E = np.real(E)
    S2_ND = np.real((np.fft.ifftn(E)) / cum)

    if larger:
        S2_ND = my_shift(S2_ND)

    return S2_ND

##############################################################################


def transform_ND_to_1D(X, step=1, rmax="max", D=1, scale=0):
    """
    Matrix descriptor transformation into vector.
    Transforms matrix representation of the statistical descriptor S_mn(x1,x2)
    into S_mn(r), where r=|x1-x2|. For statistically isotropic media, the joint
    probability density function describing the stochastic process is
    rotationally invariant and, therefore, S_mn depends only on the distances
    between points x1 and x2. The transformation is provided by a sequential
    numerical curve integration for an identical perimeter r from 1 to rmax.
    Parameters
    ----------
    X : 2D or 3D float NumPy array
        Input array. This array can be a statistical descriptor, i.e. two-point
        matrix probability function (2D or 3D).
    step : int, optional
        An integer number specifying the incrementation in the horizontal axis
        of the graph. Default is 1.
    rmax : int or "max", optional
        An integer number specifying at which position of the horizontal axis
        to stop with the sequential numerical integration. If "max" is set and
        the original image has shape [M,N,O], rmax is equal to
        minimum((M,N,O)). Default is "max".
    D : int, optional
        The scale of the horizontal axis mainly used for regular inclusions.
        D is equal to the diameter of the inclusion.
    scale : 0, 1, 2, optional
        The scale of the vertical axis. If the scale is equal to 1, all the
        values of the statistical descriptors S_mn(r) equals their original
        values divided by the volume fraction of the selected phase
        (i.e. S_mn(0)). S_mn(0) after scaling is then equal to 1 and other
        values are positive. If the scale is equal to 2, the values are
        linearly mapped between values 1 and 0 with the original values equal
        to S_mn(0) and S_mn(0)**2. S_mn(0) after scaling is then equal to 1
        and other values can also negative if the original value of the
        descriptor is lesser than S_mn(0)**2. If the scale is equal to 0,
        no scaling is provided. Default is 0.
    Returns
    -------
    X_vec_val : list containing 2 (in case D=1) or 3 numpy arrays
                (in case D!=1)
        The first element of the list contains the descriptor S_mn(r).
        The second element of the list contains the corresponding r lengths of
        S_mn(r). If D is not equal to 1, the third element of the list contains
        the corresponding r lengths of S_mn(r) rescaled with the parameter D.
    """
    im_shape = ((np.array(X.shape)+1)/2).astype(int)  # size of the orig. image
    dim = len(X.shape)

    if rmax == 'max':
        rmax = min(im_shape)

    coords = []
    idx = []

    # cut the medium to the required shape according to rmax
    for i in range(dim):
        coords.append(list(range(-rmax+1, rmax)))  # X cut to square
        idx.append(list(range(im_shape[i]-rmax, im_shape[i]+rmax-1)))
    X_cut = X[np.ix_(*idx)]

    X_vec_r = np.arange(0, rmax+1, step)
    X_vec_val = np.zeros(len(X_vec_r))
    count = np.zeros(len(X_vec_r))

    if dim == 2:
        pts = np.meshgrid(np.array(coords[0]),
                          np.array(coords[1]))
        crds = np.vstack([pts[0].flatten('F'),
                          pts[1].flatten('F')]).T
        dist = sptl.distance.cdist(np.array([[0, 0]]), crds, 'euclidean')
        dist = np.round(np.reshape(dist, (len(coords[0]), len(coords[1]))))

    elif dim == 3:
        pts = np.meshgrid(np.array(coords[0]),
                          np.array(coords[1]),
                          np.array(coords[2]))
        crds = np.vstack([pts[0].flatten('F'),
                          pts[1].flatten('F'),
                          pts[2].flatten('F')]).T
        dist = sptl.distance.cdist(np.array([[0, 0, 0]]), crds, 'euclidean')
        dist = np.round(np.reshape(dist,
                                   (len(coords[0]),
                                    len(coords[1]),
                                    len(coords[2]))))

    else:
        print("error: unknown shape")
        return None

    for i in range(len(X_vec_r)):
        temp = X_cut[dist == i]
        X_vec_val[i] = np.sum(temp)
        count[i] = len(temp)

    X_vec_val = np.delete(X_vec_val, -1)
    X_vec_r = np.delete(X_vec_r, -1)
    count = np.delete(count, -1)

    for i in range(0, len(X_vec_r)):
        if (count[i] != 0):
            X_vec_val[i] = X_vec_val[i] / count[i]

    if D != 1:
        X_vec_r_scale = X_vec_r / D

    if scale == 1:
        phi = X_vec_val[0]
        X_vec_val = X_vec_val / phi
    elif scale == 2:
        phi = X_vec_val[0]
        X_vec_val = (X_vec_val - phi**2) / (phi - phi**2)

    retval = []
    retval.append(X_vec_val)
    retval.append(X_vec_r)

    if D != 1:
        retval.append(X_vec_r_scale)

    return retval

###############################################################################


def BW_morph_remove(img_array, phase):
    """
    Remove interior pixels to leave an outline of the shapes.
    This function removes interior pixels in the img_array to leave an outline
    of the shapes. It operates in an 8-connected neighbourhood. If only
    a 4-connected neighbourhood is necessary, comment on the third and the
    fourth condition in the inner if statement. The function gives identical
    output as img_file.filter(ImageFilter.FIND_EDGES) - the difference is in
    the output object.
    The alternative code is
        img_edges = img_file.filter(ImageFilter.FIND_EDGES)
        edges = np.array(img_edges)
    where img_file is a BW image
    Parameters
    ----------
    img_array : integer or Boolean (or possibly float) NumPy 2D array
        Input array containing the image.
    phase : integer or Boolean (or possibly float)
        A value corresponding to at least one element in img_array. This value
        could also be float if this float value appears in img_array.
    Returns
    -------
    edges : integer NumPy 2D array
        The output array containing only the edges (elements with value 1).
        The rest of the image elements equal to 0.
    """
    row, col = img_array.shape
    edges = np.zeros((row, col), dtype=int)

    for i in range(1, row-1):
        for j in range(1, col-1):
            if img_array[i, j] == phase:
                if ((img_array[i+1, j] != img_array[i-1, j])
                        or (img_array[i, j+1] != img_array[i, j-1])
                        or (img_array[i+1, j+1] != img_array[i-1, j-1])
                        or (img_array[i+1, j-1] != img_array[i-1, j+1])):
                    edges[i, j] = 1

    PIL_image = Image.fromarray((edges*255))
    if PIL_image.mode != 'RGB':
        PIL_image = PIL_image.convert('RGB')
    PIL_image.show()

    return edges

##############################################################################


def find_edges(img_array, flag="dilate", it=1):
    """
    Edge detection of the input ND array using scipy.ndimage.
    The result is an array with the same dimensions containing True on edges
    pixels or voxels and False elsewhere.
    Parameters
    ----------
    img_array : Boolean numpy array
        Contains the N-dimensional input medium.
    flag : "dilate" or "erode", optional
        If "dilate", the algorithm returns the edges pixels/voxels that are
        outside the original cluster, i.e. the edge pixels/voxels wrap original
        clusters.
        If "erode", returning array edges contains pixels/voxels that are on
        the surfaces of the original clusters, i.e. the edge pixels/voxels were
        parts of the original clusters and the variable erode contains the rest
        of these clusters.
    it : int, optional
        The dilation or erosion is repeated iterations times (one, by default).
        If the number of iterations is lesser than 1, the dilation is repeated
        until the result does not change anymore. Only an integer of iterations
        is accepted.
    Returns
    -------
    edges : Boolean NumPy array
        The edges of the input media.
    dilate / erode : Boolean NumPy array
        Erosion or dilation of the input by the structuring element.
    """
    dim = img_array.ndim
    if dim == 2:
        struct = spim.generate_binary_structure(dim, dim)
    elif dim == 3:
        # struct = spim.generate_binary_structure(dim, dim)
        struct = np.array([[[False,True,False],[True,True,True],[False,True,False]],
               [[True,True,True],[True,True,True],[True,True,True]],
               [[False,True,False],[True,True,True],[False,True,False]]])
        

    if flag == "erode":
        erode = spim.binary_erosion(img_array, struct, iterations=it)
        edges = img_array ^ erode
        return edges, erode
    elif flag == "dilate":
        dilate = spim.binary_dilation(img_array, struct, iterations=it)
        edges = img_array ^ dilate
        return edges, dilate


##############################################################################

def real_surface_stereological_approach(im):
    """
    Interface area by stereological approach.
    Parameters
    ----------
    im: Boolean NumPy array
        Input array. True denotes the selected phase for which the algorithm
        evaluates a real surface area.
    Returns
    -------
    ssa: float
        Real surface area in [vox^2].
    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
    Microstructure and Macroscopic Properties. Interdisciplinary Applied
    Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    [2] Pascal HAGENMULLER, Modélisation  du  comportementmécanique de la neige
    à partird’images microtomographiques, 2014, Université de Grenoble, PhD
    thesis
    [3] Hagenmuller, Pascal & Matzl, Margret & Chambon, Guillaume & Schneebeli,
    Martin. (2016). Sensitivity of snow density and specific surface area
    measured by microtomography to different image processing algorithms.
    The Cryosphere Discussions. 1-28. 10.5194/tc-2015-217.
    """
    im = im.astype(int)
    im_shape = im.shape
    dim = im.ndim

    rsa = 0

    for i in range(dim):
        coords = []
        for j in range(dim):
            if i == j:
                coords.append(list(range(1, im_shape[j]))+[im_shape[j]-1])
            else:
                coords.append(list(range(0, im_shape[j])))
        im2 = im[np.ix_(*coords)]
        rsa += np.sum(np.abs(im-im2))

    # for a specific surface area uncomment following code:
    # ssa /= np.prod(im_shape)

    return rsa

##############################################################################


def real_surface_extrapolation(im, max_iter=5):
    """
    Real interface area of particle by extrapolation to zero thickness of 
    dilation region.
    
    NOTE: to obtain specific surface area, rsa needs to be divided by product 
    of ROI size (= ROI volume).
    (According to [1], the specific surface is a one-point correlation function
    and is proportional to the probability of finding a point in the dilated
    region around the spheres.)
    
    The dilation region is created by uniformly and radially dilating 
    the spheres by a differential amount deltaR. But in a simulation, this 
    thickness must be finite, which we denote DeltaR which tends to zero. 
    Therefore, several values of the thickness must be considered and 
    the result is extrapolated to the limit that DeltaR -> 0.
    
    NOTE: in principle, this method fails to calculate the area of small 
    particles since the particle morphology is seriously distrupded by adding 
    the voxel-thick delated region around the particle. As such, the real area
    obtained by this method may be negative for small particles!
    Parameters
    ----------
    im : Boolean NumPy array
        Input array. True denotes the selected phase for which the algorithm
        evaluates a real surface area.
    max_iter : int, optional
        The number of iterations for different DeltaR. The initial value of
        DeltaR is 1, and it increases with 1. Max_iter is the maximum DeltaR.
        Default is 5 (which means 5 iterations with DeltaR = [1,2,3,4,5]).
    Returns
    -------
    rsa : float
        Real surface area.
    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
        Microstructure and Macroscopic Properties. Interdisciplinary Applied
        Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    """
    s = np.zeros(max_iter)
    DeltaR = np.array(range(1, max_iter+1))

    for i in range(len(DeltaR)):
        edges, dilate = find_edges(im, it=DeltaR[i])
        # s[i] = np.sum(edges)/DeltaR[i]/np.prod(im.shape)
        s[i] = np.sum(edges)/DeltaR[i]
        
    x = np.ones((2, max_iter))
    x[1, :] = DeltaR
    x = x.T

    beta = np.linalg.lstsq(np.matmul(x.T, x), np.matmul(x.T, s.T), rcond=None)

    # for a specific surface area uncomment following code:
    # ssa = beta[0][0]*np.prod(im.shape)
    rsa = beta[0][0]

    return rsa

##############################################################################


def real_surface_differentiation_S2(img_array):
    """
    Real surface area by differentiation of the autocorrelation function.
    (According to [1] (the third equation in (2.5)), the specific surface area
    is equal to a proportion of the differentiation of the auto-correlation
    function in r=0, where r is the length of the segment that starts and ends
    in the same phase. Since the medium is discretized, also the
    auto-correlation function is discrete and only the first and the second
    element from S2() contributes to the specific surface area.)
    NOTE: I found in a literature (but I cannot find the original print) that
    specific surface area for ND should be evaluated as
        ssa = (S2_[0][0] - S2_[0][1])*2*dim
        #  dim is equal to number of dimensions
    [1] says that it should be dim=2 according to the equation above for number
    of dimensions up to 3. In comparison with different evaluations of specific
    surface area,
        ssa = (S2_[0][0] - S2_[0][1])*4
    works well.
    Parameters
    ----------
    img_array : Boolean NumPy ND array
        Input array of the medium.
    Returns
    -------
    rsa : float
        Real surface area.
    References
    ----------
    [1]  David A. Coker and Salvatore Torquato, "Extraction of morphological
    quantities from a digitized medium", Journal of Applied Physics 77,
    6087-6099 (1995) https://doi.org/10.1063/1.359134
    """
    S2 = S2_Discrete_Fourier_transform(img_array, img_array)
    # After transformation from matrix into vector notation, only first two
    # elements of the vector contributes to ssa.
    S2_ = transform_ND_to_1D(S2, rmax=2)
    # specific surface area is proportional to differentiation of the S2()
    ssa = (S2_[0][0] - S2_[0][1])*4

    im = img_array.astype(int)
    im_shape = im.shape
    # to obtain real surface area, the specific area is multiplied by the ROI
    # volume
    rsa= ssa*np.prod(im_shape)
    
    return rsa

###############################################################################


def C2_Discrete_Fourier_transform(img_array, larger=True):
    """
    Two-point cluster function evaluated by Discrete Fourier Transformation.
    Two-point cluster function C2() is a probability, that two randomly chosen
    points (pixels or voxels) x1 and x2 belong to the same cluster of the
    selected phase. This phase is always True in the input medium array
    img_array. Two points are in the same cluster if they can be connected
    within the same phase.
    Parameters
    ----------
    img_array : Boolean NumPy ND array
        Input array of the medium. True represents the selected phase.
    larger : Boolean, optional
        The original size of the two-point cluster function is larger than the
        original shape of the image (larger=True). If larger is set to False,
        the output of this function is only a section of the two-point cluster
        function. Subsequently, the maximum value of the two-point cluster
        function is located in C2_mat[0,0,0]. This value is the one-point
        cluster function equal to the volume fraction of the selected phase.
        If larger is set to True, the C2_mat contains the whole statistical
        descriptor and the origin of the coordinate system, i.e. the one-point
        cluster function, is positioned in the middle of the output.
    Returns
    -------
    C2_mat : float NumPy ND array
        Two-point matrix cluster function corresponding to the number of
        dimensions of the original image (for the shape of C2_mat, please,
        refer to the optional parameter larger). If variable larger is set to
        False, each element [m,n,o] of C2_mat represents the probability, that
        endpoints of a segment [0,0,0] and [m,n,o] translated along the medium
        lie in the same cluster, and, therefore, it is a measure, how the
        endpoints are correlated in the same cluster. C2() therefore describes
        shapes of inclusions rather than their distribution within the domain.
        For conversion from two-point matrix cluster function to two-point
        cluster function, refer to transform_ND_to_1D() (which only works for
        the statistically homogeneous and isotropic media).
    """
    shape = img_array.shape
    img_array_01, num_clstrs = spim.label(img_array)

    newshape = (np.array(shape)*2-1).astype(int)
    C2_mat = np.zeros(newshape)

    for i in range(num_clstrs):
        A = img_array_01 == i+1
        C2_mat += S2_Discrete_Fourier_transform(A, A, larger)

    return C2_mat

###############################################################################


def BresLineAlg(x0, y0, x1, y1):
    r"""
    2D Bressenham's line algorithm.
    The original algorithm works only for:
        1/ the octants, where the change in x is greater than the change in y,
           i.e. the growth of x is faster than y (variable is_steep is False,
           octants 1, 4, 5, and 8). Otherwise, the x and y coordinates are
           swapped for the evaluation (variable is_steep is True, octants 2, 3,
           6, and 7) but the x and y coordinates are recorded opposite, and for
        2/ the octants, where the x coordinates grow from the initial point to
           the endpoint, i.e. x0 > x1 (variable swap is False). Otherwise, the
           initial and the endpoints are swapped (variable swap is is True).
        For the line consistency, all the points in the line are reversed at
        the end of the algorithm in case that they were swapped at the
        beginning to keep the initial and the endpoint the same as in the
        input arguments. Variable swap is False in octants 1,2, 3 and 8 since
        the steepness is checked and treated as the first step. If the
        steepness was examined second, octants 3, 4, 5 and 6 would have
        variable swap equal to False.
    Octants and their numbering are as follows:
                    \ 6 | 7 /
                     \  |  /
                    5 \ | / 8
                       \|/
                    ----*----
                       /|\
                    4 / | \ 1
                     /  |  \
                    / 3 | 2 \
    Parameters
    ----------
    x0 : int
        The initial x coordination of the segment.
    y0 : int
        The initial y coordination of the segment.
    x1 : int
        The end x coordination of the segment.
    y1 : int
        The end y coordination of the segment.
    Returns
    -------
    coords : list of tuples
        The list of points (each point is represented by one tuple) with the
        final Bresenham's path.
    """
    dx = x1-x0
    dy = y1-y0

    is_steep = abs(dx) < abs(dy)
    if is_steep:
        dx, dy = dy, dx
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    swap = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swap = True
        dx, dy = -dx, -dy

    error = int(dx/2.0)
    ystep = 1 if y0 < y1 else -1

    y = y0
    coords = []

    for x in range(x0, x1+1):
        if is_steep:
            coords.append((y, x))
        else:
            coords.append((x, y))

        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    if swap:
        coords.reverse()

    return coords

#############################################################################


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    """
    Bresenham’s Algorithm for 3-D Line Drawing.
    Overtaken from
    https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
    Parameters
    ----------
    x1 : int
        axis 0 - coordinate of point 1
    y1 : int
        axis 1 - coordinate of point 1
    z1 : int
        axis 2 - coordinate of point 1
    x2 : int
        axis 0 - coordinate of point 2
    y2 : int
        axis 1 - coordinate of point 2
    z2 : int
        axis 2 - coordinate of point 2
    Returns
    -------
    ListOfPoints : list
        Points on the line joining point 1 and point 2.
    """
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is x-axis
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is y-axis
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints

###############################################################################


def L2_generate_paths(row, col, imgrow, imgcol, step=1):
    """
    Generate paths via Bressenham's 2-D line algorithm.
    Helper function for the L2_direct_computation() function.
    Parameters
    ----------
    row : int
        The number of rows of the path with the maximum length.
    col : int
        The number of columns of the path with the maximum length.
    imgrow : int
        The number of rows in the image.
    imgcol : int
        The number of columns in the image.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths is skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    Returns
    -------
    paths : list of tuples
        Contains all the generated paths. Each item contains one path and each
        path contains a list of tuples representing (x,y) coordinates of the
        points that together for the whole path.
    count : NumPy int array
        Represents how many times is the path used in the medium.
    """
    paths = []
    count = np.zeros((imgrow, 2*imgcol-1))

    issame = True if imgrow == row and imgcol == col else False

    # path generation - horizontal axis is constant
    for j in (-col+1, col-1):
        for i in range(0, row, step):
            coords = BresLineAlg(0, 0, i, j)
            paths.append(coords)

            if issame:
                for k in range(0, len(coords)):
                    count[coords[k][0], coords[k][1]+col-1] += row*col

    # path generation - vertical axis is constant
    for j in range(-col+2, col-1, step):
        i = row-1
        coords = BresLineAlg(0, 0, i, j)
        paths.append(coords)

        if issame:
            for k in range(0, len(coords)):
                count[coords[k][0], coords[k][1]+col-1] += row*col

    if not issame:
        for i in paths:
            count[[k[0] for k in i],
                  [k[1]+imgcol-1 for k in i]] += imgrow*imgcol

    return paths, count

##############################################################################


def L2_generate_paths_3D(dep, row, col, imgdep, imgrow, imgcol, step=1):
    """
    Generate paths via Bressenham's line 3-D algorithm.
    Helper function for the L2_direct_computation_3D() function.
    Parameters
    ----------
    dep : int
        The number of depths of the path with the maximum length.
    row : int
        The number of rows of the path with the maximum length.
    col : int
        The number of columns of the path with the maximum length.
    imgdep : int
        The number of depths in the image.
    imgrow : int
        The number of rows in the image.
    imgcol : int
        The number of columns in the image.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths are skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    Returns
    -------
    paths : list of tuples
        Contains all the generated paths. Each item contains one path and each
        path contains a list of tuples representing (x,y,z) coordinates of the
        points that together for the whole path.
    count : NumPy int array
        Represents how many times the path is used in the medium.
    """
    paths = []
    count = np.zeros((imgdep, 2*imgrow-1, 2*imgcol-1))

    # path generation - axis 0 is constant:
    for i in [0, dep-1]:
        for j in range(-row+1, row, step):
            for k in range(-col+1, col, step):
                coords = Bresenham3D(0, 0, 0, i, j, k)
                paths.append(coords)

    # path generation - axis 1 is constant:
    for i in range(1, dep-1, step):
        for j in [-row+1, row-1]:
            for k in range(-col+1, col, step):
                coords = Bresenham3D(0, 0, 0, i, j, k)
                paths.append(coords)

    # path generation - axis 2 is constant:
    for i in range(1, dep-1, step):
        for j in range(-row+2, row-1, step):
            for k in [-col+1, col-1]:
                coords = Bresenham3D(0, 0, 0, i, j, k)
                paths.append(coords)

    for i in paths:
        count[[k[0] for k in i],
              [k[1]+imgrow-1 for k in i],
              [k[2]+imgcol-1 for k in i]] += imgdep*imgrow*imgcol

    return paths, count

##############################################################################


def L2_direct_computation_2D(A, rowmax, colmax, phase=True, step=1):
    """
    Lineal path 2-D function by brute force algorithm.
    Lineal path function is a probability that a line segment lies in the same
    phase when randomly thrown into the microstructure [1]. First, all the
    paths are generated (by function L2_generate_paths). Since the lineal path
    is point symmetric [2], only a half of the paths is generated and the
    second half of the lineal path function is obtained thanks to the symmetry.
    I.e. the algorithm generates all paths from 0 degrees to 180 degrees with
    different lengths (from length = min(row, columns) to
                       length = number of image max(rows, columns)).
    Only the longest possible paths are generated and the shorter ones are
    omitted since they are already contained in the longer ones. The less
    accuracy is since the shorter paths generated by Bressenham's line
    algorithm with the same angle may differ in some pixels. This algorithm
    is inspired by the Torquato algorithm (Random heterogeneous materials [1],
    pp. 291). The Lineal path algorithm translates all paths along with the
    medium.
    v0.0 - This algorithm generated the paths for all the pixels in the medium.
           This version was much slower but slightly more accurate than the
           next version.
    v1.0 - the paths were generated only with angle 0-90 degrees
    v1.1 - the paths are generated with angle 0-180 degrees (the  original
           image is copied 3 times horizontally and 2 times vertically)
    v1.2 - L2 descriptor is symmetric, this version is generating the whole
    descriptor, instead of its half
    Parameters
    ----------
    A : numpy 2D array
        Input array of the medium.
    rowmax : int
        The maximum number of pixels of the path in the vertical direction.
    colmax : int
        The maximum number of pixels of the path in the horizontal direction.
    phase : Boolean / integer / float, optional
        The value has to correspond to at least one element in variable A and
        it represents the phase that is considered for the lineal path
        algorithm. The default is True.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths are skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    Returns
    -------
    L2_mat : numpy float array
        The lineal path matrix function.
    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
    Microstructure and Macroscopic Properties. Interdisciplinary Applied
    Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    [2] Havelka, J., Kučerová, A., & Sýkora, J. (2016). Compression and
    reconstruction of random microstructures using accelerated lineal path
    function. Computational Materials Science, 122, 102-117.
    """
    row, col = A.shape
    L2_mat = np.zeros((row, 2*col-1))

    paths, count = L2_generate_paths(rowmax, colmax, row, col, step)

    numvecs = len(paths)

    for i in range(0, row):
        for j in range(col, 2*col):
            for k in range(0, numvecs):   # kth path out of row*col paths
                if A[i % row, j % col] == phase:
                    m = 0
                    while(m < len(paths[k])
                          and A[(i+paths[k][m][0]) % row,
                                (j+paths[k][m][1]) % col] == phase):
                        L2_mat[paths[k][m][0], paths[k][m][1]+col-1] += 1
                        m += 1

    if row == rowmax and col == colmax:
        L2_mat /= count
    else:
        # since paths can be shorter than the medium, the original L2_mat has
        # to be cut to matrix-size count
        idx = []
        idx.append(list(range(0, rowmax)))
        idx.append(list(range(col-colmax, col+colmax-1)))
        L2_cut = L2_mat[np.ix_(*idx)]
        L2_cut = L2_cut / count[np.ix_(*idx)]
        L2_mat = copy.deepcopy(L2_cut)

    # L2 is symmetric, rotating the upper half part
    L22 = np.rot90(L2_mat, 2)
    L22 = np.delete(L22, L22.shape[0]-1, 0)
    L2_mat = np.concatenate((L22, L2_mat), axis=0)

    return L2_mat

###############################################################################


def L2_direct_computation_3D(A, depmax, rowmax, colmax, phase=True, step=1):
    """
    Lineal path 3-D function by brute force algorithm.
    Lineal path function is a probability that a line segment lies in the same
    phase when randomly thrown into the microstructure [1]. First, all the
    paths are generated (by function L2_generate_paths). Since the lineal path
    is point symmetric [2], only a half of the paths is generated and the
    second half of the lineal path function is obtained thanks to the symmetry.
    I.e. the algorithm generates all paths from 0 degrees to 180 degrees with
    different lengths (from length = min(row, columns) to
                       length = number of image max(rows, columns)).
    Only the longest possible paths are generated and the shorter ones are
    omitted since they are already contained in the longer ones. The less
    accuracy is since the shorter paths generated by Bressenham's line
    algorithm with the same angle may differ in some pixels. This algorithm
    is inspired by the Torquato algorithm (Random heterogeneous materials [1],
    pp. 291). The Lineal path algorithm translates all paths along with the
    medium.
    v0.0 - This algorithm generated the paths for all the pixels in the medium.
           This version was much slower but slightly more accurate than the
           next version.
    v1.0 - the paths were generated only with angle 0-90 degrees
    v1.1 - the paths are generated with angle 0-180 degrees (the  original
           image is copied 3 times horizontally and 2 times vertically)
    v1.2 - L2 descriptor is symmetric, this version is generating the whole
           descriptor, instead of its half.
    v2.0 - Computation for 3D images.
    Parameters
    ----------
    A : numpy 2D array
        Input array of the medium.
    depmax : int
        The maximum number of pixels of the path in the depth (axis 0)
        direction.
    rowmax : int
        The maximum number of pixels of the path in the vertical (axis 1)
        direction.
    colmax : int
        The maximum number of pixels of the path in the horizontal (axis 2)
        direction.
    phase : Boolean / integer / float, optional
        The value has to correspond to at least one element in variable A and
        it represents the phase that is considered for the lineal path
        algorithm. The default is True.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths are skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    Returns
    -------
    L2_mat : numpy float array
        The lineal path matrix function.
    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
    Microstructure and Macroscopic Properties. Interdisciplinary Applied
    Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    [2] Havelka, J., Kučerová, A., & Sýkora, J. (2016). Compression and
    reconstruction of random microstructures using accelerated lineal path
    function. Computational Materials Science, 122, 102-117.
    """
    dep, row, col = A.shape

    L2_mat = np.zeros((dep, 2*row-1, 2*col-1))
    paths, count = L2_generate_paths_3D(depmax, rowmax, colmax,
                                        dep, row, col, step)
    numvecs = len(paths)

    for i in range(0, dep):
        for j in range(row, 2*row):
            for k in range(col, 2*col):
                for m in range(0, numvecs):   # lth path out of all paths
                    if A[i % dep, j % row, k % col] == phase:
                        n = 0
                        while(n < len(paths[m])
                              and A[(i+paths[m][n][0]) % dep,
                                    (j+paths[m][n][1]) % row,
                                    (k+paths[m][n][2]) % col] == phase):
                            L2_mat[paths[m][n][0],
                                   paths[m][n][1]+row-1,
                                   paths[m][n][2]+col-1] += 1
                            n += 1

    if step == 1:
        if row == rowmax and col == colmax and dep == depmax:
            L2_mat /= count
        else:
            # since paths can be shorter than the medium, the original L2_mat
            #  has to be cut to matrix-size count
            idx = []
            idx.append(list(range(0, depmax)))
            idx.append(list(range(row-rowmax, row+rowmax-1)))
            idx.append(list(range(col-colmax, col+colmax-1)))
            L2_cut = L2_mat[np.ix_(*idx)]
            L2_cut = L2_cut / count[np.ix_(*idx)]
            L2_mat = copy.deepcopy(L2_cut)

    else:
        shape_L2 = L2_mat.shape
        for i in range(shape_L2[0]):
            for j in range(shape_L2[1]):
                for k in range(shape_L2[2]):
                    if count[i, j, k] != 0:
                        L2_mat[i, j, k] /= count[i, j, k]

        idx = []
        idx.append(list(range(0, depmax)))
        idx.append(list(range(row-rowmax, row+rowmax-1)))
        idx.append(list(range(col-colmax, col+colmax-1)))
        L2_cut = L2_mat[np.ix_(*idx)]
        L2_mat = copy.deepcopy(L2_cut)

    # L2 is symmetric, rotating the upper half part
    L22 = np.rot90(L2_mat, 2)
    L22 = np.delete(L22, L22.shape[0]-1, 0)
    L2_mat = np.concatenate((L22, L2_mat), axis=0)

    return L2_mat

##############################################################################


def L2_direct_computation_dll(A, depmax, rowmax, colmax, phase=True, step=1):
    """
    Lineal path function computed by brute force in C precompiled code.
    Lineal path function is a probability that a line segment lies in the same
    phase when randomly thrown into the microstructure [1]. First, all the
    paths are generated (by function L2_generate_paths). Since the lineal path
    is point symmetric [2], only a half of the paths is generated and the
    second half of the lineal path function is obtained thanks to the symmetry.
    I.e. the algorithm generates all paths from 0 degrees to 180 degrees with
    different lengths (from length = min(row, columns) to
                       length = number of image max(rows, columns)).
    Only the longest possible paths are generated and the shorter ones are
    omitted since they are already contained in the longer ones. The less
    accuracy is since the shorter paths generated by Bressenham's line
    algorithm with the same angle may differ in some pixels. This algorithm is
    inspired by the Torquato algorithm (Random heterogeneous materials [1],
    pp. 291). The Lineal path algorithm translates all paths along with the
    medium.
    v0.0 - This algorithm generated the paths for all the pixels in the medium.
           This version was much slower but slightly more accurate than the
           next version.
    v1.0 - The paths were generated only with angle 0-90 degrees.
    v1.1 - The paths are generated with angle 0-180 degrees (the  original
           image is copied 3 times horizontally and 2 times vertically).
    v1.2 - L2 descriptor is symmetric, this version is generating the whole
           descriptor, instead of its half
    v2.0 - Computation for 3D images.
    v3.0 - Rewritten to C, the code is compiled to *.dll and the *.dll is run
           by Python via this code.
           
    IMPORTANT NOTE: This function works only if lineal_path_c.dll exists, which
    can be created with following commands in a command line (if gcc is installed)
    
        gcc -std=c11 -Wall -Wextra -pedantic -c -fPIC lineal_path_c.c -o lineal_path_c.o
        gcc -shared lineal_path_c.o -o lineal_path_c.dll
        
    The created dll file has to be placed in the same folder as this module.
    
    Parameters
    ----------
    A : NumPy 2D array
        Input array of the medium.
    depmax : int
        The maximum number of pixels of the path in the depth (axis 0)
        direction.
    rowmax : int
        The maximum number of pixels of the path in the vertical (axis 1)
        direction.
    colmax : int
        The maximum number of pixels of the path in the horizontal (axis 2)
        direction.
    phase : Boolean / integer / float, optional
        The value has to correspond to at least one element in variable A and
        it represents the phase that is considered for the lineal path
        algorithm. The default is True.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths are skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    Returns
    -------
    L2_mat : NumPy float array
        The lineal path matrix function.
    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
    Microstructure and Macroscopic Properties. Interdisciplinary Applied
    Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    [2] Havelka, J., Kučerová, A., & Sýkora, J. (2016). Compression and
    reconstruction of random microstructures using accelerated lineal path
    function. Computational Materials Science, 122, 102-117.
    """
    
    if platform.system() == "Windows":
        path = pathlib.Path(__file__).parent.resolve()
        lineal_path_c_path = ctypes.util.find_library(path/("lineal_path_c"))
    elif platform.system() == "Linux":
        path= os.getcwd()
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith(".dll") :
                    lineal_path_c_path = ctypes.util.find_library("%s/%s" %(root, name))

    if not lineal_path_c_path:
        print("Unable to find the specified library.")
        sys.exit()

    try:
        lineal_path_c = ctypes.CDLL(lineal_path_c_path)
    except OSError:
        print("Unable to load the system C library")
        sys.exit()

    if A.ndim == 2:
        A = np.expand_dims(A, axis=0)
        # rowmax, colmax = maxsize
        origdim = 2
        depmax = 2
        A = np.concatenate((np.zeros(A.shape,dtype=int), A))
    elif A.ndim == 3:
        # depmax, rowmax, colmax = maxsize
        origdim = 3

    phase = phase+0  # converting from logical to int
    dep, row, col = A.shape
    depL2, rowL2, colL2 = 2*dep-1, 2*row-1, 2*col-1
    numel = (2*dep-1) * (2*row-1) * (2*col-1)

    Lineal_path = lineal_path_c.Lineal_path
    Lineal_path.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_double)]
    Lineal_path.restype = None

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)

    img_array_flat = copy.deepcopy(A)
    img_array_flat = np.reshape(img_array_flat, img_array_flat.size, order="C")
    img_array_flat = list(img_array_flat)
    img_array_flat = np.array(img_array_flat, dtype=ctypes.c_int)
    img_array_flat_p = img_array_flat.ctypes.data_as(c_int_p)

    LP_res = np.zeros(numel, dtype=ctypes.c_double)
    LP_res_p = LP_res.ctypes.data_as(c_double_p)

    Lineal_path(depmax, rowmax, colmax, dep, row, col,
                phase, step, img_array_flat_p, LP_res_p)
    L2_mat = np.reshape(LP_res, (depL2, rowL2, colL2))

    idx = []
    # idx.append(list(range(0, depmax)))
    idx.append(list(range(dep-depmax, dep+depmax-1)))
    idx.append(list(range(row-rowmax, row+rowmax-1)))
    idx.append(list(range(col-colmax, col+colmax-1)))
    L2_cut = L2_mat[np.ix_(*idx)]
    L2_mat = copy.deepcopy(L2_cut)

    # if origdim == 2:
    #     L2_mat = np.squeeze(L2_mat, axis=0).shape

    # maxsize = (depmax, rowmax, colmax)
    # idx = []
    # for i in range(A.ndim):
    #     start = int(((A.shape[i]*2-1) - (maxsize[i]*2-1)) / 2)
    #     idx.append(list(range(start, start + A.shape[i])))
    # L2_mat = L2_mat[np.ix_(*idx)]

    return L2_mat


##############################################################################


def L2_direct_computation(A, maxsize, phase=True, step=1, method="py"):
    """
    Lineal path function computed by brute force.
    Lineal path function is a probability that a line segment lies in the same
    phase when randomly thrown into the microstructure [1]. First, all the
    paths are generated (by function L2_generate_paths). Since the lineal path
    is point symmetric [2], only a half of the paths is generated and the
    second half of the lineal path function is obtained thanks to the symmetry.
    I.e. the algorithm generates all paths from 0 degrees to 180 degrees with
    different lengths (from length = min(depths, rows, columns) to
                       length = number of image max(depths, rows, columns)).
    Only the longest possible paths are generated and the shorter ones are
    omitted since they are already contained in the longer ones. The less
    accuracy is since the shorter paths generated by Bressenham's line
    algorithm with the same angle may differ in some pixels. This algorithm is
    inspired by the Torquato algorithm (Random heterogeneous materials [1],
    pp. 291). The Lineal path algorithm translates all paths along with the
    medium.
    v0.0 - This algorithm generated the paths for all the pixels in the medium.
           This version was much slower but slightly more accurate than the
           next version.
    v1.0 - The paths were generated only with angle 0-90 degrees.
    v1.1 - The paths are generated with angle 0-180 degrees (the  original
           image is copied 3 times horizontally and 2 times vertically).
    v1.2 - L2 descriptor is symmetric, this version is generating the whole
           descriptor, instead of its half
    v2.0 - Computation for 3D images.
    v3.0 - Rewritten to C, the code is compiled to *.dll and the *.dll is run
           by Python via this code.
    Parameters
    ----------
    A : NumPy 2D array
        Input array of the medium.
    maxsize : tuple with 1, 2, or 3 elements
        If maxsize has 1 element:
            The maximum length of the paths is square root of maxisize.
        If maxsize has 2 elements:
            The maximum number of pixels of the path in the vertical (axis 0)
            and horizontal (axis 1) direction.
        If maxsize has 3 elements:
            The maximum number of pixels of the path in the depth (axis 0),
            vertical (axis 1) and horizontal (axis 2) direction.
    phase : Boolean / integer / float, optional
        The value has to correspond to at least one element in variable A and
        it represents the phase that is considered for the lineal path
        algorithm. The default is True.
    step : int, optional
        Not all paths are necessarily used in the medium and the lineal path
        function. When generating them, some paths can be skipped. How many
        paths are skipped is determined by the step parameter, if the step is
        greater than 1, each step-th path is generated. The more paths skipped,
        the less accurate the lineal path function result is. The default is 1.
    method: "py" or "dll", optional
        Selection between python (py) or c (dll) evaluation. The default is
        "py".
    Returns
    -------
    L2_mat : NumPy float array
        The lineal path matrix function.
    """
    try:
        A.ndim == 2 or A.ndim == 3
    except Exception:
        print(Exception)
        return print("Medium has to be 2D or 3D.")

    if method == "py":

        if len(maxsize) == 1:
            if A.ndim == 2:
                L2_mat = L2_direct_computation_2D(A, maxsize, maxsize,
                                                  phase, step)
            elif A.dim == 3:
                L2_mat = L2_direct_computation_3D(A, maxsize, maxsize, maxsize,
                                                  phase, step)
        elif len(maxsize) == 2:
            L2_mat = L2_direct_computation_2D(A, *maxsize, phase, step)
        elif len(maxsize) == 3:
            L2_mat = L2_direct_computation_3D(A, *maxsize, phase, step)
        else:
            return print("maxsize can only have 1 to 3 elements.")

    elif method == "dll":
        if len(maxsize) == 1:
            L2_mat = L2_direct_computation_dll(A, maxsize, maxsize,
                                                   maxsize, phase, step)

        elif len(maxsize) == 2:
            #TODO: This variant needs to be checked:
            L2_mat = L2_direct_computation_dll(A, *(1,*maxsize), phase, step)
        elif len(maxsize) == 3:
            L2_mat = L2_direct_computation_dll(A, *maxsize, phase, step)
        else:
            return print("maxsize can only have 1 to 3 elements.")

    return L2_mat


##############################################################################


def shortest_distance_from_hydrate_to_clinker_surface(A, clinker_phase,
                                                      hydrate_phase,
                                                      flag="PBC",
                                                      num_in_batch=400):
    """
    Shortest distance from hydrate to the clinker surface.
    This function finds the shortest distance from each pixel or voxel of the
    hydrate phase to the clinker surface.
    Parameters
    ----------
    A : NumPy int array (2D or 3D)
        Input array of the medium. Each element represents one pixel or voxel
        of the original medium categorized into at least two phases (clinker
        phase and hydrate phase) distinguished from each other with different
        int values.
    clinker_phase : int
        The value of the clinker phase in A.
    hydrate_phase : int
        The value of the hydrate phase in A.
    flag : str, optional
        If "PBC", periodic boundary conditions are used. Otherwise, use any
        string. The default is "PBC".
    num_in_batch : int, optional
        Since the medium can be quite large and the computer memory too small
        for the scipy.spatial.distance.cdist function, the algorithm divides
        the data into independent batches that are computed serial. The default
        is 400.
    Returns
    -------
    min_dists : float array (with the length equal to the number of hydrates
                             in A)
        Minimum distances from hydrates to clinker surface.
    """
    hydrate_mask = A == hydrate_phase
    clinker_mask = A == clinker_phase
    surface_mask, xx = find_edges(clinker_mask, flag="erode")

    # if periodic boundary conditions are used:
    if flag == "PBC":
        newshape = tuple(np.ones(A.ndim, dtype=int)*3)
        surface_mask = np.tile(surface_mask, newshape)
        hydrate_mask_new = np.zeros(surface_mask.shape)

        if A.ndim == 2:
            hydrate_mask_new[A.shape[0]:A.shape[0]*2,
                             A.shape[1]:A.shape[1]*2] = hydrate_mask
            hydrate_mask = hydrate_mask_new
        elif A.ndim == 3:
            hydrate_mask_new[A.shape[0]:A.shape[0]*2,
                             A.shape[1]:A.shape[1]*2,
                             A.shape[2]:A.shape[2]*2] = hydrate_mask
            hydrate_mask = hydrate_mask_new

    # hudrates and clinker surface coordinates
    coords_hydrate = np.argwhere(hydrate_mask)
    coords_surface = np.argwhere(surface_mask)

    num_hydrates = len(coords_hydrate)
    num_batches = int(np.ceil(num_hydrates/num_in_batch))

    min_dists = np.zeros(num_hydrates)

    print("Computing {} iterations \n".format(num_batches))

    # splitting the evaluation to num_batches steps because of the computer
    # memory
    for i in range(num_batches):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        stdout.write("\r%d out of %d, %d %% done, local time: %s"
                     % (i, num_batches, i/num_batches*100, current_time))

        start = i*num_in_batch
        if i < num_batches-1:
            stop = (i+1)*num_in_batch
        elif i == num_batches-1:
            stop = num_hydrates
        dists = sptl.distance.cdist(coords_hydrate[start:stop],
                                    coords_surface, 'euclidean')
        min_dists[start:stop] = np.amin(dists, axis=1)

    if num_batches != 0:
       current_time = time.strftime("%H:%M:%S", t)
       stdout.write("\r%d out of %d, %d %% done, local time: %s \n"
                    % (i+1, num_batches, (i+1)/num_batches*100, current_time))

    vals, bin_edges = np.histogram(min_dists,range(1,int(np.ceil(np.max(min_dists))+1)))
    vals = np.append(vals,0)
    
    return vals, bin_edges

##############################################################################


def chordLengthDensityFunction_orthogonal(A, phase=True):
    """
    Chord length density function for orthogonal directions.
    This algorithm defines paths in 2 or 3 orthogonal directions along main
    axes. These paths are then translated along with the medium in the
    remaining directions for each path. A chord is a line segment between
    intersections of an infinitely long line with the two-phase interface. The
    infinitely long line is replaced by a line with a length equal to the size
    of the image in the specific direction. Chords are considered only for the
    given phase. For each possible length of the chord (i.e. from 1 to size of
    the image in the specific direction), a number of these discovered chords
    is stored in the "chords" list (this list stores chords for all directions,
    i.e. each line corresponds to one direction). All the chords in one
    direction are summed and the number of chords is divided by this sum. This
    satisfies that the resulting CLD is a probability density function with the
    area below the curve equal to one.
    Parameters
    ----------
    A : Boolean NumPy array (2D or 3D)
        Input array of the medium.
    phase : Boolean, optional
        The requested phase for the chord length density function evaluation.
        The default is True.
    Returns
    -------
    CLD : list of float arrays
        Each item of the list contains a chord length density function for each
        direction (axis 0, 1, and 2). Each element of each float array
        represents the probability that the chord with the given length is in
        the medium. In a simplified way, the chords that correspond with the
        position of the CLD element by their length are counted up and divided
        by all chords in the medium. CLD contains two or three lists depending
        on the number of dimensions.
    """
    if A.ndim == 2:
        row, col = A.shape
        shape = (row, col)

        paths = []
        paths.append(BresLineAlg(0, 0, 0, A.shape[1]-1))
        paths.append(BresLineAlg(0, 0, A.shape[0]-1, 0))

        # in chords: 0th row for horizontal direction, 1st for vertical
        chords = np.zeros((2, np.amax(A.shape)))
        CLD = []
        numvecs = len(paths)

        for k in range(0, numvecs):  # kth path out of numvecs paths
            path = np.array(paths[k])

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            stdout.write("\r%d out of %d paths, %d%% done, local time: %s"
                         % (k, numvecs, k/numvecs*100, current_time))

            dy = (path[-1, 0]-path[0, 0])
            dx = (path[-1, 1]-path[0, 1])

            assert abs(dx) == col-1 or abs(dy) == row-1

            if abs(dx) == col-1:
                axis = 0
            elif abs(dy) == row-1:
                axis = 1

            transnum = shape[axis]

            # the path is to be translated transnum times
            for i in range(0, transnum):

                path = np.array(paths[k])
                path[:, axis] = path[:, axis]+i
                numi, _ = path.shape
                count = 1

                for j in range(1, numi):
                    # if chord continues to the next pixel
                    if (A[path[j-1, 0], path[j-1, 1]]
                            == A[path[j, 0], path[j, 1]]):
                        count += 1
                    else:
                        # if chord ends in the previous pixel and this pixel
                        # begins the new chord
                        if A[path[j-1, 0], path[j-1, 1]] == phase:
                            chords[axis, count] += 1
                        count = 1

                if A[path[j, 0], path[j, 1]] == phase:
                    if count == numi:
                        j = count
                    chords[axis, count] += 1
                    count = 1

            total_num_chords = np.sum(chords[k])
            CLD.append(chords[k]/total_num_chords)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        stdout.write("\r%d out of %d paths, %d%% done, local time: %s \n"
                     % (k+1, numvecs, (k+1)/numvecs*100, current_time))

    elif A.ndim == 3:
        paths = []
        # horizontal path (path parallel to axis 0):
        paths.append(Bresenham3D(0, 0, 0, A.shape[0]-1, 0, 0))
        # vertical path (path parallel to axis 1):
        paths.append(Bresenham3D(0, 0, 0, 0, A.shape[1]-1, 0))
        # horizontal path (path parallel to axis 2):
        paths.append(Bresenham3D(0, 0, 0, 0, 0, A.shape[2]-1))

        chords = []
        CLD = []
        numvecs = len(paths)

        for k in range(0, numvecs):  # kth path out of numvecs paths
            chords.append(np.zeros(len(paths[k])))
            path = np.array(paths[k])

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            stdout.write("\r%d out of %d paths, %d%% done, local time: %s"
                         % (k, numvecs, k/numvecs*100, current_time))

            it = list({0, 1, 2} - {k})

            for i in range(0, A.shape[it[0]]):
                for ii in range(0, A.shape[it[1]]):
                    path = np.array(paths[k])
                    path[:, it[0]] = i
                    path[:, it[1]] = ii
                    count = 1

                    for j in range(1, path.shape[0]):
                        # if chord continues to the next pixel:
                        if (A[path[j-1, 0], path[j-1, 1], path[j-1, 2]]
                                == A[path[j, 0], path[j, 1], path[j, 2]]):
                            count += 1
                        else:
                            # if chord ends in the previous pixel and this
                            # pixel begins the new chord:
                            if A[path[j-1, 0],
                                 path[j-1, 1],
                                 path[j-1, 2]] == phase:
                                chords[k][count-1] += 1
                            count = 1

                    if A[path[j, 0], path[j, 1], path[j, 2]] == phase:
                        if count == path.shape[0]:
                            j = count
                        chords[k][count-1] += 1
                        count = 1

            total_num_chords = np.sum(chords[k])
            CLD.append(chords[k]/total_num_chords)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        stdout.write("\r%d out of %d paths, %d%% done, local time: %s \n"
                     % (k+1, numvecs, (k+1)/numvecs*100, current_time))

    return CLD

##############################################################################


def particle_quantification(A, all_methods=False, printPhaseVals=False):
    """
    Specific surface, volume, and diameter of equivalent sphere volume.
    
    This function returns quantification of particles. The algorithm assigns 
    an ID number for each contactless particle and for these particles, it 
    evaluates: a specific surface by a stereological approach, extrapolation,
    and differentiation of the two-point probability function; a volume 
    of voxels, and a diameter of an equivalent sphere volume.    
    Parameters
    ----------
    A : Boolean NumPy 3D array
        Input array of the medium; True represents the selected phase.
    Returns
    -------
    out : Numpy 2D array 
          Output array with number of rows equal to the number of contactless
          particles and 6 columns. The columns represent: ID number of the 
          particle, specific surface by a stereological approach, specific
          surface by an extrapolation, specific surface by differentiation of
          the two-point probability function, volume, and diameter of an
          equivalent sphere volume.
    """

    dim = A.ndim
    struct = spim.generate_binary_structure(dim, dim)
    
    particles, num_particles = spim.label(A, struct)
    
    out = np.zeros((num_particles,6))
    
    print("total number of particles: {}".format(num_particles))
    
    # scan all particles and omit any particle with contact with medium edges
    for i in range(num_particles):
        
        particle = particles == i+1
        out[i,0] = i

        # specific surface by a stereological approach        
        if all_methods == True:
            out[i,1] = real_surface_stereological_approach(particle)*np.prod(A.shape)
        
        # specific surface by an extrapolation
        out[i,2] = real_surface_extrapolation(particle, max_iter=5)
       
        # compensate for a possible negative surface area obtained for small particles:
        if out[i,2] <= 0: 
            out[i,2] = (real_surface_stereological_approach(particle)*np.prod(A.shape) + 
                        real_surface_differentiation_S2(particle)*np.prod(A.shape))/2

        # specific surface by differentiation of the two-point probability 
        # function        
        if all_methods == True:
            out[i,3] = real_surface_differentiation_S2(particle)*np.prod(A.shape)
        
        # volume
        out[i,4] = np.sum(particle)
        
        # diameter of an equivalent sphere volume 
        out[i,5] = (out[i,4]*6/np.pi)**(1/3)

        if printPhaseVals:
            if all_methods == True:
                print("{} {} {} {} {} {}".format(out[i,0],out[i,1],out[i,2],out[i,3],out[i,4],out[i,5]))
            else:
                print("{} {} {} {}".format(out[i,0],out[i,2],out[i,4],out[i,5]))
    
    if all_methods == False:           
        out = np.delete(out,[1,3],axis=1)  
    
    return out

###############################################################################


def remove_edge_particles_clusters(A):
    """
    Remove all particles that are connected to any edge of the media.
    Parameters
    ----------
    A : Boolean NumPy 3D array
        Input array of the medium; True represents the selected phase.
    Returns
    -------
    out : Boolean NumPy 3D array
          A cleaned from all particles touching any edge of the media.
    """
    dim = A.ndim

    struct = spim.generate_binary_structure(dim, dim)
    A_clstrs, num_clstrs = spim.label(A, struct)
    out = copy.deepcopy(A)
    
    for i in range(1,num_clstrs+1):
        x,y,z = (A_clstrs == i).nonzero()
        
        if (any(x==0) or any(y==0) or any(z==0) or any(x==A.shape[0]-1) or 
            any(y==A.shape[1]-1) or any(z==A.shape[2]-1)):
            
            out[A_clstrs == i] = False
               
    return out

###############################################################################

    
def enlarged_array(origMatrix,thickness=5):
    """
    Inscribe the original array into a larger array.
    
    The original array has size of W-by-H-by-D, the new array is 2*thickness
    larger to each dimension. thickness from outside to inside is always
    empty.
    Parameters
    ----------
    origMatrix : NumPy 3D array
        Input array of the medium.
    thickness : int
        Number of outside empty layers in the new array.
    Returns
    -------
    enlargedMatrix : NumPy 3D array
          Enlarged original array.
    """
    
    enlargedMatrix=np.zeros((origMatrix.shape[0]+2*thickness, 
                             origMatrix.shape[1]+2*thickness,
                             origMatrix.shape[2]+2*thickness)) 

    enlargedMatrix[thickness:-thickness,
                   thickness:-thickness,
                   thickness:-thickness]=origMatrix 
    
    return enlargedMatrix

###############################################################################


def export2gnuplot(filename='', fileDescription='', storeArrays=(), 
                   colDescription='', colFormat='%.3f'):
    """
    Save numpy arrays as columns into a textfile for plotting in gnuplot, 
    add header with detailed description.
    """
    import time,sys
    fullName= 'Data4plot_'+filename+'.txt'
    file1=open(fullName, 'w' )      
    file1.write('# File generated on '+time.strftime("%d.%m.%Y at %H:%M")+
                ' using '+str(sys.argv[0])+'\n' )
    file1.write('# File contains: %s \n' %(fileDescription))
    file1.write('#\n' )
    file1.write('# %s \n' % (colDescription))
    file1.write('##########################################################\n')
    np.savetxt(file1, np.column_stack(storeArrays), fmt=colFormat, 
               delimiter='\t')
    file1.close()
    print('\n-> Data saved to file: "'+fullName+'"<-')
    
    return 0