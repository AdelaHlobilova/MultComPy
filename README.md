# MultComPy
Library containing statistical descriptors for assessment of digitized images of multiphase composite materials

MultComPy library contains a set of statistical descriptors (i.e. n-point correlation functions) used to assess and evaluate digitized images of multiphase composite materials. Unlike other libraries available (PoreSpy), the main focus rests on delivering a concise collection of functions that are applicable to evaluate multiphase digitized images obtained e.g. from computed tomography, rather than on generating artificial images or altering the input images. As for its primary application, this library was initially developed for the assessment of the spatial configuration of the various solid phases occurring within a hydrating cement paste microstructure.

MultComPy relies on additional Python libraries like Numpy and SciPy, however, the main functions are implemented directly and optimized for evaluation speed. The most time-demanding functions are concurrently implemented both in Python and in C and allow for a mutual comparison.

## Overview of implemented functions:
- **two-point probability function** - allows to evaluate auto-/cross-correlation using brute force or discrete Fourier transform methods for ND media

- **two-point cluster function** - allows to evaluate the probability, that two randomly chosen points (pixels or voxels) x1 and x2 belong to the same cluster of the selected phase; method implemented using discrete Fourier transformation for ND media

- **chord length density** - allows evaluating the isotropy of the material by characterizing orthogonal path length along with the principal directions for 2D and 3D media

- **lineal path** - allows to evaluate the probability that a randomly-thrown line segment lies in the same phase; implemented by brute force method in both Python and C for 2D and 3D media

- **real surface of particles** - allows quantifying the area (voxel count) of selected inclusion by stereological approach, extrapolation, and differentiation of the two-point probability function for ND media

- **shortest distance from hydrate to the clinker surface** for 2D and 3D media

- **particle quantification** - evaluates the specific surface, volume, and an equivalent spherical diameter for a selected phase


<!-- Helper functions:
- my_shift (shifts the zero-frequency component to the centre of the spectrum)
- transform_ND_to_1D (transforms matrix representation of the statistical descriptor into a vector representation)
- BW_morph_remove (removes interior pixels to leave an outline of the shapes)
- find_edges (edge detection of the input ND array using scipy.ndimage)
- BresLineAlg (2D Bressenham's line algorithm)
- Bresenham3D (Bresenham’s Algorithm for 3-D Line Drawing)
- L2_generate_paths (generates paths via Bressenham's 2-D line algorithm)
- L2_generate_paths_3D (generates paths via Bressenham's line 3-D algorithm)
- remove_edge_particles_clusters (removes all particles that are connected to any edge of the media)
- enlarged_array (inscribe the original array into a larger array)
- export2gnuplot (saves numpy arrays as columns into a textfile for plotting in gnuplot, 
    adds header with detailed description) -->
