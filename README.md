# SphereDiff

## Purpose

The purpose of this package is to sample from the isotropic diffusion kernel 
on the surface of an n-dimensional unit sphere, where n is an arbitrary 
natural number dimension greater than or equal to 3.

## Installation

### From a repository checkout

```bash
pip install --user .
```

### From PyPI

```bash
pip install --user spherediff
```

## Use

The user may sample from the isotropic Gaussian distribution on the unit 
n-sphere using the `sample_spherical_kernel` function, which may be 
imported as follows:

```
>> from spherediff.sample import sample_spherical_kernel
```

This function takes three arguments and one additional, optional argument. 
The first is n, the dimension of the space in which the n-sphere is embedded. 
The second is a numpy array of shape (N, n) consisting of the n-dimensional 
unit vectors at which to center the distributions from which the samples are 
to be generated. The third is a numpy array of shape (N,) consisting of the 
scalar variance parameters of each distribution from which to generate 
samples. The fourth is a boolean flag that determines whether sampling should 
be done on the full surface of the n-sphere (if False) or on the hemisphere 
with reflecting boundary conditions for the diffusion kernel.

Example output from `sample_spherical_kernel` is:

```
>>> import numpy as np
>>> from spherediff.sample import sample_spherical_kernel
>>> np.random.seed(42)
>>> means = np.random.randn(5, 3)
>>> means /= np.linalg.norm(means, axis=1, keepdims=True)
>>> means
array([[ 0.60000205, -0.1670153 ,  0.78237039],
       [ 0.97717133, -0.15023209, -0.15022156],
       [ 0.86889694,  0.42224942, -0.25830898],
       [ 0.63675162, -0.5438697 , -0.54658314],
       [ 0.09351637, -0.73946664, -0.66666616]])
>>> vars = 0.1 * np.ones(5)
>>> sample_spherical_kernel(3, means, vars)
array([[ 0.30027556, -0.53104481,  0.79235472],
       [ 0.91657116, -0.39288942,  0.07439905],
       [ 0.81325411,  0.41495422, -0.40795926],
       [ 0.39907791, -0.44171124, -0.80350981],
       [ 0.16422958, -0.76019121, -0.62860001]])
>>> sample_spherical_kernel(3, means, vars, hemisphere=True)
array([[ 0.92723597,  0.02336567,  0.37374791],
       [ 0.99421791, -0.03878944, -0.10013055],
       [ 0.15771025,  0.6492883 , -0.74401087],
       [ 0.2418101 , -0.41127436, -0.87885225],
       [-0.11192408, -0.71437847, -0.69075061]])
```
