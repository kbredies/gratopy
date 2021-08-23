from setuptools import setup

name='gratopy'
version='0.1.0'

long_description="""
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.5221442.svg)](https://doi.org/10.5281/zenodo.5221442)
[![Documentation Status](https://readthedocs.org/projects/gratopy/badge/?version=latest)](https://gratopy.readthedocs.io/?badge=latest)

The gratopy (**Gr**az **a**ccelerated **to**mographic projections for **Py**thon) toolbox is a Python3 software package for the efficient and high-quality computation of Radon transforms, fanbeam transforms as well as the associated backprojections. The included operators are based on pixel-driven projection methods which were shown to possess [favorable approximation properties](https://epubs.siam.org/doi/abs/10.1137/20M1326635). The toolbox offers a powerful parallel OpenCL/GPU implementation which admits high execution speed and allows for seamless integration into [PyOpenCL](https://documen.tician.de/pyopencl/). Gratopy can efficiently be combined with other PyOpenCL code and is well-suited for the development of iterative tomographic reconstruction approaches, in particular, for those involving optimization algorithms.

## Highlights
* Easy-to-use tomographic projection toolbox.
* High-quality 2D projection operators.
* Fast projection due to custom OpenCL/GPU implementation.
* Seamless integration into PyOpenCL.
* Basic iterative reconstruction schemes included (Landweber, CG, total variation).
* Comprehensive documentation, tests and example code.

See the [documentation](https://gratopy.readthedocs.io/) and the project's [GitHub page](https://github.com/kbredies/gratopy) for installation, usage, updates and further information.
"""

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(name=name,
      version=version,
      description='Gratopy - Graz accelerated tomographic projections for Python',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kbredies/gratopy',
      project_urls={ "Documentation": "https://gratopy.readthedocs.io/"},
      author='Kristian Bredies, Richard Huber',
      classifiers=[
          "Environment :: Console",
          "Environment :: GPU",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Other Audience",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Programming Language :: Python :: 3",
          "Programming Language :: C",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Image Processing",
      ],
      author_email='kristian.bredies@uni-graz.at',
      license='GPLv3',
      keywords='Radon transform, fanbeam transform, pixel-driven projection methods, computed tomography, image reconstruction, pyopencl',
      packages=['gratopy'],
      package_data={'': ['gratopy/*.cl']},
      include_package_data=True,
      python_requires=">=3.6",
      install_requires=[requirements.split()],
      zip_safe=False,
      command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'source_dir': ('setup.py', 'doc/source')}})
