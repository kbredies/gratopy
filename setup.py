from setuptools import setup

name='gratopy'
version='0.1.0rc1'

long_description="""
[![Documentation Status](https://readthedocs.org/projects/gratopy/badge/?version=latest)](https://gratopy.readthedocs.io/?badge=latest)

The **Gr**az **a**ccelerated **to**mographic projection for **Py**thon **(Gratopy)**  is a software tool for Python3 developed to allow for efficient, high quality execution of projection methods
such as Radon and fanbeam transform.  The operations contained in the toolbox are based on pixel-driven projection methods, which were shown to possess suitable approximation properties.
The code is based in a powerful OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into [PyOpenCL](https://documen.tician.de/pyopencl/).
Hence this can efficiently be paired with other PyOpenCL code, in particular OpenCL based optimization algorithms.

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
      description='The Graz accelerated tomographic projection for Python (Gratopy) is a software tool developed to allow for efficient, high quality execution of projection methods such as Radon and fanbeam transform.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kbredies/gratopy',
      project_urls={ "Documentation": "https://gratopy.readthedocs.io/"}
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
