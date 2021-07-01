from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gratopy',
      version='0.1.0.dev',
      description='The Graz accelerated tomographic projection for python (Gratopy) is a software tool developed to allow for efficient, high quality execution of projection methods such as Radon and fanbeam transform.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kbredies/gratopy',
      author='Kristian Bredes, Richard Huber',
      author_email='richard.huber@uni-graz.at',
      license='GNU GENERAL PUBLIC LICENSE',
      keywords='pixel-driven projectionmethods tomography pyopencl',
      packages=['gratopy'],
      python_requires=">=3.6",
      install_requires=[
	'wheel',
	'appdirs>=1.4.4',
        'cycler>=0.10.0',
        'dataclasses>=0.8',
        'kiwisolver>=1.3.1',
        'Mako>=1.1.4',
        'MarkupSafe>=2.0.1',
        'matplotlib>=3.3.4',
        'numpy>=1.19.5',
        'Pillow>=8.2.0',
        'pkg-resources>=0.0.0',
        'pybind11>=2.6.2',
        'pyparsing>=2.4.7',
        'python-dateutil>=2.8.1',
        'pytools>=2021.2.7',
        'scipy>=1.5.4',
        'six>=1.16.0',
        'pyopencl>=2021.2.2'],
      zip_safe=False)
