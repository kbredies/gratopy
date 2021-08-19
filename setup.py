from setuptools import setup

name='gratopy'
version='0.1.0rc0'

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(name=name,
      version=version,
      description='The Graz accelerated tomographic projection for python (Gratopy) is a software tool developed to allow for efficient, high quality execution of projection methods such as Radon and fanbeam transform.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kbredies/gratopy',
      author='Kristian Bredies, Richard Huber',
      author_email='richard.huber@uni-graz.at',
      license='GPLv3',
      keywords='pixel-driven projection methods tomography pyopencl',
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
