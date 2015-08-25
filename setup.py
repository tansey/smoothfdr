from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smoothfdr',
    version='0.9.1',
    description='False discovery rate smoothing',
    long_description=long_description,
    url='https://github.com/tansey/smoothfdr',
    author='Wesley Tansey',
    author_email='tansey@cs.utexas.edu',
    license='MIT',

    classifiers=[
        # Will be upgraded to 4 - Beta after full fMRI debugging.
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='statistics biostatistics fdr hypothesis machinelearning',

    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pygfl'],
    package_data={
        'smoothfdr': [],
    },
    entry_points={
        'console_scripts': [
            'smoothfdr=smoothfdr:main',
            'neuropre=neuropre:main',
            'neuropost=neuropost:main',
        ],
    },
)