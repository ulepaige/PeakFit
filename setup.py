from setuptools import find_packages, setup

import versioneer

with open("Readme.md") as f:
    long_description = f.read()

setup(
    name="PeakFit",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="PeakFit allow for lineshape fitting in pseudo-3D NMR spectra.",
    long_description=long_description,
    license="BSD 3-Clause",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    author="Guillaume Bouvignies",
    author_email="gbouvignies@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0",
        "natsort>=2.0",
        "nmrglue",
        "lmfit>=0.9.11",
        "asteval>=0.9.11",
    ],
    python_requires=">=3.5",
    entry_points={
        "console_scripts": [
            "peakfit = peakfit.peakfit:main",
            "plot_cpmg = peakfit.plot_cpmg:main",
            "plot_cest = peakfit.plot_cest:main",
        ]
    },
)
