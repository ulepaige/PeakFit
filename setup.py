from setuptools import find_packages, setup

import versioneer

with open('Readme.md') as f:
    long_description = f.read()

setup(
    name='PeakFit',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='PeakFit allow for lineshape fitting in pseudo-3D NMR spectra.',
    long_description=long_description,
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    author='Guillaume Bouvignies',
    author_email='gbouvignies@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'nmrglue', 'lmfit'],
    entry_points={'console_scripts': ['peakfit = peakfit.__main__:main', ]})
