from distutils.core import setup, Extension
from distutils.command.build import build
import os
import fnmatch
import sys
import re
import subprocess


def find_package_data_files(directory):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, '*'):
                filename = os.path.join(root, basename)
                yield filename.replace('cf/', '', 1)

def find_test_files(directory):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if (fnmatch.fnmatch(basename, '*.sh') or
                fnmatch.fnmatch(basename, '*.nc') or
                fnmatch.fnmatch(basename, '*.pp')):
                filename = os.path.join(root, basename)
                yield filename.replace('cf/', '', 1)

def _read(fname):
    '''Returns content of a file.

    '''
    fpath = os.path.dirname(__file__)
    fpath = os.path.join(fpath, fname)
    with open(fpath, 'r') as file_:
        return file_.read()

def _get_version():
    '''Returns library version by inspecting __init__.py file.

    '''
    return re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                     _read("cf/__init__.py"),
                     re.MULTILINE).group(1)

version      = _get_version()
packages     = ['cf']
etc_files    = [f for f in find_package_data_files('cf/etc')]
umread_files = [f for f in find_package_data_files('cf/umread_lib/c-lib')]
test_files   = [f for f in find_test_files('cf/test')]

package_data = etc_files + umread_files + test_files

class build_umread(build):
    '''Adpated from
    https://github.com/Turbo87/py-xcsoar/blob/master/setup.py

    '''
    def run(self):
        # Run original build code
        build.run(self)

        # Build umread
        print('Running build_umread')

        build_dir = os.path.join(os.path.abspath(self.build_lib), 'cf/umread_lib/c-lib')

        cmd = ['make', '-C', build_dir]

#        rc = subprocess.call(cmd)

        def compile():
            print('*' * 80)
            print('Running:', ' '.join(cmd), '\n')

            try:
                rc = subprocess.call(cmd)
            except Exception as error:
                print(error)
                rc = 40

            print('\n', '-' * 80)
            if not rc:
                print('SUCCESSFULLY built UM read C library')
            else:
                print('WARNING: Failed to build the UM read C library.')
                print('         Attempting to read UKMO PP and UM format files will result in failure.')
                print('         This will not affect any other cf functionality.')
                print('         In particular, netCDF file processing is unaffected.')

            print('-' * 80)
            print('\n', '*' * 80)
            print()
            print("cf build successful")
            print()


        self.execute(compile, [], 'compiling umread')
#--- End: class


long_description="""
CF Python
=========

The Python cf package is an Earth science data analysis library that
is built on a complete implementation of the CF data model


Documentation
=============

http://ncas-cms.github.io/cf-python

Tutorial
========

https://ncas-cms.github.io/cf-python/tutorial

Installation
============

http://ncas-cms.github.io/cf-python/installation


Functionality
=============

The `cf` package implements the CF data model
(https://doi.org/10.5194/gmd-10-4619-2017) for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets. This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The `cf` package can:

* read field constructs from netCDF, PP and UM datasets,

* create new field constructs in memory,

* inspect field constructs,

* test whether two field constructs are the same,

* modify field construct metadata and data,

* create subspaces of field constructs,

* write field constructs to netCDF datasets on disk,

* incorporate, and create, metadata stored in external files,

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays), whilst presenting a view of the
  data in its uncompressed form,

* read, write, and create coordinates defined by geometry cells,

* combine field constructs arithmetically,

* manipulate field construct data by arithmetical and trigonometrical
  operations,

* perform statistical collapses on field constructs,

* perform histogram, percentile and binning operations on field
  constructs,

* regrid field constructs,

* apply convolution filters and moving means to field constructs,

* calculate derivatives of field constructs,

* create field constructs to create derived quantities (such as
  vorticity).


Visualization
=============

Powerful, flexible, and very simple to produce visualizations of field
constructs are available with the `cfplot` package
(http://ajheaps.github.io/cf-plot), that needs to be installed
seprately to the `cf` package.

See the `cfplot` gallery (http://ajheaps.github.io/cf-plot/gallery.html)
for the full range range plotting possibilities with example code.


Command line utilities
======================

During installation the ``cfa`` command line utility is also
installed, which

* generates text descriptions of field constructs contained in files,
  and

* creates new datasets aggregated from existing files.


Tests
=====

Tests are run from within the ``cf/test`` directory:

    python run_tests.py
"""

setup(name = "cf-python",
      long_description = long_description,
      version      = version,
      description  = "A CF-compliant earth science data analysis library",
      author       = "David Hassell",
      maintainer   = "David Hassell",
      maintainer_email  = "david.hassell@ncas.ac.uk",
      author_email = "david.hassell@ncas.ac.uk",
      url          = "https://ncas-cms.github.io/cf-python",
      platforms    = ["Linux", "MacOS"],
      keywords     = ['cf', 'netcdf', 'UM', 'data', 'science',
                      'oceanography', 'meteorology', 'climate'],
      classifiers  = ["Development Status :: 5 - Production/Stable",
                      "Intended Audience :: Science/Research",
                      "License :: OSI Approved :: MIT License",
                      "Topic :: Scientific/Engineering :: Mathematics",
                      "Topic :: Scientific/Engineering :: Physics",
                      "Topic :: Scientific/Engineering :: Atmospheric Science",
                      "Topic :: Utilities",
                      "Operating System :: POSIX :: Linux",
                      "Operating System :: MacOS"
                      ],
      packages     = ['cf',
                      'cf.abstract',
                      'cf.mixin',
                      'cf.data',
                      'cf.data.abstract',
                      'cf.read_write',
                      'cf.read_write.um',
                      'cf.read_write.netcdf',
                      'cf.umread_lib',
                      'cf.test'
                     ],
      package_data = {'cf': package_data},
      scripts      = ['scripts/cfa'],
      install_requires = ['netCDF4>=1.5.3',
                          'cftime>=1.1.3',
                          'numpy>=1.15',
                          'cfdm>=1.8.5, <1.9',
                          'psutil>=0.6.0',
                          'cfunits>=3.2.7'
#                          'scipy>=1.1.0',
#                          'matplotlib>=3.0.0',
#                          'mpi4py>=3.0.0',
#                          'ESMF>=8.0',
#                          'udunits2==2.2.25',
                      ],
      cmdclass     = {'build': build_umread}, #https://docs.python.org/2/distutils/apiref.html
  )
