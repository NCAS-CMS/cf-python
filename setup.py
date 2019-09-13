from distutils.core import setup, Extension
from distutils.command.build import build
import os
import fnmatch
import sys
import imp
import re
import subprocess

def find_package_data_files(directory):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, '*'):
                filename = os.path.join(root, basename)
                yield filename.replace('cf/', '', 1)

def find_test_data_files(directory):
    print ('dir=', directory)
    for root, dirs, files in os.walk(directory):        
        for basename in files:
            if fnmatch.fnmatch(basename, '*.nc') or fnmatch.fnmatch(basename, '*.pp'):
                print ('basename=', basename)
                filename = os.path.join(root, basename)
                yield filename.replace('cf/', '', 1)
#--- End: def

def _read(fname):
    """Returns content of a file.

    """
    fpath = os.path.dirname(__file__)
    fpath = os.path.join(fpath, fname)
    with open(fpath, 'r') as file_:
        return file_.read()

def _get_version():
    """Returns library version by inspecting __init__.py file.

    """
    return re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                     _read("cf/__init__.py"),
                     re.MULTILINE).group(1)



version      = _get_version()
packages     = ['cf']
etc_files    = [f for f in find_package_data_files('cf/etc')]
umread_files = [f for f in find_package_data_files('cf/umread_lib/c-lib')]
test_files   = [f for f in find_test_data_files('cf/test')]

package_data = etc_files + umread_files + test_files

class build_umread(build):
    '''
Adpated from https://github.com/Turbo87/py-xcsoar/blob/master/setup.py
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
                print('         This will not affect any other cf-python functionality.')
                print('         In particular, netCDF file processing is unaffected.')

            print('-' * 80)
            print('\n', '*' * 80)
            print()
            print("cf build successful")
            print()
        #--- End: def

        self.execute(compile, [], 'compiling umread')
#--- End: class

long_description = """
* **Note:** There have been some API changes between version 1.x and
  2.x of cf-python. Version 1.x will not get any new functionality,
  but will be patched for the foreseeable future. The latest 1.x
  release is `version 1.5.4.post6
  <https://pypi.python.org/pypi/cf-python/1.5.4.post6>`_

Home page
=========

* `cf-python <http://cfpython.bitbucket.io>`_

Documentation
=============

* `Online documentation for the latest release
  <http://cfpython.bitbucket.io/docs/latest/>`_

Dependencies
============

* The package runs on Linux and Mac OS operating systems.

* Requires Python version 3.5 or later
 
* Requires cfdm version 1.7.5 or later

* Requires numpy version 1.15 or later

* Requires psutil version 0.6.0 or later

* Requires netCDF4 version 1.4.0 or later

Visualisation
=============

* The `cfplot package <https://pypi.python.org/pypi/cf-plot>`_ does
  not currently work for versions 2.x of cf-python (it does work for
  versions 1.x). This will be resolved soon.


Command line utilities
======================

* The `cfdump` tool generates text representations on standard output
  of the CF fields contained in the input files.

* The `cfa` tool creates and writes to disk the CF fields contained in
  the input files.

* During installation these scripts will be copied automatically to a
  location given by the ``PATH`` environment variable.

Code license
============

* `MIT License <http://opensource.org/licenses/mit-license.php>`_"""

setup(name = "cf-python",
      long_description = long_description,
      version      = version,
      description  = "Python interface to the CF conventions",
      author       = "David Hassell",
      maintainer   = "David Hassell",
      maintainer_email  = "david.hassell@ncas.ac.uk",
      author_email = "david.hassell@ncas.ac.uk",
      url          = "http://cfpython.bitbucket.io/",
      platforms    = ["Linux", "MacOS"],
      keywords     = ['cf','netcdf','data','science',
                      'oceanography','meteorology','climate'],
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
                      'cf.test',                      
      ],
      package_data = {'cf': package_data},
      scripts      = ['scripts/cfa'],
      install_requires = ['netCDF4>=1.4.0',                        
                          'numpy>=1.15',
                          'cfdm>=1.7.5',
                          'psutil>=0.6.0',
                          'cfunits>=3.2.0',
                      ],
      cmdclass     = {'build': build_umread}, #https://docs.python.org/2/distutils/apiref.html
  )

