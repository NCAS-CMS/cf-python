**DEVELOPMENT VERSION FOR PYTHON3 - LIABLE TO CHANGE WITHOUT NOTICE**

**STABLE RELEASE DATE: 2019-10-01**


The Python cf package is an Earth science data analysis library that
is built on a complete implementation of the CF data model

CF Python
=========

The cf python package implements the CF data model for the reading,
writing and processing of data and metadata.

# Functionality

The cf package implements the CF data model [1] for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets.This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The cf package can:

    read field constructs from netCDF, PP and UM datasets,
    create new field constructs in memory,
    inspect field constructs,
    test whether two field constructs are the same,
    modify field construct metadata and data,
    create subspaces of field constructs,
    write field constructs to netCDF datasets on disk,
    incorporate, and create, metadata stored in external files,
    read, write, and create data that have been compressed by convention (i.e. ragged or gathered arrays), whilst presenting a view of the data in its uncompressed form,
    Combine field constructs arithmetically,
    Manipulate field construct data by arithmetical and trigonometrical operations,
    Perform statistical collapses on field constructs,
    Regrid field constructs,
    Apply convolution filters to field constructs,
    Calculate derivatives of field constructs,
    Create field constructs to create derived quantities (such as vorticity).

# Visualization

Powerful, flexible, and very simple to produce visualizations of field constructs are available with the cfplot package (that needs to be installed seprately to cf).

See the cfplot gallery for the full range range plotting possibilities with example code.

This branch implements Python3
----------------------------------------------------------------------

Home page
=========

[**cf-python**](http://cfpython.bitbucket.io)

----------------------------------------------------------------------

Documentation
=============

* [**Online documentation for the latest stable
  release**](http://cfpython.bitbucket.io/docs/latest/ "cf-python
  documentation")

* Online documentation for previous releases: [**cf-python documention
  archive**](http://cfpython.bitbucket.io/docs/archive.html)

* Offline documention for the installed version may be found by
  pointing a browser to ``docs/build/index.html``.

* [**Change log**](https://bitbucket.org/cfpython/cf-python/src/master/Changelog.md)

----------------------------------------------------------------------

Dependencies
============

* **Required:** A
  [**GNU/Linux**](http://www.gnu.org/gnu/linux-and-gnu.html) or [**Mac
  OS**](http://en.wikipedia.org/wiki/Mac_OS) operating system.

* **Required:** A [**python**](http://www.python.org) version 3 or later.
 
* **Required:** The [**python psutil
  package**](https://pypi.python.org/pypi/psutil) at version 0.6.0 or
  newer (the latest version is recommended).

* **Required:** The [**python numpy
  package**](https://pypi.python.org/pypi/numpy) at version 1.15 or
  newer.

* **Required:** The [**python matplotlib
  package**](https://pypi.python.org/pypi/matplotlib) at version 1.4.2
  or newer.

* **Required:** The [**python scipy
  package**](https://pypi.python.org/pypi/scipy) at version 0.14.0 or
  newer.

* **Required:** The [**python netCDF4
  package**](https://pypi.python.org/pypi/netCDF4) at version 1.4.0 or
  newer. This package requires the
  [**netCDF**](http://www.unidata.ucar.edu/software/netcdf),
  [**HDF5**](http://www.hdfgroup.org/HDF5) and
  [**zlib**](ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4)
  libraries.

* **Required:** The [**CF units
  package**](https://pypi.python.org/pypi/cfunits) at version 3.0.0 or
  newer.

* **Optional:** For regridding to work, the [**ESMF
  package**](https://www.earthsystemcog.org/projects/esmf) at version
  7.0.0 or newer is required. If this package is not installed then
  regridding will not work, but all other cf-python functionality will
  be unaffected. ESMF may be installed via
  [**conda**](http://conda.pydata.org/docs) (see below) or from source
  (see the file [**ESMF.md**](ESMF.md) for instructions).

* **Optional:** The [**cf-plot
  package**](https://pypi.python.org/pypi/cf-plot) does not currently
  work for versions 2.x of cf. This will be resolved soon.

----------------------------------------------------------------------

Installation
============

* To install by [**conda**](http://conda.pydata.org/docs):

        conda install -c ncas -c conda-forge cf-python udunits2=2.2.20

    These two commands will install cf-python, all of its required
    dependencies and the two optional packages cf-plot (for
    visualisation) and ESMF (for regridding). To install cf-python and
    all of its required dependencies alone:

        conda install -c ncas -c conda-forge cf-python cf-plot udunits2=2.2.20

    To update cf-python, cf-plot and ESMF to the latest versions::

        conda install -c ncas -c conda-forge cf-python cf-plot udunits2=2.2.20
        conda install -c conda-forge mpich esmpy
	
* To install the **latest** version from
  [**PyPI**](https://pypi.python.org/pypi/cf-python):

        pip install cf-python

* To install from source:

   Download the cf package from
   `<https://pypi.python.org/pypi/cf-python>`_
   
   Unpack the library (replacing <version> with the approrpriate release,
   e.g. ``2.3.3``):
   
      $ tar zxvf cf-python-<version>.tar.gz
      $ cd cf-python-<version>
   
   To install the cf package to a central location:
   
      $ python setup.py install
   
   To install the cf package locally to the user in the default location
   (often within ``~/.local``):
   
      $ python setup.py install --user
   
   To install the cf package in the <directory> of your choice::

      $ python setup.py install --home=<directory>

----------------------------------------------------------------------

Tests
=====

The test scripts are in the ``test`` directory. To run all tests:

    python test/run_tests.py


----------------------------------------------------------------------

Command line utilities
======================

The ``cfdump`` tool generates text representations on standard output
of the CF fields contained in the input files. 

The ``cfa`` tool creates and writes to disk the CF fields contained in
the input files.

During the installation described above, these scripts will be copied
automatically to a location given by the ``PATH`` environment
variable.

For usage instructions, use the ``-h`` option to display the manual
pages:

    cfdump -h
    cfa -h

----------------------------------------------------------------------

Code license
============

[**MIT License**](http://opensource.org/licenses/mit-license.php)

  * Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

  * The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
