**DEVELOPMENT VERSION FOR PYTHON3 - LIABLE TO CHANGE WITHOUT NOTICE**

**STABLE RELEASE DATE: 2019-10-01**


The Python cf package is an Earth science data analysis library that
is built on a complete implementation of the CF data model

# CF Python

The Python cf package is an Earth science data analysis library that
is built on a complete implementation of the CF data model

# Documentation

http://ncas-cms.github.io/cf-python

# Installation

http://ncas-cms.github.io/cf-python/installation.html

# Functionality

The cf package implements the CF data model [1] for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets. This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The cf package can:

  * read field constructs from netCDF, PP and UM datasets,
  
  * create new field constructs in memory,
  
  * inspect field constructs,
  
  * test whether two field constructs are the same,
  
  * modify field construct metadata and data,
  
  * create subspaces of field constructs,
  
  * write field constructs to netCDF datasets on disk,
  
  * incorporate, and create, metadata stored in external files,
  
  * read, write, and create data that have been compressed by
    convention (i.e. ragged or gathered arrays), whilst presenting a
    view of the data in its uncompressed form,    
  
  * Combine field constructs arithmetically,
  
  * Manipulate field construct data by arithmetical and
    trigonometrical operations,
  
  * Perform statistical collapses on field constructs,
  
  * Regrid field constructs,
  
  * Apply convolution filters to field constructs,
  
  * Calculate derivatives of field constructs,
  
  * Create field constructs to create derived quantities (such as
    vorticity).

# Visualization

Powerful, flexible, and very simple to produce visualizations of field
constructs are available with the cfplot package
http(://ajheaps.github.io/cf-plot), that needs to be installed
seprately to cf.

See the cfplot gallery (http://ajheaps.github.io/cf-plot/gallery.html)
for the full range range plotting possibilities with example code.

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

    cfa -h

# Tests

Tests are run from within the cf-python/test directory:

   $ python run_tests.py

# Code license

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
