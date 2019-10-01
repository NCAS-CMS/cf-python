.. currentmodule:: cf
.. default-role:: obj

.. _Installation:

**Installation**
================

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry

.. _Operating-systems:

**Operating systems**
---------------------

cf works only for Linux and Mac operating systems.

----

.. _Python-versions:

**Python versions**
-------------------

cf works only for Python 3.5 or later. (Versions 2.x of cf work only
for Python 2.7.)

----

.. _conda:
  
**conda**
---------

cf-python is in the ``ncas`` conda channel. To install cf with all of
its :ref:`required <Required>` and :ref:`optional <Optional>`
dependencies, and the `cf-plot visualisation package
<http://ajheaps.github.io/cf-plot>`_, run

.. code-block:: shell
   :caption: *Install with conda.*
	     
   conda install -c ncas -c conda-forge cf-python cf-plot udunits2
   conda install -c conda-forge mpich esmpy

The second of the two ``conda`` commands is required for
:ref:`regridding <Regridding>` to work.

----

.. _pip:
  
**pip**
-------

cf-python is in the Python package index:
https://pypi.org/project/cf-python
  
To install cf and all of its :ref:`required dependencies <Required>`
run, for example:

.. code-block:: shell
   :caption: *Install as root, with any missing dependencies.*
	     
   pip install cf-python

.. code-block:: shell
   :caption: *Install as a user, with any missing dependencies.*
	     
   pip install cf-python --user

To install cf without any of its dependencies then run, for example:

.. code-block:: shell
   :caption: *Install as root without installing any of the
             dependencies.*
	     
   pip install cf-python --no-deps

See the `documentation for pip install
<https://pip.pypa.io/en/stable/reference/pip_install/>`_ for further
options.

The :ref:`optional dependencies <Optional>` are **not** automatically
installed via ``pip``.

----

.. _Source:

**Source**
----------

To install from source:

1. Download the cf-python package from https://pypi.org/project/cf-python

2. Unpack the library (replacing ``<version>`` with the version that
   you want to install, e.g. ``3.0.0``):

   .. code:: bash
	 
      tar zxvf cf-python-<version>.tar.gz
      cd cf-python-<version>

3. Install the package:
  
  * To install the cf-python package to a central location:

    .. code:: bash
	 
       python setup.py install

  * To install the cf-python package locally to the user in the default
    location:

    .. code:: bash

       python setup.py install --user

  * To install the cf-python package in the <directory> of your choice:

    .. code:: bash

       python setup.py install --home=<directory>

----

.. _cfa-utility:

**cfa utility**
---------------

During installation the ``cfa`` command line utility is also
installed, which

* :ref:`generates text descriptions of field constructs contained in
  files <File-inspection-with-cfa>`, and 

* :ref:`creates new datasets aggregated from existing files
  <Creation-with-cfa>`.

----

.. _Tests:

**Tests**
---------

Tests are run from within the ``cf-python/test`` directory:

.. code:: bash
 
   python run_tests.py
       
----

.. _Dependencies:

**Dependencies**
----------------

.. _Required:

Required
^^^^^^^^

* `Python <https://www.python.org/>`_, version 3 or newer.

* `numpy <https://pypi.org/project/numpy/>`_, version 1.15 or newer.

* `netCDF4 <https://pypi.org/project/netcdf4/>`_, version 1.4.0 or
  newer.

* `cftime <https://pypi.org/project/cftime/>`_, version 1.0.2 or
  newer. (Note that this package is installed with netCDF4.)

* `cfdm <https://pypi.org/project/cfdm/>`_, version 1.7.7 or newer.
  
* `cfunits <https://pypi.org/project/cfunits/>`_, version 3.2.2 or newer.
  
* `psutil <https://pypi.org/project/psutil/>`_, version 0.6.0 or newer.

* `UNIDATA UDUNITS-2 library
  <http://www.unidata.ucar.edu/software/udunits>`_, version 2.2.20 or
  newer.

  This is a C library which provides support for units of physical
  quantities. If the UDUNITS-2 shared library file
  (``libudunits2.so.0`` on GNU/Linux or ``libudunits2.0.dylibfile`` on
  MacOS) is in a non-standard location then its path should be added
  to the ``LD_LIBRARY_PATH`` environment variable. It may also be
  necessary to specify the location of the ``udunits2.xml`` file in
  the ``UDUNITS2_XML_PATH`` environment variable (although the default
  location is usually correct).

.. _Optional:

Optional
^^^^^^^^

Some further dependencies that, enable further functionality, are
optional. This to faciliate cf-python being installed in restricted
environments:

**Regridding**

* `ESMF <https://www.earthsystemcog.org/projects/esmpy/>`_, version
  7.1.0r or newer. This is easily installed via conda with

  .. code:: bash

     conda install -c conda-forge mpich esmpy

  or may be installed from source.

**Convolution filters, derivatives and relative vorticity**

* `scipy <https://pypi.org/project/scipy>`_, version 1.3.0 or newer.

**Subspacing based on N-d construct cells (N > 1) containing a given value**

* `matplotlib <https://pypi.org/project/matplotlib>`_, version 3.0.0
  or newer.

**Parallel processing**

* `mpi4py <https://pypi.org/project/mpi4py>`_

----

.. _Code-repository:

**Code repository**
-------------------

The complete source code is available at https://github.com/NCAS-CMS/cf-python

.. .. rubric:: Footnotes

   .. [#installfiles] The ``requirements.txt`` file contains

     .. include:: ../../requirements.txt
        :literal:
     
