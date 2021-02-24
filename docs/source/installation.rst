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

.. note:: The latest versions of cf available from the Python package index
          (PyPI) and conda are confirmed at `the top of the README document
          <https://github.com/NCAS-CMS/cf-python#cf-python>`_.

.. _Operating-systems:

**Operating systems**
---------------------

The cf package works only for Linux and Mac operating systems.

If you have a Windows operating system then you can either install the
`Microsoft Windows Subsystem for Linux (WSL)
<https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux>`_, or
installing a Linux Virtual Machine also works.

----

.. _Python-versions:

**Python versions**
-------------------

The cf package is only Python 3.6 or later. (Versions 2.x of cf work
only for Python 2.7.)

----

.. _pip:

**pip**
-------

cf is in the Python package index: https://pypi.org/project/cf-python

cf has some :ref:`optional dependencies <Optional>` which are **not**
automatically installed via ``pip``, but to install cf and all of its
:ref:`required dependencies <Required>` run, for example:

.. code-block:: console
   :caption: *Install as root, with any missing dependencies.*

   $ pip install cf-python

.. code-block:: console
   :caption: *Install as a user, with any missing dependencies.*

   $ pip install cf-python --user

To install cf without any of its dependencies then run, for example:

.. code-block:: console
   :caption: *Install as root without installing any of the
             dependencies.*

   $ pip install cf-python --no-deps

See the `documentation for pip install
<https://pip.pypa.io/en/stable/reference/pip_install/>`_ for further
options.

Note that :ref:`some environment variables might also need setting
<UNIDATA-UDUNITS-2-library>` in order for the UDUNITS library to work
properly, although the defaults are usually sufficient.

----

.. _conda:

**conda**
---------

The cf package is in the ``ncas`` conda channel. To install cf with
all of its :ref:`required <Required>` and :ref:`optional <Optional>`
dependencies, and the `cf-plot visualisation package
<http://ajheaps.github.io/cf-plot>`_, run

.. code-block:: console
   :caption: *Install with conda.*

   $ conda install -c ncas -c conda-forge cf-python cf-plot udunits2==2.2.25
   $ conda install -c conda-forge mpich esmpy

The second of the two ``conda`` commands is required for
:ref:`regridding <Regridding>` to work. (Note, however, that the
installation of ``esmpy`` does not work for Anaconda version
``2019.10``.)

Note that :ref:`some environment variables might also need setting
<UNIDATA-UDUNITS-2-library>` in order for the UDUNITS library to work
properly, although the defaults are usually sufficient.

----

.. _Source:

**Source**
----------

To install from source:

1. Download the cf package from https://pypi.org/project/cf-python

2. Unpack the library (replacing ``<version>`` with the version that
   you want to install, e.g. ``3.9.0``):

   .. code-block:: console

      $ tar zxvf cf-python-<version>.tar.gz
      $ cd cf-python-<version>

3. Install the package:

  * To install the cf-python package to a central location:

    .. code-block:: console

       $ python setup.py install

  * To install the cf-python package locally to the user in the default
    location:

    .. code-block:: console

       $ python setup.py install --user

  * To install the cf-python package in the ``<directory>`` of your
    choice:

    .. code-block:: console

       $ python setup.py install --home=<directory>

Note that :ref:`some environment variables might also need setting
<UNIDATA-UDUNITS-2-library>` in order for the UDUNITS library to work
properly, although the defaults are usually sufficient.

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

.. _Dependencies:

**Dependencies**
----------------

.. _Required:

Required
^^^^^^^^

* `Python <https://www.python.org/>`_, 3.6 or newer.

* `numpy <https://pypi.org/project/numpy/>`_, 1.15 or newer.

* `netCDF4 <https://pypi.org/project/netcdf4/>`_, 1.5.3 or newer.

* `cftime <https://pypi.org/project/cftime/>`_, version 1.4.0 or newer
  (note that this package may be installed with netCDF4).

* `cfdm <https://pypi.org/project/cfdm/>`_, version 1.8.9.0 or up to,
  but not including, 1.8.10.0.

* `cfunits <https://pypi.org/project/cfunits/>`_, version 3.3.1 or newer.

* `psutil <https://pypi.org/project/psutil/>`_, version 0.6.0 or newer.

.. _UNIDATA-UDUNITS-2-library:

* `UNIDATA UDUNITS-2 library
  <http://www.unidata.ucar.edu/software/udunits>`_, 2.2.20 <=
  version. UDUNITS-2 is a C library that provides support for units of
  physical quantities.

  If the UDUNITS-2 shared library file (``libudunits2.so.0`` on
  GNU/Linux or ``libudunits2.0.dylibfile`` on MacOS) is in a
  non-standard location then its directory path should be added to the
  ``LD_LIBRARY_PATH`` environment variable.

  It may also be necessary to specify the location (directory path
  *and* file name) of the ``udunits2.xml`` file in the
  ``UDUNITS2_XML_PATH`` environment variable, although the default
  location is usually correct. For example, ``export
  UDUNITS2_XML_PATH=/home/user/anaconda3/share/udunits/udunits2.xml``.
  If you get a run-time error that looks like ``assert(0 ==
  _ut_unmap_symbol_to_unit(_ut_system, _c_char_p(b'Sv'), _UT_ASCII))``
  then setting the ``UDUNITS2_XML_PATH`` environment variable is the
  likely solution.

.. _Optional:

Optional
^^^^^^^^

Some further dependencies that enable further functionality are
optional. This to facilitate cf-python being installed in restricted
environments for which these features are not required.

.. rubric:: Regridding

* `ESMF <https://www.earthsystemcog.org/projects/esmpy/>`_, version
  8.0.0 or newer. This is easily installed via conda with

  .. code-block:: console

     $ conda install -c conda-forge mpich esmpy

  or may be installed from source.

.. rubric:: Convolution filters, derivatives and relative vorticity

* `scipy <https://pypi.org/project/scipy>`_, version 1.1.0 or newer.

.. rubric:: Subspacing based on N-dimensional construct cells (N > 1)
            containing a given value

* `matplotlib <https://pypi.org/project/matplotlib>`_, version 3.0.0
  or newer.

.. rubric:: Parallel processing

* `mpi4py <https://pypi.org/project/mpi4py>`_

----

.. _Tests:

**Tests**
---------

Tests are run from within the ``cf/test`` directory:

.. code-block:: console

   $ python run_tests.py

----

.. _Code-repository:

**Code repository**
-------------------

The complete source code and issue tracker is available at
https://github.com/NCAS-CMS/cf-python

----

.. .. rubric:: Footnotes

   .. [#installfiles] The ``requirements.txt`` file contains

     .. include:: ../../requirements.txt
        :literal:
