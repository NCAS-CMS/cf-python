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

The cf package works only for Linux and Mac operating systems.

If you have a Windows operating system then you can use the `Microsoft
Windows Subsystem for Linux (WSL)
<https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux>`_.

----

.. _Python-versions:

**Python versions**
-------------------

The cf package is only for Python 3.8 or newer.

----

.. _pip:

**pip**
-------

cf is in the Python package index: https://pypi.org/project/cf-python

To install cf and its :ref:`required dependencies <Required>` (apart
from :ref:`Udunits <Udunits>`, and there are also has some
:ref:`optional dependencies <Optional>` which are **not**
automatically installed via ``pip``) run, for example :

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
<https://pip.pypa.io/en/stable/cli/pip_install/>`_ for further
options.

.. _Udunits:

Udunits
^^^^^^^

Udunits (a C library that provides support for units of physical
quantities) is a required dependency that is not installed by ``pip``,
but it is easily installed in a ``conda`` environment:

.. code-block:: console

   $ conda install -c conda-forge udunits2

Alternatively, Udunits is often available in from operating system
software download managers, or may be installed from source.
    
Note that :ref:`some environment variables might also need setting
<UNIDATA-UDUNITS-2-library>` in order for the Udunits library to work
properly, although the defaults are usually sufficient.

See the :ref:`required dependencies <Required>` section for more
details.

----

.. _conda:

**conda**
---------

To install cf with all of its :ref:`required <Required>` and
:ref:`optional <Optional>` dependencies, and the `cf-plot
visualisation package <http://ajheaps.github.io/cf-plot>`_, run

.. code-block:: console
   :caption: *Install with conda.*

   $ conda install -c conda-forge cf-python cf-plot udunits2
   $ conda install -c conda-forge esmpy>=8.0.0

The second of the two ``conda`` commands is required for
:ref:`regridding <Regridding>` to work.

Note that :ref:`some environment variables might also need setting
<UNIDATA-UDUNITS-2-library>` in order for the Udunits library to work
properly, although the defaults are usually sufficient.

----

.. _Source:

**Source**
----------

To install from source (without any dependencies):

1. Download the cf package from https://pypi.org/project/cf-python

2. Unpack the library (replacing ``<version>`` with the version that
   you want to install, e.g. ``3.15.0``):

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
<UNIDATA-UDUNITS-2-library>` in order for the Udunits library to work
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

* `Python <https://www.python.org/>`_, 3.8.0 or newer.

* `numpy <https://pypi.org/project/numpy/>`_, 1.22.0 or newer.

* `dask <https://pypi.org/project/dask/>`_, 2022.12.1 or newer.

* `netCDF4 <https://pypi.org/project/netcdf4/>`_, 1.5.4 or newer.

* `cftime <https://pypi.org/project/cftime/>`_, version 1.6.0 or newer
  (note that this package may be installed with netCDF4).

* `cfdm <https://pypi.org/project/cfdm/>`_, version 1.10.1.0 or up to,
  but not including, 1.10.2.0.

* `cfunits <https://pypi.org/project/cfunits/>`_, version 3.3.5 or newer.

* `psutil <https://pypi.org/project/psutil/>`_, version 0.6.0 or newer.

* `packaging <https://pypi.org/project/packaging/>`_, version 20.0 or newer.

.. _UNIDATA-UDUNITS-2-library:

* `UNIDATA Udunits-2 library
  <http://www.unidata.ucar.edu/software/udunits>`_, version 2.2.25
  or newer. UDUNITS-2 is a C library that provides support for units of
  physical quantities.

  If the Udunits-2 shared library file (``libudunits2.so.0`` on
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

* `esmpy <https://earthsystemmodeling.org/esmpy/>`_, previously
  named `ESMF` with the old module name also being accepted for import,
  version 8.0.0 or newer. This is easily installed via conda with

  .. code-block:: console

     $ conda install -c conda-forge esmpy>=8.0.0

  or may be installed from source.

.. rubric:: Convolution filters, derivatives and relative vorticity

* `scipy <https://pypi.org/project/scipy>`_, version 1.10.0 or newer.

.. rubric:: Subspacing based on N-dimensional construct cells (N > 1)
            containing a given value

* `matplotlib <https://pypi.org/project/matplotlib>`_, version 3.0.0
  or newer.

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
