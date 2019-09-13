.. currentmodule:: cf
.. default-role:: obj

.. _Installation:

**Installation**
================

----

Version |release| for version |version| of the CF conventions.

.. _Python-versions:

**Python versions**
-------------------

As of version 3.0.0, cf works for Python 3 only.

(Versions 2.x of cf work for Python 2 only.)

.. _pip:
  
**pip**
-------

----

To install cf and all of its :ref:`dependencies <Dependencies>` run,
for example:

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

.. _Source:

**Source**
----------

----

To install from source:

1. Download the cf-python package from https://pypi.org/project/cf-python

2. Unpack the library (replacing ``<version>`` with the version that
   you want to install, e.g. ``1.7.0``):

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

.. _cfdump-utility:

**cfdump utility**
------------------

----

During installation the :ref:`cfdump command line utility <cfdump>` is
also installed, which generates text descriptions of the field
constructs contained in a netCDF dataset.

.. _Tests:

**Tests**
---------

----

Tests are run from within the ``cf-python/test`` directory:

.. code:: bash
 
   python run_tests.py
       
.. _Dependencies:

**Dependencies**
----------------

----

The cf-python package requires:

* `Python <https://www.python.org/>`_, version 3 or newer,

* `numpy <http://www.numpy.org/>`_, version 1.15 or newer,

* `netCDF4 <http://unidata.github.io/netcdf4-python/>`_, version 1.4.0
  or newer, and

* `cftime <http://unidata.github.io/netcdf4-python/>`_, version 1.0.2
or newer - this package is installed with netCDF,

* `cfdm <https://pypi.org/project/cfdm/>`_, version 1.7.7 or newer,
  
* `cfunits <https://pypi.org/project/cfunits/>`_, version 3.0.0 or newer, and
  
* `psutil <https://pypi.org/project/psutil/>`_, version 0.6.0 or newer.


.. _Code-repository:

**Code repository**
-------------------

----

The complete source code is available at https://github.com/NCAS-CMS/cf-python

.. .. rubric:: Footnotes

   .. [#installfiles] The ``requirements.txt`` file contains

     .. include:: ../../requirements.txt
        :literal:
     
