.. currentmodule:: cf
.. default-role:: obj

.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

.. raw:: html

    <style> .blue {color:blue} </style>

.. role:: blue

**Introduction**
================

----

Version |release| for version |version| of the CF conventions.

The Python `cf` package is an Earth Science data analysis library that
is built on a complete implementation of the :ref:`CF-data-model`.

.. contents::
   :local:
   :backlinks: entry

**Functionality**
-----------------

The `cf` package implements the :ref:`CF-data-model` for its internal
data structures and so is able to process any CF-compliant dataset. It
is not strict about CF-compliance, however, so that partially
conformant datasets may be ingested from existing datasets and written
to new datasets.This is so that datasets that are partially conformant
may nonetheless be modified in memory.

.. code-block:: python
   :caption: *A basic example of reading a field construct from a
             file and inspecting it.*

   >>> import cf
   >>> f = cf.read('file.nc')
   >>> f
   [<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]
   >>> print(f[0])
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(12), latitude(64), longitude(128)) K
   Cell methods    : time(12): mean (interval: 1.0 month)
   Dimension coords: time(12) = [1991-11-16 00:00:00, ..., 1991-10-16 12:00:00] noleap
                   : latitude(64) = [-87.8638, ..., 87.8638] degrees_north
                   : longitude(128) = [0.0, ..., 357.1875] degrees_east
                   : height(1) = [2.0] m

The `cf` package uses :ref:`Dask <Performance>` for all of its array
manipulation and can:

* read :term:`field constructs <field construct>` and :term:`domain
  constructs <domain construct>` from netCDF, CDL, PP and UM datasets,

* create new field constructs in memory,

* write and append field constructs to netCDF datasets on disk,

* read, write, and create coordinates defined by geometry cells,

* read netCDF and CDL datasets containing hierarchical groups,

* inspect field constructs,

* test whether two field constructs are the same,

* modify field construct metadata and data,

* create subspaces of field constructs,

* write field constructs to netCDF datasets on disk,

* incorporate, and create, metadata stored in external files,

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays, or coordinate arrays compressed by
  subsampling), whilst presenting a view of the data in its
  uncompressed form,

* combine field constructs arithmetically,

* manipulate field construct data by arithmetical and trigonometrical
  operations,

* perform statistical collapses on field constructs,

* perform histogram, percentile and binning operations on field
  constructs,

* regrid field constructs with (multi-)linear, nearest neighbour,
  first- and second-order conservative and higher order patch recovery
  methods,

* apply convolution filters to field constructs,

* create running means from field constructs,

* apply differential operators to field constructs,

* create derived quantities (such as relative vorticity).

----

**Visualisation**
-----------------

Powerful, flexible, and user-friendly visualisations of field
constructs are available with the `cf-plot` package that is installed
separately to `cf` (see http://ajheaps.github.io/cf-plot for details).

See the `cf-plot gallery
<http://ajheaps.github.io/cf-plot/gallery.html>`_ for the wide range
of plotting possibilities with example code.

.. figure:: images/cfplot_example.png

   *Example output of cf-plot displaying a cf field construct.*

----

**Performance**
---------------

As of version 3.14.0 (released 2023-01-31), cf uses :ref:`Dask
<Performance>` for all of its data manipulations, which provides lazy,
parallelised, and out-of-core computations of array operations.

----

**Command line utilities**
--------------------------

During installation the ``cfa`` command line utility is also
installed, which

* generates text descriptions of field constructs contained in files,
  and

* creates new datasets aggregated from existing files.

----

**References**
--------------

Eaton, B., Gregory, J., Drach, B., Taylor, K., Hankin, S., Caron, J.,
  Signell, R., et al. NetCDF Climate and Forecast (CF) Metadata
  Conventions. CF Conventions Committee.
  https://cfconventions.org/cf-conventions/cf-conventions.html

Hassell, D., and Bartholomew, S. L. (2020). cfdm: A Python reference
  implementation of the CF data model. Journal of Open Source
  Software, 5(54), 2717, https://doi.org/10.21105/joss.02717

Hassell, D., Gregory, J., Blower, J., Lawrence, B. N., and
  Taylor, K. E. (2017). A data model of the Climate and Forecast
  metadata conventions (CF-1.6) with a software implementation
  (cf-python v2.1), Geosci. Model Dev., 10, 4619-4646,
  https://doi.org/10.5194/gmd-10-4619-2017

Rew, R., and Davis, G. (1990). NetCDF: An Interface for Scientific
  Data Access. IEEE Computer Graphics and Applications, 10(4),
  76–82. https://doi.org/10.1109/38.56302

Rew, R., Hartnett, E., and Caron, J. (2006). NetCDF-4: Software
  Implementing an Enhanced Data Model for the Geosciences. In 22nd
  International Conference on Interactive Information Processing
  Systems for Meteorology, Oceanography, and Hydrology. AMS. Retrieved
  from
  https://www.unidata.ucar.edu/software/netcdf/papers/2006-ams.pdf

