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

The Python cf package is an Earth Science data analysis library that
is built on a complete implementation of the :ref:`CF-data-model`.

.. contents::
   :local:
   :backlinks: entry

**Functionality**
-----------------

The cf package implements the :ref:`CF-data-model` [#cfdm]_ for its
internal data structures and so is able to process any CF-compliant
dataset. It is not strict about CF-compliance, however, so that
partially conformant datasets may be ingested from existing datasets
and written to new datasets.This is so that datasets that are
partially conformant may nonetheless be modified in memory.

.. code-block:: python
   :caption: *A simple example of reading a field construct from a
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
   Dimension coords: time(12) = [0450-11-16 00:00:00, ..., 0451-10-16 12:00:00] noleap
                   : latitude(64) = [-87.8638, ..., 87.8638] degrees_north
                   : longitude(128) = [0.0, ..., 357.1875] degrees_east
                   : height(1) = [2.0] m

The cf package can:

* read :term:`field constructs <field construct>` and :term:`domain
  constructs <domain construct>` from netCDF, CDL, PP and UM datasets,

* create new field and domain constructs in memory,

* write field and domain constructs to netCDF datasets on disk,

* read, write, and create coordinates defined by geometry cells,

* read and write netCDF4 string data-type variables,

* read, write, and create netCDF and CDL datasets containing
  hierarchical groups,

* inspect field and domain constructs,

* test whether two constructs are the same,

* modify field and domain construct metadata and data,

* create subspaces of field and domain constructs,

* incorporate, and create, metadata stored in external files,

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays), whilst presenting a view of the
  data in its uncompressed form,
    
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

* create moving means from field constructs,

* apply differential operators to field constructs,

* create derived quantities (such as relative vorticity).

All of the above use LAMA functionality, which allows multiple
fields larger than the available memory to exist and be manipulated.

----

**Visualisation**
-----------------


Powerful, flexible, and very simple to produce visualisations of field
constructs are available with the `cfplot` package, that is installed
separately to cf (see http://ajheaps.github.io/cf-plot for details).

See the `cfplot gallery
<http://ajheaps.github.io/cf-plot/gallery.html>`_ for the wide range
range plotting possibilities with example code.

.. figure:: images/cfplot_example.png

   *Example output of cfplot displaying a cf field construct.*

----

**Old versions**
----------------

Since version 3.0.0 (released 2019-10-01), cf is for Python 3 only and
there are :ref:`incompatible differences between versions 2.x and 3.x
<two-to-three-changes>` of cf.

Scripts written for version 2.x but running under version 3.x should
either work as expected, or provide informative error messages on the
new API usage. However, it is advised that the outputs of older
scripts are checked when running with Python 3 versions of the cf
library.

For version 2.x documentation, see the :ref:`releases <Releases>`
page.

----
   
.. [#cfdm] Hassell, D., Gregory, J., Blower, J., Lawrence, B. N., and
           Taylor, K. E.: A data model of the Climate and Forecast
           metadata conventions (CF-1.6) with a software
           implementation (cf-python v2.1), Geosci. Model Dev., 10,
           4619-4646, https://doi.org/10.5194/gmd-10-4619-2017, 2017.
