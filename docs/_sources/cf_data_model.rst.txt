.. currentmodule:: cf
.. default-role:: obj

.. _CF-data-model:

**CF data model**
=================

----

The CF (Climate and Forecast) metadata conventions
(http://cfconventions.org) provide a description of the physical
meaning of data and of their spatial and temporal properties and are
designed to promote the creation, processing, and sharing of climate
and forecasting data using netCDF files and libraries
(https://www.unidata.ucar.edu/software/netcdf).

The `CF data model <https://doi.org/10.5194/gmd-10-4619-2017>`_
[#cfdm]_ identifies the fundamental elements ("constructs") of the CF
conventions and shows how they relate to each other, independently of
the netCDF encoding.

The **field** construct defined by the CF data model, which
corresponds to a CF-netCDF data variable with all of its metadata, is
the central construct that includes all of the other constructs. It
consists of

- descriptive properties that apply to field construct as a whole
  (e.g. the standard name),

- a data array, and

- "metadata constructs" that describe
  
  - the locations of each cell of the data array (i.e. the "domain"),
    and

  - the physical nature of each cell's datum.

The domain is defined by

- **domain axis** constructs (corresponding to CF-netCDF dimensions or
  scalar coordinate variables),

- **dimension coordinate** constructs (corresponding to CF-netCDF
  coordinate variables or numeric scalar coordinate variables),

- **auxiliary coordinate** constructs (corresponding to CF-netCDF
  auxiliary coordinate variables and non-numeric scalar coordinate
  variables),

- **coordinate reference** constructs (corresponding to CF-netCDF grid
  mapping variables or the formula_terms attribute of a coordinate
  variable),

- **domain ancillary** constructs (corresponding to CF-netCDF
  variables named by the formula_terms attribute of a coordinate
  variable), and

- **cell measure** constructs (corresponding to CF-netCDF cell measure
  variables).
  
The physical nature of individual data values are described by 

- **field ancillary** constructs (corresponding to CF-netCDF ancillary
  variables), and

- **cell method** constructs (corresponding to a CF-netCDF
  cell_methods attribute of data variable).


.. [#cfdm] Hassell, D., Gregory, J., Blower, J., Lawrence, B. N., and
           Taylor, K. E.: A data model of the Climate and Forecast
           metadata conventions (CF-1.6) with a software
           implementation (cf-python v2.1), Geosci. Model Dev., 10,
           4619-4646, https://doi.org/10.5194/gmd-10-4619-2017, 2017.
	   
