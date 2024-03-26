cf-python
=========

The Python `cf` package is an Earth Science data analysis library that
is built on a complete implementation of the CF data model.

[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/NCAS-CMS/cf-python?color=000000&label=latest%20version)](https://ncas-cms.github.io/cf-python/Changelog.html)
[![PyPI](https://img.shields.io/pypi/v/cf-python?color=000000)](https://pypi.org/project/cf-python/)
[![Conda](https://img.shields.io/conda/v/conda-forge/cf-python?color=000000)](https://anaconda.org/conda-forge/cf-python)

[![Conda](https://img.shields.io/conda/pn/conda-forge/cf-python?color=2d8659)](https://ncas-cms.github.io/cf-python/installation.html#operating-systems)
[![Website](https://img.shields.io/website?color=2d8659&down_message=online&label=documentation&up_message=online&url=https%3A%2F%2Fncas-cms.github.io%2Fcf-python%2F)](https://ncas-cms.github.io/cf-python/index.html)
[![GitHub](https://img.shields.io/github/license/NCAS-CMS/cf-python?color=2d8659)](https://github.com/NCAS-CMS/cf-python/blob/main/LICENSE)

[![Codecov](https://img.shields.io/codecov/c/github/NCAS-CMS/cfdm?color=006666)](https://codecov.io/gh/NCAS-CMS/cfdm)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/NCAS-CMS/cfdm/run-test-suite.yml?branch=main?color=006666&label=test%20suite%20workflow)](https://github.com/NCAS-CMS/cfdm/actions)

[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

#### References

[![Website](https://img.shields.io/website?down_color=264d73&down_message=10.5281%2Fzenodo.3894533&label=DOI&up_color=264d73&up_message=10.5281%2Fzenodo.3894533&url=https%3A%2F%2Fdoi.org%2F10.5281%2Fzenodo.3894533)](https://doi.org/10.5281/zenodo.3894533)
[![Website](https://img.shields.io/website?down_color=264d73&down_message=10.5194%2Fgmd-10-4619-2017&label=GMD&up_color=264d73&up_message=10.5194%2Fgmd-10-4619-2017&url=https%3A%2F%2Fwww.geosci-model-dev.net%2F10%2F4619%2F2017%2F)](https://www.geosci-model-dev.net/10/4619/2017/)
[![Website](https://img.shields.io/website?down_color=264d73&down_message=10.21105%2Fjoss.02717&label=JOSS&up_color=264d73&up_message=10.21105%2Fjoss.02717&url=https:%2F%2Fjoss.theoj.org%2Fpapers%2F10.21105%2Fjoss.02717%2Fstatus.svg)](https://doi.org/10.21105/joss.02717)

Dask
====

From version 3.14.0 the `cf` package uses
[Dask](https://docs.dask.org) for all of its data manipulations.

Documentation
=============

http://ncas-cms.github.io/cf-python

Installation
============

http://ncas-cms.github.io/cf-python/installation.html

Cheat Sheet
===========

https://ncas-cms.github.io/cf-python/cheat_sheet.html

Recipes
=======

https://ncas-cms.github.io/cf-python/recipes

Tutorial
========

https://ncas-cms.github.io/cf-python/tutorial.html

Functionality
=============

The `cf` package implements the [CF data
model](https://cfconventions.org/cf-conventions/cf-conventions.html#appendix-CF-data-model)
for its internal data structures and so is able to process any
CF-compliant dataset. It is not strict about CF-compliance, however,
so that partially conformant datasets may be ingested from existing
datasets and written to new datasets. This is so that datasets which
are partially conformant may nonetheless be modified in memory.

A simple example of reading a field construct from a file and
inspecting it:

    >>> import cf
    >>> f = cf.read('file.nc')
    >>> print(f[0])
    Field: air_temperature (ncvar%tas)
    ----------------------------------
    Data            : air_temperature(time(12), latitude(64), longitude(128)) K
    Cell methods    : time(12): mean (interval: 1.0 month)
    Dimension coords: time(12) = [1991-11-16 00:00:00, ..., 1991-10-16 12:00:00] noleap
                    : latitude(64) = [-87.8638, ..., 87.8638] degrees_north
                    : longitude(128) = [0.0, ..., 357.1875] degrees_east
                    : height(1) = [2.0] m

The `cf` package uses
[Dask](https://ncas-cms.github.io/cf-python/performance.html) for all
of its array manipulation and can:

* read field constructs from netCDF, CDL, PP and UM datasets,
* create new field constructs in memory,
* write and append field and domain constructs to netCDF datasets on disk,
* read, create, and manipulate UGRID mesh topologies,
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
* perform weighted statistical collapses on field constructs,
  including those with geometry cells and UGRID mesh topologies,
* perform histogram, percentile and binning operations on field
  constructs,
* regrid structured grid, mesh and DSG field constructs with
  (multi-)linear, nearest neighbour, first- and second-order
  conservative and higher order patch recovery methods, including 3-d
  regridding,
* apply convolution filters to field constructs,
* create running means from field constructs,
* apply differential operators to field constructs,
* create derived quantities (such as relative vorticity).

Visualization
=============

Powerful, flexible, and very simple to produce visualizations of field
constructs are available with the [cfplot
package](http://ajheaps.github.io/cf-plot), that needs to be installed
seprately to the `cf` package.

See the [cf-plot
gallery](http://ajheaps.github.io/cf-plot/gallery.html) for the full
range of plotting possibilities with example code.

![Example output of cf-plot displaying a `cf` field construct](docs/source/images/cfplot_example.png)

Command line utilities
======================

During installation the ``cfa`` command line utility is also
installed, which

* generates text descriptions of field constructs contained in files,
  and

* creates new datasets aggregated from existing files.


Tests
=====

Tests are run from within the ``cf/test`` directory:

    python run_tests.py
