.. currentmodule:: cf
.. default-role:: obj

.. _parallel-collapse:

Parallel Collapse
=================

cf-python scripts may be run in parallel using mpirun and any calls to
collapse will be parallelised. This requires the mpi4py module to be
installed. Only calls to collapse will be parallelised. Other methods
are not yet parallelised. This is an experimental feature and is not
recommended for operational use.

Installing mpi4py with conda
----------------------------

To install mpi4py with conda type::

  $ conda install mpi4py

This also installs the mpich implementation of MPI. To install MPI 3
rather than 2 type::

  $ conda install -c conda-forge mpi4py

It is planned to use MPI 3 features in future versions of cf-python.

If you are using another version of Python other than Anaconda you
will need to install either mpich or openmpi and mpi4py.

Note that if you are using regridding in a parallel script ESMF must
be compiled without parallel support.

Running a cf-python script in parallel
--------------------------------------

To use a cf-python script in parallel with 2 processors type::

  $ mpirun -n 2 python name_of_script.py

To use more processors increase the number after n. We would expect
this to work across multiple nodes as well as on a single node,
although the use of memory may not be optimal. If using across
multiple nodes `cf.TEMPDIR` must be set to a shared location.

If you get an error `gethostbyname failed` you could consider adding
the following line to `/etc/hosts`::

  127.0.0.1 computername

where `computername` is the name of your machine.

Optimising a parallel collapse operation
----------------------------------------

There are three possible modes of optimisation, which can be set in
your script using `cf.COLLAPSE_PARALLEL_MODE`:

0.  This attempts to maximise parallelism, possibly at the expense of
    extra communication. This is the default mode.

1.  This minimises communication, possibly at the expense of the
    degree of parallelism. If collapse is running slower than you
    would expect, you can try changing to mode 1 to see if this
    improves performance. This is only likely to work if the output of
    collapse will be a sizeable array, not a single point.

2.  This is here for debugging purposes, but we would expect this to
    maximise communication possibly at the expense of parallelism. The
    use of this mode is, therefore, not recommended.

Calling `cf.COLLAPSE_PARALLEL_MODE` with no arguments returns the
current value, otherwise the value prior to the change is returned.

:Examples:

>>> cf.COLLAPSE_PARALLEL_MODE()
0
>>> cf.COLLAPSE_PARALLEL_MODE(1)
0
>>> cf.COLLAPSE_PARALLEL_MODE()
1

The parallelism is based on partitions created by :ref:`LAMA <LAMA>`
and will be affected by the size and number of those partitions. The
size of the partitions can be changed by calling `cf.CHUNKSIZE` before
reading a file.
