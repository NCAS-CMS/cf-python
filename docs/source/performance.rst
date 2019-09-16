.. currentmodule:: cf
.. default-role:: obj

**Performance**
===============

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry


**Lazy operations**
-------------------

Many data operations of field and metadata constructs are lazy
i.e. the operation is actually performed until the result is needed,
or the underlying data are shared with a `copy-on-write
<https://en.wikipedia.org/wiki/Copy-on-write>`_ technique, or `lazy
loading <https://en.wikipedia.org/wiki/Lazy_loading>`_ delays reading
data from disk until it is needed. These are:

* Reading datasets from disk
* Subspacing (in index and metadata space)
* Axis manipulations (such as rearranging the axis order  of the data)
* Copying
* Field construct aggregation (and data concatenation in general)
* Changing the units
  
Operations that involve changing the data values (apart from changing
the units) are not lazy; such arithmetic, trigonometrial, rounding,
collapse, etc. functions.

All of these lazy techniques make use of :ref:`LAMA (Large Amounts
of Massive Arrays) <LAMA>` functionality.


----

**Parallelization**
-------------------

cf-python scripts may be run in parallel using mpirun and any calls to
collapse will be parallelised. This requires the mpi4py module to be
installed. Only calls to collapse will be parallelised. Other methods
are not yet parallelised. This is an experimental feature and is not
yet recommended for operational use.

Installing mpi4py with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install mpi4py with conda type::

  $ conda install mpi4py

This also installs the mpich implementation of MPI. To install MPI3
rather than MPI2 type::

  $ conda install -c conda-forge mpi4py

It is planned to use MPI3 features in future versions of cf-python.

If you are using another version of Python other than Anaconda you
will need to install either mpich or openmpi and mpi4py.

Note that if you are using regridding in a parallel script then ESMF
must be compiled *without* parallel support.

Running a cf-python script in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a cf-python script in parallel, pass the python invocation of
the script to ``mpirun``

.. code-block:: shell
   :caption: *Run my_script.py in parallel with 3 processors.*
	     
   $ mpirun -n 3 python my_script.py

This will work across multiple nodes, as well as on a single node. If
using across multiple nodes then `cf.TEMPDIR` must be set to a shared
location. Note that internode communication costs my be reduce the
speed-up when multiple nodes are in use.

(If you get the error ``gethostbyname failed`` you could consider
adding ``127.0.0.1 computername`` to the file ``/etc/hosts``, where
``computername`` is the name of your machine.)

Optimising a parallel collapse operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three possible modes of optimisation, which can be set in
your script using `cf.COLLAPSE_PARALLEL_MODE`:

=====  ===============================================================
Mode   Description
=====  ===============================================================
``0``  This attempts to maximise parallelism, possibly at the expense
       of extra communication. This is the default mode.

``1``  This minimises communication, possibly at the expense of the
       degree of parallelism. If collapse is running slower than you
       would expect, you can try changing to mode 1 to see if this
       improves performance. This is only likely to work if the output
       of collapse will be a sizeable array, not a single point.

``2``  This is here for debugging purposes, but we would expect this
       to maximise communication possibly at the expense of
       parallelism. The use of this mode is, therefore, not
       recommended.
=====  ===============================================================
       
Calling `cf.COLLAPSE_PARALLEL_MODE` with no arguments returns the
current value, otherwise the value prior to the change is returned.

.. code-block:: python
   :caption: *TODO*

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

----

.. _LAMA:

**Large Amounts of Massive Arrays (LAMA)**
------------------------------------------

Data are stored and manipulated in a very memory efficient manner such
that large numbers of constructs may co-exist and be manipulated
regardless of the size of their data arrays. The limiting factor on
array size is not the amount of available physical memory, but the
amount of disk space available, which is generally far greater.

The basic functionality is:

* **Arrays larger than the available physical memory may be created.**

  Arrays larger than a preset number of bytes are partitioned into
  smaller sub-arrays which are not kept in memory but stored on disk,
  either as part of the original file they were read in from or as
  newly created temporary files, whichever is appropriate.  Therefore
  data arrays larger than a preset size need never wholly exist in
  memory.

* **Large numbers of arrays which are in total larger than the
  available physical memory may co-exist.**

  Large numbers of data arrays which are collectively larger than the
  available physical memory may co-exist, regardless of whether any
  individual array is in memory or stored on disk.

* **Arrays larger than the available physical memory may be operated
  on.**
  
  Array operations (such as subspacing, assignment, arithmetic,
  comparison, collapses, etc.) are carried out on a
  partition-by-partition basis. When an operation traverses the
  partitions, if a partition is stored on disk then it is returned to
  disk before processing the next partition.

* **The memory management does not need to be known at the API
  level.**

  As the array's metadata (such as size, shape, data-type, number of
  dimensions, etc.) always reflect the data array in its entirety,
  regardless of how the array is partitioned and whether or not its
  partitions are in memory, constructs containing data arrays may be
  used in the API as if they were normal, in-memory objects (like
  numpy arrays). Partitioning does carry a performance overhead, but
  this may be minimised for particular applications or hardware
  configurations.


.. _LAMA_reading:


Reading from files
^^^^^^^^^^^^^^^^^^

When a field construct is read from a file, the data array is not
realized in memory, however large or small it may be. Instead each
partition refers to part of the original file on disk. Therefore
reading even very large field constructs is initially very fast and
uses up only a very small amount of memory.

.. _LAMA_copying:

Copying
^^^^^^^

When a field construct is deep copied with its `~Field.copy` method or
the :py:func:`copy.deepcopy` function, the partitions of its data
array are transferred to the new field construct as object identities
and are *not* deep copied. Therefore copying even very large field
constructs is initially very fast and uses up only a very small amount
of memory.

The independence of the copied field construct is preserved, however,
since each partition that is stored in memory (as opposed to on disk)
is deep copied if and when the data in the new field construct is
actually accessed, and then only if the partition's data still exists
in the original (or any other) field construct.

Aggregation
^^^^^^^^^^^

When two field constructs are aggregated to form one, larger field
construct there is no need for either field construct's data array
partitions to be accessed, and therefore brought into memory if they
were stored on disk. The resulting field construct recombines the two
field constructs' array partitions as object identities into the new
larger array. Thereby creating an aggregated field construct that uses
up only a very small amount of extra memory.

The independence of the new field construct is preserved, however,
since each partition that is stored in memory (as opposed to on disk)
is deep copied when the data in the new field construct is actually
accessed, and then only if the partition's data still exists in its
the original (or any other) field construct.

.. _LAMA_subspacing:

Subspacing
^^^^^^^^^^

When a new field construct is created by :ref:`subspacing
<Subspacing>` a field construct, the new field construct is actually a
:ref:`LAMA deep copy <LAMA_copying>` of the original but with
additional instructions on each of its data partitions to only use the
part specified by the subspace indices.

As with copying, creating subspaced field constructs is initially very
fast and uses up only a very small amount of memory, with the added
advantage that a deep copy of only the requested parts of the data
array needs to be carried out at the time of data access, and then
only if the partition's data still exists in the original field
construct.

When subspacing a field construct that has previously been subspacing
but has not yet had its data accessed, the new subspace merely updates
the instructions on which parts of the array partitions to use. For
example:

>>> f.shape = (12, 73, 96)
>>> g = f[::-2, ...]
>>> h = g[2:5, ...]

is equivalent to 

>>> h = f[7:2:-2, ...]

and if all of the partitions of field construct``f`` are stored on
disk then in both cases so are all of the partitions of field
construct ``h`` and no data has been read from disk.

Speed and memory management
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The creation of temporary files for array partitions of large arrays
and the reading of data from files on disk can create significant
speed overheads (for example, recent tests show that writing a 100
megabyte array to disk can take O(1) seconds), so it may be desirable
to configure the maximum size of array which is kept in memory, and
therefore has fast access.

The data array memory management is configurable as follows:

* Data arrays larger than a given size are partitioned into
  sub-arrays, each of which is smaller than the chunk size. By default
  this size is set to 1% of the total physical memory and is found and
  set with the `cf.CHUNKSIZE` function.

* In-memory sub-arrays may be written to tempoary files on disk when
  the amount of available physical memory falls below a specified
  amount. By default this amount is 10% of the total physical memory
  and is found and set with the `cf.MINNCFCM` function.

.. _LAMA_temporary_files:

Temporary files
^^^^^^^^^^^^^^^

The directory in which temporary files is found and set with the
`cf.TEMPDIR` function:

>>> cf.TEMPDIR()
'/tmp'
>>> cf.TEMPDIR('/home/me/tmp')
>>> cf.TEMPDIR()
'/home/me/tmp'

The removal of temporary files which are no longer required works in
the same way as python's automatic garbage collector. When a
partition's data is stored in a temporary file, that file will only
exist for as long as there are partitions referring to it. When no
partitions require the file it will be deleted automatically.

When python exits normally, all temporary files are always deleted.

Changing the temporary file directory does not prevent temporary files
in the original directory from being garbage collected.


Partitioning
^^^^^^^^^^^^

To maximise looping efficiency, array partitioning preserves as much
as is possible the faster varying (inner) dimensions' sizes in each of
the sub-arrays.

**Examples**
   If an array with shape (2, 4, 6) is partitioned into 2 partitions then both
   sub-arrays will have shape (1, 4, 6).
 
   If the same array is partitioned into 4 partitions then all four sub-arrays
   will have shape (1, 2, 6).

   If the same array is partitioned into 8 partitions then all eight
   sub-arrays will have shape (1, 2, 3).

   If the same array is partitioned into 48 partitions then all forty eight
   sub-arrays will have shape (1, 1, 1).

