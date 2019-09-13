.. currentmodule:: cf
.. default-role:: obj

.. _LAMA:

Large Amounts of Massive Arrays (LAMA)
======================================

Data arrays contained within fields are stored and manipulated in a
very memory efficient manner such that large numbers of fields may
co-exist and be manipulated regardless of the size of their data
arrays. The limiting factor on array size is not the amount of
available physical memory, but the amount of disk space available,
which is generally far greater.

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
  partitions are in memory, fields and other variables containing data
  arrays may be used in the API as if they were normal, in-memory
  objects (like numpy arrays). Partitioning does carry a performance
  overhead, but this may be minimised for particular applications or
  hardware configurations.


.. _LAMA_reading:


Reading from files
------------------

When a field is read from a file, the data array is not realized in
memory, however large or small it may be. Instead each partition
refers to part of the original file on disk. Therefore reading even
very large fields is initially very fast and uses up only a very small
amount of memory.

.. _LAMA_copying:

Copying
-------

When a field is deep copied with its `~Field.copy` method or the
:py:func:`copy.deepcopy` function, the partitions of its data array
are transferred to the new field as object identities and are *not*
deep copied. Therefore copying even very large fields is initially
very fast and uses up only a very small amount of memory.

The independence of the copied field is preserved, however, since each
partition that is stored in memory (as opposed to on disk) is deep
copied if and when the data in the new field is actually accessed, and
then only if the partition's data still exists in the original (or any
other) field.

Aggregation
-----------

When two fields are aggregated to form one, larger field there is no
need for either field's data array partitions to be accessed, and
therefore brought into memory if they were stored on disk. The
resulting field recombines the two fields' array partitions as object
identities into the new larger array. Therevy creating an aggregated
field that uses up only a very small amount of extra memory.

The independence of the new field is preserved, however, since each
partition that is stored in memory (as opposed to on disk) is deep
copied when the data in the new field is actually accessed, and then
only if the partition's data still exists in its the original (or any
other) field.

.. _LAMA_subspacing:

Subspacing
----------

When a new field is created by :ref:`subspacing <Subspacing>` a field,
the new field is actually a :ref:`LAMA deep copy <LAMA_copying>` of
the original but with additional instructions on each of its data
partitions to only use the part specified by the subspace indices.

As with copying, creating subspaced fields is initially very fast and
uses up only a very small amount of memory, with the added advantage
that a deep copy of only the requested parts of the data array needs
to be carried out at the time of data access, and then only if the
partition's data still exists in the original field.

When subspacing a field that has previously been subspacing but has
not yet had its data accessed, the new subspace merely updates the
instructions on which parts of the array partitions to use. For
example:

>>> f.shape = (12, 73, 96)
>>> g = f[::-2, ...]
>>> h = g[2:5, ...]

is equivalent to 

>>> h = f[7:2:-2, ...]

and if all of the partitions of field ``f`` are stored on disk then in
both cases so are all of the partitions of field ``h`` and no data has
been read from disk.

Speed and memory management
---------------------------

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

