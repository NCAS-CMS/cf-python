.. currentmodule:: cf
.. default-role:: obj

.. TODODASK - review this entire section

.. currentmodule:: cf
.. default-role:: obj

.. _Performance:


**Performance**
===============

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry

.. _Dask:

**Dask**
--------

A data array in `cf` is stored internally by a `Dask array
<https://docs.dask.org/en/latest/array.html>`_, that provides lazy,
parallelised, and out-of-core array computations of array
operations. Computations are automatically optimised to maximise the
re-use of data in memory, and to avoid calculations that do not
contribute for the final result.

The performance of `cf` is largely a function of the performance of
Dask, and most of the techniques that Dask supports for `improving
performance <https://docs.dask.org/en/stable/best-practices.html>`_
can be applied through the `cf` API.

----

.. _Lazy-operations:


**Lazy operations**
-------------------

In general, all `cf` operations (such as reading from disk,
regridding, collapsing, subspacing, arithmetic, etc.) are lazy,
meaning that an operation is not actually performed until the result
is actually inspected, for instance by creating a plot of the data or
writing the data to disk. When multiple operations are applied one
after another, none of the operations are computed until the result of
final one is requested.

There are, however, a small number of `cf` operations that require the
automatic computation of some data operations. The results of these
computations are not saved in memory, so if the operation is repeated
then the computed is re-calculated in its entirety unless the
construct's `~.cf.Field.persist` method was called beforehand.

Some notable cases where non-lazy computation occurs are:

* **Aggregation**

  When two or more field or domain constructs are aggregated to form a
  single construct, either by `cf.read` or `cf.aggregate`, the data
  arrays of some metadata constructs (coordinates, cell measures,
  etc.)  must be inspected to ascertain if the aggregation is
  possible.

..

* **Regridding**

  When regridding a field construct with its `cf.Field.regrids` or
  `cf.Field.regridc` method, the regridding weights are computed
  non-lazily, which requires inspection of some or all of the
  coordinate data. These calculations can be much more costly than the
  regridding itself. When multiple regrid operations have the same
  weights, performance can be improved be calculating the weights once
  and re-using them:
  
  .. code-block:: python
     :caption: *For a list of fields 'fl' with the same horizontal
               domain, regrid then all using pre-computed regridding
               weights to the domain defined by field 'dst'.*
  		
     >>> weights = fl[0].regrids(dst, method='conservative', return_operator=True)
     >>> regridded = [f.regrids(weights) for f in fl]
   
----

.. _Workflow-chunks:


**Workflow chunks**
-------------------

A workflow chunk - part of the data which is associated with one
processing task - is identical to a `Dask chunk
<https://docs.dask.org/en/stable/array-chunks.html>`_.

In general, good performance results from following these `rules
<https://docs.dask.org/en/stable/array-chunks.html#specifying-chunk-shapes>`_:

* A chunk should be small enough to fit comfortably in memory. There
  will have many chunks in memory at once. Dask will often have as
  many chunks in memory as twice the number of active threads.

* A chunk must be large enough so that computations on that chunk take
  significantly longer than the ``1 ms`` overhead per task that Dask
  scheduling incurs. A task should take longer than ``100 ms``.

* Chunk sizes between ``10 MiB`` and ``1 GiB`` are common, depending
  on the availability of RAM and the duration of computations.

* Chunks should align with the computation that you want to do. For
  example, if you plan to frequently slice along a particular
  dimension, then it's more efficient if your chunks are aligned so
  that you have to touch fewer chunks. If you want to add two arrays,
  then its convenient if those arrays have matching chunks patterns.

* Chunks should align with your storage, if applicable.

By default, chunks have a size of at most ``128 MiB`` and prefer
square-like shapes. A new default chunk size may be set with the
`cf.chunksize` function. The default chunk size may be overridden loc by
`cf.read` and when creating `cf.Data` instances. Any data may be
re-chunked after its creation with the `cf.Data.rechunk` method.

For more information, see `Choosing good chunk sizes in Dask
<https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes>`_.

----

.. _Parallel-computation:

**Parallel computation**
------------------------

All operations are executed in parallel using `Dask dynamic task
scheduling <https://docs.dask.org/en/stable/scheduling.html>`_.

Operations are stored by Dask in `task graphs
<https://docs.dask.org/en/stable/graphs.html>`_ where each node in the
graph is a task defined by an operation on a workflow chunk, and data
created by one node are used as inputs to the next node in the
graph. The tasks in the graph are passed by a `scheduler
<https://docs.dask.org/en/stable/scheduling.html>`_ to the available
pool of processing elements (PEs) which execute the tasks in parallel
until the final result has been computed. The default scheduler uses
threads on the local machine, but it is easy to use local processes,
or cluster of many machines, or a single-threaded synchronous
scheduler that executes all computations in the local thread with no
parallelism at all.

Implementing a different scheduler is done via any of the methods
supported by Dask, and all `cf` operations executed after a new
scheduler has been defined will use that scheduler.

.. code-block:: python
   :caption: *One technique for executing operations on a remote
             server.*

   >>> from dask.distributed import Client
   >>> client = Client('127.0.0.1:8786')
   >>> import cf

----
