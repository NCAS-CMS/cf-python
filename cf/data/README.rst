`cf.Data` developer notes
=========================

Masked arrays
-------------

Whether there is a mask or not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For methods such as `equals`, we need to consider whether an array is
a masked one, and if so, we need to consider the *masks* (e.g. whether they
are equal), as well as the *data* (equality or otherwise).

But the difficulty is that some level of inspection, i.e. computation, is
required to know whether the object in question is masked or not! (This is
due to, fundamentally, the underlying netCDF or PP representation.)
And we want to avoid early computation, as again it is inefficient.

Consider, for example, the case of a set of computations in which an
array may acquire a mask, or may not: until the `compute` is run,
we don't know whether there is a mask at the end. Note there is a
distinction here between a standard `array` and a `masked` array
which may have a trivial (say, all `False`) or non-trivial mask, e.g.
for Dask array cases (similarly for `np.ma` etc.):

**Masked array with a non-trivial mask:**

.. code-block:: python

   >>> dx = da.from_array(np.ma.array([1, 2, 3], mask=[1, 0, 0]))
   >>> dx
   dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.MaskedArray>

**Masked array with a trivial i.e. all-Falsy mask:**

.. code-block:: python

   >>> dy = da.from_array(np.ma.array([1, 2, 3], mask=[0, 0, 0]))
   >>> dy
   dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.MaskedArray>

**Standard array i.e. no mask:**

.. code-block:: python

   >>> dz = da.from_array(np.array([1, 2, 3]))
   >>> dz
   dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>


Solution
########

To work around the complication of not being able to know whether an array
is a masked one or not in any cases of computation where a mask may be
added, we will, for all these cases, use the fact that standard arrays (i.e.
example 3 above) can also be queried with `da.ma.getmaskarray`, returning
an all-False mask (just like a masked array with an all-False mask, i.e.
example 2 above, would):

.. code-block:: python

   >>> dz = da.from_array(np.array([1, 2, 3]))  # i.e. example 3 above
   >>> mz = da.ma.getmaskarray(dz)
   >>> mz.compute()
   array([False, False, False])

   >>> dy = da.from_array(np.ma.array([1, 2, 3], mask=[0, 0, 0]))  # i.e. example 2
   >>> my = da.ma.getmaskarray(dy)
   >>> my.compute()
   array([False, False, False])


Hardness of the mask
^^^^^^^^^^^^^^^^^^^^

Any `cf.Data` method that changes the dask array should consider
whether or not the mask hardness needs resetting before
returning. This will be necessary if there is the possibility that the
operation being applied to the dask array could lose the "memory" on
its chunks of whether or not the mask is hard.

A common situation that causes a chunk to lose its memory of whether
or not the mask is hard is when a chunk could have contained a
unmasked `numpy` array prior to the operation, but the operation could
convert it to a masked `numpy` array. The new masked array will always
have the `numpy` default hardness (i.e. soft), which may be
incorrect.

The mask hardness is most easily reset with the
`cf.Data._reset_mask_hardness` method.

`cf.Data.__setitem__` and `cf.Data.where` are examples of methods that
need to reset the mask in this manner.


Laziness
--------

To *be* lazy, or *not to be* lazy (in `cf.Data` itself)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Central to Dask is lazy execution i.e. delayed computation:
Dask operations essentially construct a graph of calculations
or transformations (etc.) that are ready to run later,
and only get evaluated together when requested with
a `<dask object>.compute` call.

We want to utilise this laziness because it is central to the
efficiency from using Dask, but to what extent to do we want
to incorporate laziness into `cf.Data`? Namely, for
an arbitary `cf.Data` method previously returning some result
(say, a Boolean or an array), which of these should we return:

1. The **pre-computed result**, i.e. the outcome from running
   `compute` on the result graph constructed in the method
   (e.g. the same Boolean or an array, etc., as before); or
2. The **uncomputed result**, i.e. a Dask object which only
   evaluates to the result in (1) when either the user or
   the code under-the-hood, later, runs a `compute`?

Arguments for choice (1) [advantages to (1) and disadvantages to (2)]:

* The simpler choice:

  * means output is the same as before so documentation is easier and
    less change relative to previous versions;
  * logging and error handling can remain simple and as-is, whereas
    choice (2) would mean we don't know whether a given log or error
    message, dependent on the outcome, is applicable, so we can't
    call it immediately (perhaps at all?). We might have to defer to
    standard Dask messages, which would reduce specificity and clarity.
  * Testing will be simpler, as with (2) we would have to add `compute`
    calls in at appropriate points before running test assertions, etc.
  * Inspection methods can return as they do now, whereas with choice (2)
    we would have to work out what to show when certain aspects aren't
    yet computed.

Arguments for choice (2):

* The technically more complicated but more efficient choice, overall:

  * This choice is more efficient when we build up chains of operations,
    because it avoids intermediate computation meaning parallelisation can
    be optimised more comprehensively by Dask.

As well as choice (1) or (2) outright, there are further options for
a mixture or a flexible choice of return object in this respect:

3. Make use of a common keyword argument such as `precompute`
   on methods so users and under-the-hood in
   the code we can dictate whether or not to return the pre-computed or
   uncomputed result? That would give extra flexibility, but mean more
   boilerplate code (which can be consolidated somewhat, but at best
   will require some extra lines per method).

   If this option is chosen, what would the best default be, `True`
   or `False`?

4. (DH's suggestion) Methods that return new cf.Data objects
   (such as transpose) should be lazy and other methods should not be
   (e.g. __repr__ and equals).

**We have agreed that (4) is the most sensible approach to take, therefore
the working plan is** that:

* **any method (previously) returning a cf.Data object will,
  post-daskification, belazy and return the uncomputed result**, i.e. a
  Dask object that, when computed, will evaluate to the final cf.Data
  object (e.g. if computed immediately after the method runs, the result
  would be the same cf.Data object as that previously returned); but
* **any method returning another object, such as a Boolean or a string
  representation of the object, will not be lazy and
  return the pre-computed object as before**.


Logging and error handling
^^^^^^^^^^^^^^^^^^^^^^^^^^

When Dask operations are uncomputed, we don't know whether certain logging
and error messages are applicable or not.

Can we raise these in a delayed way, when we don't want to compute
early, in the case we are in the middle of under-the-hood operations and
also perhaps if we choose case (2) from the above points on extent of
laziness? How can it be done? Possible ideas include:

* Using a `try/except` block whenever a custom error message is required,
  catching the corresponding Dask errors and raising our own messages.


Inheritance from `cfdm`
-----------------------

Generally, how do we deal with optimisation for objects and logic inherited
from `cfdm`, since the current plan is not to Daskify `cfdm.Data`?

Returned Booleans
-----------------

When a method currently returns a Boolean (such as `Data.all`), should
it in fact return a lazy size 1 `Data` object?. The numpy and dask
`all` functions have an "axis" keyword that allows non-scalar outputs,
and a keepdims argument.
