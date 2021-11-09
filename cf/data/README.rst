`cf.Data` developer notes
=========================

Hardness of the mask
--------------------

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

There may be at least one other option:

3. Could we, temporariliy or permanently, make use of a common keyword
   argument such as `precompute` on methods so users and under-the-hood in
   the code we can dictate whether or not to return the pre-computed or
   uncomputed result? That would give extra flexibility, but mean more
   boilerplate code (which can be consolidated somewhat, but at best
   will require some extra lines per method).

   If so, what would the best default be, `True` or `False`?

I think we need to ensure we are consistent in our approach, so choose either
(1), (2) or (3) (or another alternative), rather than a mixture, as that
will be a maintenance nightmare!


Logging and error handling
^^^^^^^^^^^^^^^^^^^^^^^^^^

When Dask operations are uncomputed, we don't know whether certain logging
and error messages are applicable or not.

Can we raise these in a delayed way, when we don't want to compute
early, in the case we are in the middle of under-the-hood operations and
also perhaps if we choose case (2) from the above points on extent of
laziness? How can it be done?
