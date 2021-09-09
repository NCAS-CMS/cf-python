`cf.Data` developer notes
=========================

Hardness of the mask
--------------------

Every `cf.Data` method that changes the dask array should consider
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

The mask hardness is reset as follows:

.. code-block:: python

   self._reset_mask_hardness()

See `cf.Data.__setitem__` for an example.
