.. currentmodule:: cf
.. default-role:: obj


Collapse FAQs
=============

See the `cf.Field.collapse` documentation for full details.


Daily means
-----------

Create daily means from sub-daily data:

>>> g = f.collapse('T: mean', group=cf.D())

Create pentad means from sub-pentad data:

>>> g = f.collapse('T: mean', group=cf.D(5))
