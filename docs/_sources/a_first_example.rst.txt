.. currentmodule:: cf
.. default-role:: obj

A first example
===============

The cf package allows a data array and its associated metadata to be
contained and manipulated as a single entity called a *field*, which
is stored in a `cf.Field` object.

Much of the basic field manipulation syntax can be seen in this simple
read-modify-write example which:

    * Reads a field from a file on disk and find out information about
      it.

    * Modifies a CF property and the units of its data.

    * Modifies the data values.

    * Modifies a subspace of the data values.

    * Writes it out to another file on disk.

The example may be reproduced by downloading the :download:`sample
netCDF file (file.nc) <../file.nc>` (taking care not to overwrite an
existing file with that name) [#f1]_.

.. .. raw:: html
      :file: examples/first_example.html

..
   :download:`1 <examples/first_example.html>`
   :download:`2 <examples/regridc_example.html>`
   :download:`3 <examples/regrids_example_2.html>`
   :download:`4 <examples/regrids_example.html>`


**1.** Import the cf package.

>>> import cf

**2.** Read a field from disk and find a summary of its contents.

>>> f = cf.read_field('file.nc')
>>> f
<CF Field: air_temperature(latitude(4), longitude(5)) K>
>>> print f
Field: air_temperature (ncvar%tas)
----------------------------------
Data           : air_temperature(latitude(4), longitude(5)) K
Cell methods   : time: mean
Axes           : time(1) = [2000-01-16T00:00:00Z] 360_day
               : height(1) = [2.0] m
               : latitude(4) = [-2.5, ..., 5.0] degrees_north
               : longitude(5) = [0.0, ..., 15.0] degrees_east

**3.** Find all of the field's CF properties and its data array as a
numpy array.

>>> f.properties()
{'Conventions': 'CF-1.5',
 '_FillValue': 1e+20,
 'experiment_id': 'stabilization experiment (SRES A1B)',
 'long_name': 'Air Temperature',
 'standard_name': 'air_temperature',
 'title': 'SRES A1B'}
>>> print f.array
[[ 274.15,  276.15,  275.15,  277.15,  278.15],
 [ 274.15,  275.15,  276.15,  277.15,  276.15],
 [ 277.15,  275.15,  278.15,  274.15,  278.15],
 [ 276.15,  274.15,  275.15,  277.15,  274.15]]

**4.**. Modify the field's long name CF property and change the field's
data from units of Kelvin to Celsius.

.. note:: Changing the units automatically changes the data when it is
          next accessed.

>>> f.long_name
'Air Temperature'
>>> f.long_name = 'Surface Air Temperature'
>>> f.long_name
'Surface Air Temperature'
>>> print f.Units
K
>>> f.Units -= 273.15
>>> print f.Units       
K @ 273.15
>>> print f.array
[[ 1.  3.  2.  4.  5.]
 [ 1.  2.  3.  4.  3.]
 [ 4.  2.  5.  1.  5.]
 [ 3.  1.  2.  4.  1.]]

**5.** Check that the field has 'air_temperature' as its standard name
       and that at least one of its latitude coordinate values is
       greater than 0.0.

>>> f.match('air_temperature', items={'latitude' : cf.gt(0)})
True

**6.** Modify the data values.

>>> g = f + 100
>>> print g.array
[[ 101.  103.  102.  104.  105.]
 [ 101.  102.  103.  104.  103.]
 [ 104.  102.  105.  101.  105.]
 [ 103.  101.  102.  104.  101.]]
>>> g = f + f
>>> print g.array
[[  2.   6.   4.   8.  10.]
 [  2.   4.   6.   8.   6.]
 [  8.   4.  10.   2.  10.]
 [  6.   2.   4.   8.   2.]]
>>> g = f / cf.Data(2, 'seconds')
>>> print g.array
[[0.5 1.5 1.0 2.0 2.5]
 [0.5 1.0 1.5 2.0 1.5]
 [2.0 1.0 2.5 0.5 2.5]
 [1.5 0.5 1.0 2.0 0.5]]
>>> print g.Units
s-1.K

**7.** Access and modify a subspace of the data values.

>>> g = f[0::2, 2:4]
>>> print g.array
[[ 2.  4.]
 [ 5.  1.]]
>>> f.subspace(longitude=0)
>>> print f.array
[[ 274.15,
 [ 274.15,
 [ 277.15,
 [ 276.15]]
>>> f[0::2, ...] = -10
>>> print f.array
[[-10. -10. -10.  -10.  -10.]
 [  1.   2.   3.    4.    3.]
 [-10. -10. -10.  -10.  -10.]
 [  3.   1.   2.    4.    1.]]

**8.** Write the modified field to disk.

>>> cf.write(f, 'newfile.nc')

----

.. rubric:: Footnotes

.. [#f1] This file may be also found in the ``docs/build/_downloads/``
         directory of the installation.

