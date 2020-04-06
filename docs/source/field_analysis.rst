.. currentmodule:: cf
.. default-role:: obj

.. _Statistical-collapses:

**Statistical collapses**
-------------------------

Collapsing one or more dimensions reduces their size and replaces the
data along those axes with representative statistical values. The
result is a new field construct with consistent metadata for the
collapsed values. Collapses are carried with the `~Field.collapse`
method of the field construct.

By default all axes with size greater than 1 are collapsed completely
(i.e. to size 1) with a given :ref:`collapse method
<Collapse-methods>`.

.. code-block:: python
   :caption: *Find the minimum of the entire data.*

   >>> a = cf.read('timeseries.nc')[0]
   >>> print(a)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
   Cell methods    : area: mean
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> b = a.collapse('minimum')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(1), latitude(1), longitude(1)) K
   Cell methods    : area: mean time(1): latitude(1): longitude(1): minimum
   Dimension coords: time(1) = [1964-11-30 12:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(b.array)
   [[[198.9]]]

In the above example, note that the operation has been recorded in a
new cell method construct (``time(1): latitude(1): longitude(1):
minimum``) in the output field construct, and the dimension coordinate
constructs each now have a single cell. The air pressure time
dimension was not included in the collapse because it already had size
1 in the original field construct.

The collapse can also be applied to any subset of the field
construct's dimensions. In this case, the domain axis and coordinate
constructs for the non-collapsed dimensions remain the same. This is
implemented either with the *axes* keyword, or with a `CF-netCDF cell
methods`_\ -like syntax for describing both the collapse dimensions
and the collapse method in a single string. The latter syntax uses
:ref:`construct identities <Construct-identities>` instead of netCDF
dimension names to identify the collapse axes.

Statistics may be created to represent variation over one dimension or
a combination of dimensions.

.. code-block:: python
   :caption: *Two equivalent techniques for creating a field construct
             of temporal maxima at each horizontal location.*

   >>> b = a.collapse('maximum', axes='T')
   >>> b = a.collapse('T: maximum')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(1), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(1): maximum
   Dimension coords: time(1) = [1964-11-30 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa   
   >>> print(b.array)
   [[[310.6 309.1 309.9 311.2 310.4 310.1 310.7 309.6]
     [310.  310.7 311.1 311.3 310.9 311.2 310.6 310. ]
     [308.9 309.8 311.2 311.2 311.2 309.3 311.1 310.7]
     [310.1 310.3 308.8 311.1 310.  311.3 311.2 309.7]
     [310.9 307.9 310.3 310.4 310.8 310.9 311.3 309.3]]]

.. code-block:: python
   :caption: *Find the horizontal maximum, with two equivalent
             techniques.*

   >>> b = a.collapse('maximum', axes=['X', 'Y'])
   >>> b = a.collapse('X: Y: maximum')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(1), longitude(1)) K
   Cell methods    : area: mean latitude(1): longitude(1): maximum
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa

Variation over horizontal area may also be specified by the special
identity ``'area'``. This may be used for any horizontal coordinate
reference system.

.. code-block:: python
   :caption: *Find the horizontal maximum using the special identity
             'area'.*

   >>> b = a.collapse('area: maximum')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(1), longitude(1)) K
   Cell methods    : area: mean area: maximum
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa
					      
.. _Collapse-methods:    
    
Collapse methods
^^^^^^^^^^^^^^^^

The following collapse methods are available, over any subset of the
domain axes. The "Cell method" column in the table gives the method of
the new cell method construct (if one is created).

============================  ========================================  ==========================
Method                        Description                               Cell method
============================  ========================================  ==========================
``'maximum'``                 The maximum of the values.                ``maximum``
                          
``'minimum'``                 The minimum of the values.                ``minimum``
                                   
``'maximum_absolute_value'``  The maximum of the absolute values.       ``maximum_absolute_value``
                          
``'minimum_absolute_value'``  The minimum of the absolute values.       ``minimum_absolute_value``
                          
``'mid_range'``               The average of the maximum and the        ``mid_range``
                              minimum of the values.
                              
``'range'``                   The absolute difference between the       ``range``
                              maximum and the minimum of the values.
			      
``'median'``                  The median of the values.                 ``median`` 

``'sample_size'``             The sample size, :math:`N`, as would be   ``point``
                              used for other calculations, i.e. the
			      number of non-missing values.
                              
``'sum_of_weights'``          The sum of :math:`N` weights              ``sum``
                              :math:`w_i`, as would be used for other
                              calculations, is

			      .. math:: V_{1}=\sum_{i=1}^{N} w_i
			      
``'sum_of_weights2'``         The sum of the squares of :math:`N`       ``sum``
                              weights :math:`w_i`, as would be used
                              for other calculations, is

			      .. math:: V_{2}=\sum_{i=1}^{N}  w_i^{2}

``'sum'``                     The unweighted sum of :math:`N` values    ``sum``
                              :math:`x_i` is
			      
                              .. math:: t=\sum_{i=1}^{N} x_i

``'sum_of_squares'``          The unweighted sum of the squares of      ``sum_of_squares``
                              :math:`N` values :math:`x_i` is
			      
                              .. math:: t_2=\sum_{i=1}^{N} x_{i}^{2}
                              
``'integral'``                The integral of :math:`N` values          ``sum``
                              :math:`x_i` with corresponding cell       
                              measures :math:`m_i` is
			      
                              .. math:: i=\sum_{i=1}^{N} m_i x_i

			      Note that the integral differs from a
			      weighted sum in that the units of the
			      cell measures are incorporated into the
			      result.
			      
``'mean'``                    The unweighted mean of :math:`N` values   ``mean``
                              :math:`x_i` is
                              
                              .. math:: \mu=\frac{1}{N}\sum_{i=1}^{N}
					               x_i
                              
                              The :ref:`weighted <Collapse-weights>`
                              mean of :math:`N` values :math:`x_i`
                              with corresponding weights :math:`w_i`
                              is
			      
                              .. math:: \hat{\mu}=\frac{1}{V_{1}}
                                                    \sum_{i=1}^{N}
                                         w_i x_i
					
``'mean_absolute_value'``     The unweighted mean of :math:`N`          ``mean_absolute_value``
                              values :math:`x_i` absoluted is 
                              
                              .. math:: \mu_{abs}=\frac{1}{N}
				       \sum_{i=1}^{N}|x_i|
                              
                              The :ref:`weighted <Collapse-weights>`
                              mean of :math:`N` values :math:`x_i`
                              absoluted with corresponding weights
                              :math:`w_i` is
			      
                              .. math:: \hat{\mu}_{abs}=
                                             \frac{1}{V_{1}}
					     \sum_{i=1}^{N} w_i |x_i|

``'mean_of_upper_decile'``    The weighted or unweighted mean of the    ``mean_of_upper_decile``
                              upper group of data values defined by
                              the upper tenth of their distribution

``'variance'``                The unweighted variance of :math:`N`      ``variance``
                              values :math:`x_i` and with 
                              :math:`N-ddof` degrees of freedom
   			      (:math:`ddof\ge0`) is
			      
                              .. math:: s_{N-ddof}^{2}=
          		                \frac{1}{N-ddof}
                                        \sum_{i=1}^{N} (x_i - \mu)^2
			      
                              The unweighted biased estimate of the
                              variance (:math:`s_{N}^{2}`) is given by
                              :math:`ddof=0` and the unweighted
                              unbiased estimate of the variance using
                              Bessel's correction
                              (:math:`s^{2}=s_{N-1}^{2}`) is given by
                              :math:`ddof=1`.
			      
                              The :ref:`weighted <Collapse-weights>`
                              biased estimate of the variance of
                              :math:`N` values :math:`x_i` with
                              corresponding weights :math:`w_i` is
			      
                              .. math:: \hat{s}_{N}^{2}=
					\frac{1}{V_{1}}
                                        \sum_{i=1}^{N}
                                        w_i(x_i -
                                        \hat{\mu})^{2}
                                   
                              The corresponding :ref:`weighted
                              <Collapse-weights>` unbiased estimate of
                              the variance is
                              
                              .. math:: \hat{s}^{2}=\frac{1}{V_{1} -
                                                    (V_{1}/V_{2})}
                                                    \sum_{i=1}^{N}
                                                    w_i(x_i -
                                                    \hat{\mu})^{2}
			      
                              In both cases, the weights are assumed
                              to be non-random reliability weights, as
                              opposed to frequency weights.
                                  
``'standard_deviation'``      The standard deviation is the square      ``standard_deviation``
                              root of the unweighted or
			      :ref:`weighted <Collapse-weights>`
		              variance, as defined in this table.
			       
``'root_mean_square'``        The unweighted root mean square of        ``root_mean_square`` 
                              :math:`N` values :math:`x_i` is
			      
                              .. math:: RMS=\sqrt{\frac{1}{N}
				                   \sum_{i=1}^{N}
				                   x_{i}^2}
			      
                              The :ref:`weighted <Collapse-weights>`
                              root mean square of :math:`N` values
                              :math:`x_i` with corresponding weights
                              :math:`w_i` is
			      
                              .. math:: \hat{RMS}=\sqrt{
				              \frac{1}{V_{1}}
				              \sum_{i=1}^{N} w_i
				              x_{i}^2}			      
============================  ========================================  ==========================

.. _Data-type-and-missing-data:

Data type and missing data
^^^^^^^^^^^^^^^^^^^^^^^^^^

In all collapses, missing data array elements are accounted for in the 
calculation.

Any collapse method that involves a calculation (such as calculating a
mean), as opposed to just selecting a value (such as finding a
maximum), will return a field containing double precision floating
point numbers. If this is not desired then the data type can be reset
after the collapse with the `~Field.dtype` attribute of the field
construct.

.. _Collapse-weights:

Collapse weights
^^^^^^^^^^^^^^^^

.. The calculations of means, standard deviations and variances are,
   by default, not weighted. For weights to be incorporated in the
   collapse, the axes to be weighted must be identified with the
   *weights* keyword.

For weights to be incorporated in the collapse, the axes to be
weighted must be identified with the *weights* keyword. A collapse by
a particular method is either never weighted, or may be weighted, or
is always weighted, as described in the following table:

============================  ============================  ========
Method                        Description                   Weighted  
============================  ============================  ========
``'maximum'``                 The maximum of the values.    Never
                          
``'minimum'``                 The minimum of the values.    Never

``'maximum_absolute_value'``  The maximum of the absolute.  Never
              
``'minimum_absolute_value'``  The minimum of the absolute.  Never

``'mid_range'``               The average of the maximum    Never
                              and the minimum of the
                              values.
                              
``'range'``                   The absolute difference       Never
                              between the maximum and the
                              minimum of the values.
                              
``'median'``                  The median of the values.     Never
                              
``'sum'``                     The sum of the values.        Never
                                                                        
``'sum_of_squares'``          The sum of the squares of     Never
                              values.
                              
``'sample_size'``             The sample size, i.e. the     Never
                              number of non-missing
                              values.

``'sum_of_weights'``          The sum of weights, as        Never
                              would be used for other
                              calculations.
                              
``'sum_of_weights2'``         The sum of squares of         Never
                              weights, as would be used
                              for other calculations.

``'mean'``                    The weighted or unweighted    May be
                              mean of the values.

``'mean_absolute_value'``     The mean of the absolute      May be
                              values.
			      
``'mean_of_upper_decile'``    The mean of the upper group   May be
                              of data values defined by
			      the upper tenth of their
			      distribution.
                              
``'variance'``                The weighted or unweighted    May be
                              variance of the values, with
                              a given number of degrees of
                              freedom.
                                  
``'standard_deviation'``      The weighted or unweighted    May be
                              standard deviation of the
			      values with a given number
			      of degrees of freedom.
                              
``'root_mean_square'``        The square root of the        May be
                              weighted or unweighted mean
                              of the squares of the
                              values.
                              
``'integral'``                The integral of values.       Always
============================  ============================  ========


* Collapse methods that are "Never" weighted ignore the *weights*
  parameter, even if it is set.

* Collapse methods that "May be" weighted will only be weighted if the
  *weights* parameter is set.

* Collapse methods that are "Always" weighted require the *weights*
  parameter to be set.

Weights are either derived from the field construct's metadata (such
as cell sizes), or may be provided explicitly in the form of other
field constructs containing data of weights values. In either case,
the weights actually used are those derived by the `~Field.weights`
method of the field construct called with the same *weights* keyword
value. Collapsed axes that are not identified by the *weights* keyword
are unweighted during the collapse operation.

.. code-block:: python
   :caption: *Create a weighted time average.*
 	     
   >>> b = a.collapse('T: mean', weights=True)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(1), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(1): mean
   Dimension coords: time(1) = [1964-11-30 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print (b.array)
   [[[254.03120723 255.89723515 253.06490556 254.17815494 255.18458801 253.3684369  253.26624692 253.63818779]
     [248.92058582 253.99597591 259.67957843 257.08967972 252.57333698 252.5746236  258.90938954 253.86939502]
     [255.94716671 254.77330961 254.35929373 257.91478237 251.87670408 252.72723789 257.26038872 258.19698878]
     [258.08639474 254.8087873  254.9881741  250.98064604 255.3513003  256.66337257 257.86895702 259.49299206]
     [263.80016425 253.35825349 257.8026006  254.3173556  252.2061867  251.74150014 255.60930742 255.06260608]]]

To inspect the weights, call the  `~Field.weights` method directly.

.. code-block:: python
   :caption: *Create and view weights derived from the field construct’s
             time axis.*
 	     
   >>> w = a.weights(True)
   >>> print(w)
   Field: long_name=weights (ncvar%air_potential_temperature)
   ----------------------------------------------------------
   Data            : long_name=weights(time(120)) d
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]   
   >>> print(w.array)
   [31. 31. 29. 31. 30. 31. 30. 31. 31. 30. 31. 30. 31. 31. 28. 31. 30. 31.
    30. 31. 31. 30. 31. 30. 31. 31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30.
    31. 31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30. 31. 31. 29. 31. 30. 31.
    30. 31. 31. 30. 31. 30. 31. 31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30.
    31. 31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30. 31. 31. 28. 31. 30. 31.
    30. 31. 31. 30. 31. 30. 31. 31. 29. 31. 30. 31. 30. 31. 31. 30. 31. 30.
    31. 31. 28. 31. 30. 31. 30. 31. 31. 30. 31. 30.]

.. code-block:: python
   :caption: *Calculate the mean over the time and latitude axes, with
             weights only applied to the latitude axis.*
 	     
   >>> b = a.collapse('T: Y: mean', weights='Y')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(1), latitude(1), longitude(8)) K
   Cell methods    : area: mean time(1): latitude(1): mean
   Dimension coords: time(1) = [1964-11-30 12:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print (b.array)
   [[[256.15819444 254.625      255.73666667 255.43041667 253.19444444 253.31277778 256.68236111 256.42055556]]]

Specifying weighting by horizontal cell area may also use the special
``'area'`` syntax.
   
.. code-block:: python
   :caption: *Alternative syntax for specifying area weights.*
 	     
   >>> b = a.collapse('area: mean', weights=True)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(1), longitude(1)) K
   Cell methods    : area: mean area: mean
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa

An alternative technique for specifying weights is to set the
*weights* keyword to the output of a call to the `~Field.weights`
method.
		   
.. code-block:: python
   :caption: *Alternative syntax for specifying weights.*
 	     
   >>> b = a.collapse('area: mean', weights=a.weights('area'))
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(1), longitude(1)) K
   Cell methods    : area: mean area: mean
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa

See the `~Field.weights` method for full details on how weights may be
specified.

.. _Multiple-collapses:

Multiple collapses
^^^^^^^^^^^^^^^^^^

Multiple collapses normally require multiple calls to
`~Field.collapse`: one on the original field construct and then one on
each interim field construct.

.. code-block:: python
   :caption: *Calculate the temporal maximum of the weighted areal
             means using two independent calls.*

   >>> b = a.collapse('area: mean', weights=True).collapse('T: maximum')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(1), latitude(1), longitude(1)) K
   Cell methods    : area: mean latitude(1): longitude(1): mean time(1): maximum
   Dimension coords: time(1) = [1964-11-30 12:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(b.array)
   [[[271.77199724]]]

If preferred, multiple collapses may be carried out in a single call
to `~Field.collapse` by using the `CF-netCDF cell methods`_\ -like
syntax (note that the colon (``:``) is only used after the construct
identity that specifies each axis, and a space delimits the separate
collapses).
   
.. code-block:: python
   :caption: *Calculate the temporal maximum of the weighted areal
             means in a single call, using the cf-netCDF cell
             methods-like syntax.*

   >>> b = a.collapse('area: mean T: maximum', weights=True)
   >>> print(b.array)
   [[[271.77199724]]]

.. _Grouped-collapses:

Grouped collapses
^^^^^^^^^^^^^^^^^

A grouped collapse is one for which an axis is not collapsed
completely to size 1. Instead the collapse axis is partitioned into
non-overlapping groups and each group is collapsed to size 1. The
resulting axis will generally have more than one element. For example,
creating 12 annual means from a timeseries of 120 months would be a
grouped collapse. The groups do not need to be created from adjacent
cells, as would be the case when creating 12 multi-annual monthly
means from a timeseries of 120 months.

Selected statistics for overalapping groups can be calculated with the
`~cf.Field.moving_window` method of the field construct.

The *group* keyword of `~Field.collapse` defines the size of the
groups. Groups can be defined in a variety of ways, including with
`cf.Query`, `cf.TimeDuration` (see the :ref:`Time-duration` section)
and `cf.Data` instances.

An element of the collapse axis can not be a member of more than one
group, and may be a member of no groups. Elements that are not
selected by the *group* keyword are excluded from the result.

.. code-block:: python
   :caption: *Create annual maxima from a time series, defining a year
             to start on 1st January.*

   >>> y = cf.Y(month=12)
   >>> y
   <CF TimeDuration: P1Y (Y-12-01 00:00:00)>
   >>> b = a.collapse('T: maximum', group=y)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(10), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(10): maximum
   Dimension coords: time(10) = [1960-06-01 00:00:00, ..., 1969-06-01 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   
.. code-block:: python
   :caption: *Find the maximum of each group of 6 elements along an
             axis.*
	     
   >>> b = a.collapse('T: maximum', group=6)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(20), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(20): maximum
   Dimension coords: time(20) = [1960-03-01 12:00:00, ..., 1969-08-31 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa

.. code-block:: python
   :caption: *Create December, January, February maxima from a time series.*

   >>> b = a.collapse('T: maximum', group=cf.djf())
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(10), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(10): maximum time(10): maximum
   Dimension coords: time(10) = [1960-01-15 12:00:00, ..., 1969-01-15 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa

.. code-block:: python
   :caption: *Create maxima for each 3-month season of a timeseries
             (DJF, MAM, JJA, SON).*

   >>> c = cf.seasons()
   >>> c
   [<CF Query: month[(ge 12) | (le 2)]>
    <CF Query: month(wi (3, 5))>,
    <CF Query: month(wi (6, 8))>,
    <CF Query: month(wi (9, 11))>]
   >>> b = a.collapse('T: maximum', group=c)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(40), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(40): maximum time(40): maximum
   Dimension coords: time(40) = [1960-01-15 12:00:00, ..., 1969-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa

.. code-block:: python
   :caption: *Calculate zonal means for the western and eastern
             hemispheres.*
	     
   >>> b = a.collapse('X: mean', group=cf.Data(180, 'degrees'))
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(5), longitude(2)) K
   Cell methods    : area: mean longitude(2): mean longitude(2): mean
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(2) = [90.0, 270.0] degrees_east
                   : air_pressure(1) = [850.0] hPa

Groups can be further described with the *group_span* (to include
groups whose actual span is not equal to a given value) and the
*group_contiguous* (to include non-contiguous groups, or any
contiguous group containing overlapping cells) keywords of
`~Field.collapse`.

.. _Climatological-statistics:

Climatological statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

`Climatological statistics`_ may be derived from corresponding
portions of the annual cycle in a set of years (e.g. the average
January temperatures in the climatology of 1961-1990, where the values
are derived by averaging the 30 Januarys from the separate years); or
from corresponding portions of the diurnal cycle in a set of days
(e.g. the average temperatures for each hour in the day for May
1997). A diurnal climatology may also be combined with a multiannual
climatology (e.g. the minimum temperature for each hour of the average
day in May from a 1961-1990 climatology).

Calculation requires two or three collapses, depending on the quantity
being created, all of which are grouped collapses. Each collapse
method needs to indicate its climatological nature with one of the
following qualifiers,

================  =======================
Method qualifier  Associated keyword
================  =======================
``within years``  *within_years*
``within days``   *within_days*
``over years``    *over_years* (optional)
``over days``     *over_days* (optional)
================  =======================

and the associated keyword to `~Field.collapse` specifies how the
method is to be applied.

.. code-block:: python
   :caption: *Calculate the multiannual average of the seasonal means.*
	     
   >>> b = a.collapse('T: mean within years T: mean over years',
   ...                within_years=cf.seasons(), weights=True)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(4), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(4): mean within years time(4): mean over years
   Dimension coords: time(4) = [1960-01-15 12:00:00, ..., 1960-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(b.coordinate('T').bounds.datetime_array)
   [[cftime.DatetimeGregorian(1959-12-01 00:00:00) cftime.DatetimeGregorian(1969-03-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-03-01 00:00:00) cftime.DatetimeGregorian(1969-06-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-06-01 00:00:00) cftime.DatetimeGregorian(1969-09-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-09-01 00:00:00) cftime.DatetimeGregorian(1969-12-01 00:00:00)]]
		   
.. code-block:: python
   :caption: *Calculate the multiannual variance of the seasonal
             minima. Note that the units of the result have been
             changed from 'K' to 'K2'.*
	     
   >>> b = a.collapse('T: minimum within years T: variance over years',
   ...                within_years=cf.seasons(), weights=True)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(4), latitude(5), longitude(8)) K2
   Cell methods    : area: mean time(4): minimum within years time(4): variance over years
   Dimension coords: time(4) = [1960-01-15 12:00:00, ..., 1960-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(b.coordinate('T').bounds.datetime_array)
   [[cftime.DatetimeGregorian(1959-12-01 00:00:00) cftime.DatetimeGregorian(1969-03-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-03-01 00:00:00) cftime.DatetimeGregorian(1969-06-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-06-01 00:00:00) cftime.DatetimeGregorian(1969-09-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-09-01 00:00:00) cftime.DatetimeGregorian(1969-12-01 00:00:00)]]
		      
When collapsing over years, it is assumed by default that the each
portion of the annual cycle is collapsed over all years that are
present. This is the case in the above two examples. It is possible,
however, to restrict the years to be included, or group them into
chunks, with the *over_years* keyword to `~Field.collapse`.

.. code-block:: python
   :caption: *Calculate the multiannual average of the seasonal means
             in 5 year chunks.*
	     
   >>> b = a.collapse('T: mean within years T: mean over years', weights=True,
   ...                within_years=cf.seasons(), over_years=cf.Y(5))
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(8), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(8): mean within years time(8): mean over years
   Dimension coords: time(8) = [1960-01-15 12:00:00, ..., 1965-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(b.coordinate('T').bounds.datetime_array)
   [[cftime.DatetimeGregorian(1959-12-01 00:00:00) cftime.DatetimeGregorian(1964-03-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-03-01 00:00:00) cftime.DatetimeGregorian(1964-06-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-06-01 00:00:00) cftime.DatetimeGregorian(1964-09-01 00:00:00)]
    [cftime.DatetimeGregorian(1960-09-01 00:00:00) cftime.DatetimeGregorian(1964-12-01 00:00:00)]
    [cftime.DatetimeGregorian(1964-12-01 00:00:00) cftime.DatetimeGregorian(1969-03-01 00:00:00)]
    [cftime.DatetimeGregorian(1965-03-01 00:00:00) cftime.DatetimeGregorian(1969-06-01 00:00:00)]
    [cftime.DatetimeGregorian(1965-06-01 00:00:00) cftime.DatetimeGregorian(1969-09-01 00:00:00)]
    [cftime.DatetimeGregorian(1965-09-01 00:00:00) cftime.DatetimeGregorian(1969-12-01 00:00:00)]]


.. code-block:: python
   :caption: *Calculate the multiannual average of the seasonal means,
             restricting the years from 1963 to 1968.*

   >>> b = a.collapse('T: mean within years T: mean over years', weights=True,
   ...                within_years=cf.seasons(), over_years=cf.year(cf.wi(1963, 1968)))
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(4), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(4): mean within years time(4): mean over years
   Dimension coords: time(4) = [1963-01-15 00:00:00, ..., 1963-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa   
   >>> print(b.coordinate('T').bounds.datetime_array)
   [[cftime.DatetimeGregorian(1962-12-01 00:00:00) cftime.DatetimeGregorian(1968-03-01 00:00:00)]
    [cftime.DatetimeGregorian(1963-03-01 00:00:00) cftime.DatetimeGregorian(1968-06-01 00:00:00)]
    [cftime.DatetimeGregorian(1963-06-01 00:00:00) cftime.DatetimeGregorian(1968-09-01 00:00:00)]
    [cftime.DatetimeGregorian(1963-09-01 00:00:00) cftime.DatetimeGregorian(1968-12-01 00:00:00)]]

Similarly for collapses over days, it is assumed by default that the
each portion of the diurnal cycle is collapsed over all days that are
present, But it is possible to restrict the days to be included, or
group them into chunks, with the *over_days* keyword to
`~Field.collapse`.

The calculation can be done with multiple collapse calls, which can be
useful if the interim stages are needed independently, but be aware
that the interim field constructs will have non-CF-compliant cell
method constructs.
		   
.. code-block:: python
   :caption: *Calculate the multiannual maximum of the seasonal
             standard deviations with two separate collapse calls.*

   >>> b = a.collapse('T: standard_deviation within years',
   ...                within_years=cf.seasons(), weights=True)
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(40), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(40): standard_deviation within years
   Dimension coords: time(40) = [1960-01-15 12:00:00, ..., 1969-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa   
   >>> c = b.collapse('T: maximum over years')
   >>> print(c)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(4), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(4): standard_deviation within years time(4): maximum over years
   Dimension coords: time(4) = [1960-01-15 12:00:00, ..., 1960-10-16 12:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa

----
   
.. _ Other-statistical-operations:

**Other statistical operations**
--------------------------------

.. _Cumulative-sums:

Cumulative sums
^^^^^^^^^^^^^^^

The `~Field.cumsum` method of the field construct calculates the
cumulative sum of elements along a given axis. The cell bounds of the
axis are updated to describe the ranges over which the sums apply, and
a new ``sum`` cell method construct is added to the resulting field
construct.

.. code-block:: python
   :caption: *Calculate cumulative sums along the "T" axis, showing
             the cell bounds before and after the operation.*
   
   >>> a = cf.read('timeseries.nc')[0]
   >>> print(a)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
   Cell methods    : area: mean
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> b = a.cumsum('T')
   >>> print(b)
   Field: air_potential_temperature (ncvar%air_potential_temperature)
   ------------------------------------------------------------------
   Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
   Cell methods    : area: mean time(120): sum
   Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : air_pressure(1) = [850.0] hPa
   >>> print(a.coordinate('T').bounds[-1].dtarray)
   [[cftime.DatetimeGregorian(1969-11-01 00:00:00)
     cftime.DatetimeGregorian(1969-12-01 00:00:00))]]
   >>> print(b.coordinate('T').bounds[-1].dtarray)
   [[cftime.DatetimeGregorian(1959-11-01 00:00:00)
     cftime.DatetimeGregorian(1969-12-01 00:00:00))]]

The treatment of missing values can be specified, as well as the
positioning of coordinate values in the summed axis of the returned
field construct.

.. _Histograms:

Histograms
^^^^^^^^^^

The `cf.histogram` function is used to record the distribution of a
set of variables in the form of an N-dimensional histogram.

Each dimension of the histogram is defined by a field construct
returned by the `~Field.digitize` method of a field construct. This
"digitized" field construct defines a sequence of bins and provides
indices to the bins that each value of one of the variables belongs.

.. code-block:: python
   :caption: *Create a one-dimensional histogram of a field construct
             based on 10 equally-sized bins that exactly span the data
             range.*
   
   >>> q, t = cf.read('file.nc')     
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]       
   >>> print(q.array)
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
   >>> indices, bins = q.digitize(10, return_bins=True)
   >>> print(indices)
   Field: long_name=Bin index to which each 'specific_humidity' value belongs (ncvar%q)
   ------------------------------------------------------------------------------------
   Data            : long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_eastg
                   : time(1) = [2019-01-01 00:00:00]
   >>> print(indices.array)
   [[0 2 0 0 1 2 1 1]
    [1 2 2 4 3 4 0 4]
    [7 8 8 9 5 6 3 0]
    [1 3 2 4 3 4 0 0]
    [0 2 1 2 1 2 2 0]]
   >>> print(bins.array)
   [[0.003  0.0173]
    [0.0173 0.0316]
    [0.0316 0.0459]
    [0.0459 0.0602]
    [0.0602 0.0745]
    [0.0745 0.0888]
    [0.0888 0.1031]
    [0.1031 0.1174]
    [0.1174 0.1317]
    [0.1317 0.146 ]]
   >>> h = cf.histogram(indices)                             
   >>> print(h) 
   Field: number_of_observations
   -----------------------------
   Data            : number_of_observations(specific_humidity(10)) 1
   Cell methods    : latitude: longitude: point
   Dimension coords: specific_humidity(10) = [10.15, ..., 138.85000000000002] 1
   >>> print(h.array)
   [9 7 9 4 5 1 1 1 2 1]
   >>> print(h.coordinate('specific_humidity').bounds.array)
   [[0.003  0.0173]
    [0.0173 0.0316]
    [0.0316 0.0459]
    [0.0459 0.0602]
    [0.0602 0.0745]
    [0.0745 0.0888]
    [0.0888 0.1031]
    [0.1031 0.1174]
    [0.1174 0.1317]
    [0.1317 0.146 ]]


.. code-block:: python
   :caption: *Create a two-dimensional histogram based on specific
             humidity and temperature bins. The temperature bins in
             this example are derived from a dummy temperature field
             construct with the same shape as the specific humidity
             field construct already in use.*

   >>> g = q.copy()
   >>> g.standard_name = 'air_temperature'
   >>> import numpy
   >>> g[...] = numpy.random.normal(loc=290, scale=10, size=40).reshape(5, 8)
   >>> g.override_units('K', inplace=True)
   >>> print(g)
   Field: air_temperature (ncvar%q)
   --------------------------------
   Data            : air_temperature(latitude(5), longitude(8)) K
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> indices_t = g.digitize(5)
   >>> h = cf.histogram(indices, indices_t)
   >>> print(h)
   Field: number_of_observations
   -----------------------------
   Data            : number_of_observations(air_temperature(5), specific_humidity(10)) 1
   Cell methods    : latitude: longitude: point
   Dimension coords: air_temperature(5) = [281.1054839143287, ..., 313.9741786365939] K
                   : specific_humidity(10) = [0.01015, ..., 0.13885] 1
   >>> print(h.array)
   [[2  1  5  3  2 -- -- -- -- --]
    [1  1  2 --  1 --  1  1 -- --]
    [4  4  2  1  1  1 -- --  1  1]
    [1  1 -- --  1 -- -- --  1 --]
    [1 -- -- -- -- -- -- -- -- --]]
   >>> h.sum()
   <CF Data(): 40 1>

.. _Binning-operations:

Binning operations
^^^^^^^^^^^^^^^^^^

The `~Field.bin` method of the field construct groups its data into
bins, where each group is defined by the elements that correspond to
an :ref:`N-dimensional histogram bin of another set of variables
<Histograms>`, and collapses the elements in each group to a single
representative value. The same :ref:`collapse methods
<Collapse-methods>` and :ref:`weighting options <Collapse-weights>` as
the `~Field.collapse` method are available.

The result of the binning operation is a field construct whose domain
axis and dimension coordinate constructs describe the sizes of the
N-dimensional bins of the other set of variables.

.. code-block:: python
   :caption: *Find the range of values that lie in each of bin 10
             equally-sized bins of the data itself.*

   >>> q, t = cf.read('file.nc')     
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 0.001 1
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]       
   >>> print(q.array)
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
   >>> indices = q.digitize(5)                                             
   >>> b = q.bin('range', digitized=indices)                             
   >>> print(b)                                    
   Field: specific_humidity
   ------------------------
   Data            : specific_humidity(specific_humidity(5)) 1
   Cell methods    : latitude: longitude: range
   Dimension coords: specific_humidity(5) = [0.0173, ..., 0.1317] 1
   >>> print(b.array)
   [0.026 0.025 0.025 0.007 0.022]  
   >>> print(b.coordinate('specific_humidity').bounds.array)
   [[0.003  0.0316]
    [0.0316 0.0602]
    [0.0602 0.0888]
    [0.0888 0.1174]
    [0.1174 0.146 ]]

.. code-block:: python
   :caption: *Find the area-weighted mean of specific humidity values
             that correspond to two-dimensional bins defined by
             temperature and pressure values.*

   >>> p, t = cf.read('file2.nc')
   >>> print(t)
   Field: air_temperature (ncvar%t)
   --------------------------------
   Data            : air_temperature(latitude(5), longitude(8)) degreesC
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> print(p)      
   Field: air_pressure (ncvar%p)
   -----------------------------
   Data            : air_pressure(latitude(5), longitude(8)) hPa
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> t_indices = t.digitize(4)
   >>> p_indices = p.digitize(6)
   >>> b = q.bin('mean', digitized=[t_indices, p_indices], weights='area')
   >>> print(b)
   Field: specific_humidity
   ------------------------
   Data            : specific_humidity(air_pressure(6), air_temperature(4)) 1
   Cell methods    : latitude: longitude: mean
   Dimension coords: air_pressure(6) = [966.6225003326126, ..., 1033.6456080043665] hPa
                   : air_temperature(4) = [-12.735821567738295, ..., 9.9702610462581] degreesC
   >>> print(b.array)
   [[     --       --       --  0.011  ]
    [0.131    0.0145   0.0345   0.05052]
    [0.05742  0.01727  0.06392  0.0105 ]
    [     --  0.04516  0.05272  0.10194]
    [0.124    0.024    0.059    0.006  ]
    [     --  0.08971       --       --]]


.. _Percentiles:

Percentiles
^^^^^^^^^^^

Percentiles of the data can be computed along any subset of the axes
with the `~Field.percentile` method of the field construct.

.. code-block:: python
   :caption: *Find the 20th, 40th, 50th, 60th and 80th percentiles.*

   >>> q, t = cf.read('file.nc')
   >>> print(q)
   Field: specific_humidity
   ------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean
   Dimension coords: time(1) = [2019-01-01 00:00:00]
                   : latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
   >>> print(q.array)
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
   >>> p = q.percentile([20, 40, 50, 60, 80])
   >>> print(p)
   Field: specific_humidity
   ------------------------
   Data            : specific_humidity(long_name=Percentile ranks for latitude, longitude dimensions(5), latitude(1), longitude(1)) 1
   Dimension coords: time(1) = [2019-01-01 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : long_name=Percentile ranks for latitude, longitude dimensions(5) = [20, ..., 80]
   >>> print(p.array)
   [[[0.0164]]
    [[0.032 ]]
    [[0.036 ]]
    [[0.0414]]
    [[0.0704]]]

.. code-block:: python
   :caption: *Find the standard deviation of the values above the 80th
             percentile.*

   >>> p80 = q.percentile(80)
   >>> print(p80)
   Field: specific_humidity
   ------------------------
   Data            : specific_humidity(latitude(1), longitude(1)) 1
   Dimension coords: time(1) = [2019-01-01 00:00:00]
                   : latitude(1) = [0.0] degrees_north
                   : longitude(1) = [180.0] degrees_east
                   : long_name=Percentile ranks for latitude, longitude dimensions(1) = [80]
   >>> g = q.where(q<=p80, cf.masked)
   >>> print(g.array)
   [[  --    --    --    --    --    -- -- --]
    [  --    --    --    --    -- 0.073 -- --]
    [0.11 0.131 0.124 0.146 0.087 0.103 -- --]
    [  --    --    --    --    -- 0.072 -- --]
    [  --    --    --    --    --    -- -- --]]
   >>> g.collapse('standard_deviation', weights=True).data
   <CF Data(1, 1): [[0.024609938742357642]] 1>

.. code-block:: python
   :caption: *Find the mean of the values above the 45th percentile
             along the X axis.*

   >>> p45 = q.percentile(45, axes='X')
   >>> print(p45.array)
   [[0.0189 ]
    [0.04515]
    [0.10405]
    [0.04185]
    [0.02125]]
   >>> g = q.where(q<=p45, cf.masked)
   >>> print(g.array)
   [[  -- 0.034    --    --    -- 0.037 0.024 0.029]
    [  --    --    -- 0.062 0.046 0.073    -- 0.066]
    [0.11 0.131 0.124 0.146    --    --    --    --]
    [  -- 0.059    -- 0.07  0.058 0.072    --    --]
    [  -- 0.036    -- 0.035   --  0.037 0.034    --]]
   >>> print(g.collapse('X: mean', weights=True).array)
   [[0.031  ]
    [0.06175]
    [0.12775]
    [0.06475]
    [0.0355 ]]

.. code-block:: python
   :caption: *Find the histogram bin boundaries associated with given
             percentiles, and digitize the data based on these bins.*

   >>> bins = q.percentile([0, 10, 50, 90, 100], squeeze=True)
   >>> print(bins.array)
   [0.003  0.0088 0.036  0.1037 0.146 ]
   >>> i = q.digitize(bins, closed_ends=True)
   >>> print(i.array)
   [[0 1 0 1 1 2 1 1]
    [1 2 2 2 2 2 0 2]
    [3 3 3 3 2 2 2 1]
    [1 2 2 2 2 2 1 1]
    [0 2 1 1 1 2 1 1]]

----

.. _Regridding:

**Regridding**
--------------

Regridding, also called remapping or interpolation, is the process of
changing the domain of a field construct whilst preserving the
qualities of the original data.

The field construct has two regridding methods: `~Field.regrids` for
regridding data between domains with spherical coordinate systems; and
`~Field.regridc` for regridding data between domains with Cartesian
coordinate systems. The interpolation is carried by out using the
`ESMF <https://www.earthsystemcog.org/projects/esmpy>`_ package, a
Python interface to the Earth System Modeling Framework regridding
utility.

As with :ref:`statistical collapses <Statistical-collapses>`,
regridding may be applied over a subset of the domain axes, and the
domain axis constructs and coordinate constructs for the non-regridded
dimensions remain the same.

:ref:`Domain ancillary constructs <domain-ancillaries>` whose data
spans the regridding dimensions are also regridded, but :ref:`field
ancillary constructs <field-ancillaries>` whose data spans the
regridding dimensions are removed from the regridded field construct.

.. _Regridding-methods:

Regridding methods
^^^^^^^^^^^^^^^^^^

The following regridding methods are available (in this table,
"source" and "destination" refer to the domain of the field construct
being regridded, and the domain that it is being regridded to,
respectively):

+--------------------------+-----------------------------------------+
| Method                   | Notes                                   |
+==========================+=========================================+
| Linear (``'linear'``,    | Linear interpolation in the number of   |
| previously called        | dimensions corresponding to the domain. |
| ``'bilinear'``, which    |                                         |
| is still supported,      | For example, for 2D domains this        |
| but you are encouraged   | amounts to *bilinear*                   |
| to use ``'linear'``      | interpolation (that is, linear          |
| instead now)             | interpolation in *both* axes) and for   |
|                          | regridding in 3D (only available with   |
|                          | `Cartesian-regridding`_) it amounts to  |
|                          | *trilinear* interpolation over the      |
|                          | three axes.                             |
+--------------------------+-----------------------------------------+
| *First-order*            | Preserve the area integral of the data  |
| conservative             | across the interpolation from source    |
| (``'conservative'`` or   | to destination. It uses the proportion  |
| ``'conservative_1st'``)  | of the area of the overlapping source   |
|                          | and destination cells to determine      |
|                          | appropriate weights.                    |
|                          |                                         |
|                          | In particular, the weight of            |
|                          | a source cell is the ratio of           |
|                          | the area of intersection of the source  |
|                          | and destination cells to the area of    |
|                          | the whole destination cell.             |
|                          |                                         |
|                          | It does not account for the             |
|                          | field gradient across the source        |
|                          | cell, unlike the second-order           |
|                          | conservative method (see below).        |
+--------------------------+-----------------------------------------+
| *Second-order*           | As with first-order (see above),        |
| conservative             | preserves the area integral of the      |
| (``'conservative_2nd'``) | field between source and destination    |
|                          | using a weighted sum, with weights      |
|                          | based on the proportionate area of      |
|                          | intersection.                           |
|                          |                                         |
|                          | But unlike first-order, the             |
|                          | second-order method incorporates        |
|                          | further terms to take into              |
|                          | consideration the gradient of the       |
|                          | field across the source cell,           |
|                          | thereby typically producing a           |
|                          | smoother result of higher accuracy.     |
+--------------------------+-----------------------------------------+
| Higher order patch       | A second degree polynomial regridding   |
| recovery (``'patch'``)   | method, which uses a least squares      |
|                          | algorithm to calculate the polynomial.  |
|                          |                                         |
|                          | This method gives better                |
|                          | derivatives in the resulting            |
|                          | destination data than the linear        |
|                          | method.                                 |
+--------------------------+-----------------------------------------+
| Nearest neighbour        | Nearest neighbour interpolation, which  |
| interpolation mapping    | is useful for extrapolation of          |
| *destination to nearest* | categorical data. In this variant,      |
| *source*                 | *each destination point* is mapped      |
| (``'nearest_stod'``)     | to the *closest source*.                |
|                          |                                         |
|                          | See also below for the                  |
|                          | the other variant of the                |
|                          | nearest neighbour approach.             |
+--------------------------+-----------------------------------------+
| Nearest neighbour        | Nearest neighbour interpolation, which  |
| interpolation mapping    | is useful for extrapolation of          |
| *source to nearest*      | categorical data. In this variant,      |
| *destination*            | *each source point* is mapped to the    |
| (``'nearest_dtos'``)     | *closest destination*.                  |
|                          |                                         |
|                          | In this case, a given destination       |
|                          | point may receive input from multiple   |
|                          | source points, but no source point      |
|                          | will map to more than one               |
|                          | destination point.                      |
|                          |                                         |
|                          | See also above for the other            |
|                          | variant of nearest neighbour            |
|                          | interpolation.                          |
+--------------------------+-----------------------------------------+

.. _Spherical-regridding:

Spherical regridding
^^^^^^^^^^^^^^^^^^^^

Regridding from and to spherical coordinate systems using the
`~cf.Field.regrids` method is only available for the 'X' and 'Y' axes
simultaneously. All other axes are unchanged. The calculation of the
regridding weights is based on areas and distances on the surface of
the sphere, rather in :ref:`Euclidean space <Cartesian-regridding>`.

The following combinations of spherical source and destination domain
coordinate systems are available to the `~Field.regrids` method:
		   
==============================  ==============================
Spherical source domain         Spherical destination domain
==============================  ==============================
`Latitude-longitude`_           `Latitude-longitude`_
`Latitude-longitude`_           `Rotated latitude-longitude`_
`Latitude-longitude`_           `Plane projection`_
`Latitude-longitude`_           `Tripolar`_
`Rotated latitude-longitude`_   `Latitude-longitude`_
`Rotated latitude-longitude`_   `Rotated latitude-longitude`_
`Rotated latitude-longitude`_   `Plane projection`_
`Rotated latitude-longitude`_   `Tripolar`_
`Plane projection`_             `Latitude-longitude`_
`Plane projection`_             `Rotated latitude-longitude`_
`Plane projection`_             `Plane projection`_
`Plane projection`_             `Tripolar`_
`Tripolar`_                     `Latitude-longitude`_
`Tripolar`_                     `Rotated latitude-longitude`_
`Tripolar`_                     `Plane projection`_
`Tripolar`_                     `Tripolar`_
==============================  ==============================

The most convenient usage is when the destination domain exists
in another field construct. In this case, all you need to specify is the
field construct having the desired destination domain and the
regridding method to use:

.. code-block:: python
   :caption: *Regrid the field construct a conservatively onto a grid
             contained in field construct b.*

   >>> a = cf.read('air_temperature.nc')[0]
   >>> b = cf.read('precipitation_flux.nc')[0]
   >>> print(a)
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(2), latitude(73), longitude(96)) K
   Cell methods    : time(2): mean
   Dimension coords: time(2) = [1860-01-16 00:00:00, 1860-02-16 00:00:00] 360_day
                   : latitude(73) = [-90.0, ..., 90.0] degrees_north
                   : longitude(96) = [0.0, ..., 356.25] degrees_east
                   : height(1) = [2.0] m
   >>> print(b)
   Field: precipitation_flux (ncvar%tas)
   -------------------------------------
   Data            : precipitation_flux(time(1), latitude(64), longitude(128)) kg m-2 day-1
   Cell methods    : time(1): mean (interval: 1.0 month)
   Dimension coords: time(1) = [0450-11-16 00:00:00] noleap
                   : latitude(64) = [-87.86380004882812, ..., 87.86380004882812] degrees_north
                   : longitude(128) = [0.0, ..., 357.1875] degrees_east
                   : height(1) = [2.0] m
   >>> c = a.regrids(b, 'conservative')
   >>> print(c)
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(2), latitude(64), longitude(128)) K
   Cell methods    : time(2): mean
   Dimension coords: time(2) = [1860-01-16 00:00:00, 1860-02-16 00:00:00] 360_day
                   : latitude(64) = [-87.86380004882812, ..., 87.86380004882812] degrees_north
                   : longitude(128) = [0.0, ..., 357.1875] degrees_east
                   : height(1) = [2.0] m

It is generally not necessary to specify which are the 'X' and 'Y'
axes in the domains of both the source and destination field
constructs, since they will be automatically identified by their
metadata. However, in cases when this is not possible (such as for
tripolar domains) the *src_axes* or *dst_axes* keywords of the
`~Field.regrids` method can be used.

It may be that the required destination domain does not exist in a
field construct. In this case, the latitude and longitudes of the
destination domain may be defined solely by dimension or auxiliary
coordinate constructs.

.. code-block:: python
   :caption: *Regrid 'a' onto two-dimensional (curvilinear) dimension
             coordinates latitude and longitude.*

   >>> import numpy
   >>> lat = cf.DimensionCoordinate(data=cf.Data(numpy.arange(-90, 92.5, 2.5), 'degrees_north'))
   >>> lon = cf.DimensionCoordinate(data=cf.Data(numpy.arange(0, 360, 5.0), 'degrees_east'))
   >>> c = a.regrids({'latitude': lat, 'longitude': lon}, 'linear')
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(2), latitude(73), longitude(72)) K
   Cell methods    : time(2): mean
   Dimension coords: time(2) = [1860-01-16 00:00:00, 1860-02-16 00:00:00] 360_day
                   : latitude(73) = [-90.0, ..., 90.0] degrees_north
                   : longitude(72) = [0.0, ..., 355.0] degrees_east
                   : height(1) = [2.0] m

A destination domain defined by two-dimensional (curvilinear) latitude
and longitude auxiliary coordinate constructs can also be specified in
a similar manner.

An axis is :ref:`cyclic <Cyclic-domain-axes>` if cells at both of its
ends are actually geographically adjacent. In spherical regridding,
only the 'X' axis has the potential for being cyclic. For example, a
longitude cell spanning 359 to 360 degrees east is proximate to the
cell spanning 0 to 1 degrees east.

When a cyclic dimension can not be automatically detected, such as
when its dimension coordinate construct does not have bounds,
cyclicity may be set with the *src_cyclic* or *dst_cyclic* keywords of
the `~Field.regrids` method.

To find out whether a dimension is cyclic use the `~Field.iscyclic`
method of the field construct, or to manually set its cyclicity use
the `cyclic` method. If the destination domain has been defined by a
dictionary of dimension coordinate constructs, then cyclicity can be
registered by setting a period of cyclicity with the
`~DimensionCoordinate.period` method of the dimension coordinate
construct.

.. _Cartesian-regridding:

Cartesian regridding
^^^^^^^^^^^^^^^^^^^^

Cartesian regridding with the `~cf.Field.regridc` method is very
similar to :ref:`spherical regridding <Spherical-regridding>`, except
regridding dimensions are not restricted to the horizontal plane, the
source and destination domains are assumed to be `Euclidian spaces
<https://en.wikipedia.org/wiki/Euclidean_space>`_ for the purpose of
calculating regridding weights, and all dimensions are assumed to be
non-cyclic by default.

Cartesian regridding can be done in up to three dimensions. It is
often used for regridding along the time dimension. A plane projection
coordinate system can be regridded with Cartesian regridding, which
will produce similar results to using using spherical regridding.

.. code-block:: python
   :caption: *Regrid the time axis 'T' of field 'a' with the linear
             method onto the grid specified in the dimension coordinate
             time.*

   >>> time = cf.DimensionCoordinate()
   >>> time.standard_name='time'
   >>> time.set_data(cf.Data(numpy.arange(0.5, 60, 1),
   ...                       units='days since 1860-01-01', calendar='360_day'))
   >>> time
   <CF DimensionCoordinate: time(60) days since 1860-01-01 360_day>
   >>> c = a.regridc({'T': time}, axes='T', method='linear')
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(60), latitude(73), longitude(96)) K
   Cell methods    : time(60): mean
   Dimension coords: time(60) = [1860-01-01 12:00:00, ..., 1860-02-30 12:00:00] 360_day
                   : latitude(73) = [-90.0, ..., 90.0] degrees_north
                   : longitude(96) = [0.0, ..., 356.25] degrees_east
                   : height(1) = [2.0] m


Note the requirement for the conservative method of contiguous,
non-overlapping bounds on the destination domain:

.. code-block:: python
   :caption: *Regrid the time axis 'T' of field 'a' conservatively
             (to first order) onto the grid specified in the dimension
             coordinate time.*

   >>> c = a.regridc({'T': time}, axes='T', method='conservative')  # Raises Exception
   ValueError: Destination coordinates must have contiguous, non-overlapping bounds for conservative regridding.
   >>> bounds = time.create_bounds()
   >>> time.set_bounds(bounds)
   >>> c = a.regridc({'T': time}, axes='T', method='conservative')
   >>> print(c)
   Field: air_temperature (ncvar%tas)
   ----------------------------------
   Data            : air_temperature(time(60), latitude(73), longitude(96)) K
   Cell methods    : time(60): mean
   Dimension coords: time(60) = [1860-01-01 12:00:00, ..., 1860-02-30 12:00:00] 360_day
                   : latitude(73) = [-90.0, ..., 90.0] degrees_north
                   : longitude(96) = [0.0, ..., 356.25] degrees_east
                   : height(1) = [2.0] m


Cartesian regridding to the dimension of another field construct is
also possible, similarly to spherical regridding.


.. _Regridding-masked-data:

Regridding masked data
^^^^^^^^^^^^^^^^^^^^^^

The data mask of the source field construct is taken into account,
such that the regridded data will be masked in regions where the
source data is masked. By default the mask of the destination field
construct is not used, but can be taken into account by setting
*use_dst_mask* keyword to the `~Field.regrids` or `~Field.regridc`
methods. For example, this is useful when part of the destination
domain is not being used (such as the land portion of an ocean grid).

For conservative regridding, masking is done on cells. Masking a
destination cell means that the cell won't participate in the
regridding. For all other regridding methods, masking is done on
points. For these methods, masking a destination point means that the
point will not participate in the regridding.

.. _Vertical-regridding:

Vertical regridding
^^^^^^^^^^^^^^^^^^^

The only option for regridding along a vertical axis is to use
Cartesian regridding. However, care must be taken to ensure that the
vertical axis is transformed so that it's coordinate values vary
linearly. For example, to regrid data on one set of vertical pressure
coordinates to another set, the pressure coordinates may first be
transformed into the logarithm of pressure, and then changed back to
pressure coordinates after the regridding operation.

.. code-block:: python
   :caption: *Regrid a field construct from one set of pressure levels
             to another.*

   >>> v = cf.read('vertical.nc')[0]
   >>> print(v)
   Field: eastward_wind (ncvar%ua)
   -------------------------------
   Data            : eastward_wind(time(3), air_pressure(5), grid_latitude(11), grid_longitude(10)) m s-1
   Cell methods    : time(3): mean
   Dimension coords: time(3) = [1979-05-01 12:00:00, 1979-05-02 12:00:00, 1979-05-03 12:00:00] gregorian
                   : air_pressure(5) = [850.0, ..., 50.0] hPa
                   : grid_latitude(11) = [23.32, ..., 18.92] degrees
                   : grid_longitude(10) = [-20.54, ..., -16.58] degrees
   Auxiliary coords: latitude(grid_latitude(11), grid_longitude(10)) = [[67.12, ..., 66.07]] degrees_north
                   : longitude(grid_latitude(11), grid_longitude(10)) = [[-45.98, ..., -31.73]] degrees_east
   Coord references: grid_mapping_name:rotated_latitude_longitude
   >>> z_p = v.construct('Z')
   >>> print(z_p.array)
   [850. 700. 500. 250.  50.]
   >>> z_ln_p = z_p.log()
   >>> print(z_ln_p.array)
   [6.74523635 6.55108034 6.2146081  5.52146092 3.91202301]
   >>> _ = v.replace_construct('Z', z_ln_p)
   >>> new_z_p = cf.DimensionCoordinate(data=cf.Data([800, 705, 632, 510, 320.], 'hPa'))
   >>> new_z_ln_p = new_z_p.log()
   >>> new_v = v.regridc({'Z': new_z_ln_p}, axes='Z', method='linear')
   >>> new_v.replace_construct('Z', new_z_p)
   >>> print(new_v)
   Field: eastward_wind (ncvar%ua)
   -------------------------------
   Data            : eastward_wind(time(3), Z(5), grid_latitude(11), grid_longitude(10)) m s-1
   Cell methods    : time(3): mean
   Dimension coords: time(3) = [1979-05-01 12:00:00, 1979-05-02 12:00:00, 1979-05-03 12:00:00] gregorian
                   : Z(5) = [800.0, ..., 320.0] hPa
                   : grid_latitude(11) = [23.32, ..., 18.92] degrees
                   : grid_longitude(10) = [-20.54, ..., -16.58] degrees
   Auxiliary coords: latitude(grid_latitude(11), grid_longitude(10)) = [[67.12, ..., 66.07]] degrees_north
                   : longitude(grid_latitude(11), grid_longitude(10)) = [[-45.98, ..., -31.73]] degrees_east
   Coord references: grid_mapping_name:rotated_latitude_longitude

Note that the `~Field.replace_construct` method of the field construct
is used to easily replace the vertical dimension coordinate construct,
without having to manually match up the corresponding domain axis
construct and construct key.

----
   
.. _Mathematical-operations:

**Mathematical operations**
---------------------------

.. _Arithmetical-operations:

Arithmetical operations
^^^^^^^^^^^^^^^^^^^^^^^

A field construct may be arithmetically combined with another field
construct, or any other object that is broadcastable to its data. See
the :ref:`comprehensive list of available binary operations
<Field-binary-arithmetic>`.

When combining with another field construct, its data is actually
combined, but only after being transformed so that it is broadcastable
to the first field construct's data. This is done by using the
metadata constructs of the two field constructs to create a mapping of
physically compatible dimensions between the fields, and then
:ref:`manipulating the dimensions <Manipulating-dimensions>` of the
other field construct's data to ensure that they are broadcastable.

In any case, a field construct may appear as the left or right
operand, and augmented assignments are possible.

Automatic units conversions are also carried out between operands
during operations, and if one operand has no units then the units of
the other are assumed.

.. code-block:: python
   :caption: *Apply some binary arithmetic operations to combine the
              data for a pair of field constructs.*

   >>> q, t = cf.read('file.nc')
   >>> t.data.stats()   
   {'min': <CF Data(): 260.0 K>,
    'mean': <CF Data(): 269.9244444444445 K>,
    'max': <CF Data(): 280.0 K>,
    'range': <CF Data(): 20.0 K>,
    'mid_range': <CF Data(): 270.0 K>,
    'standard_deviation': <CF Data(): 5.942452002538104 K>,
    'sample_size': 90}
   >>> x = t + t
   >>> x
   <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>
   >>> x.min()
   <CF Data(): 520.0 K>
   >>> (t - 2).min()
   <CF Data(): 258.0 K>
   >>> (2 + t).min()
   <CF Data(): 262.0 K>
   >>> (t * list(range(9))).min()
   <CF Data(): 0.0 K>
   >>> (t + cf.Data(numpy.arange(20, 29), '0.1 K')).min()          
   <CF Data(): 262.6 K>

.. code-block:: python
   :caption: *Apply a binary addition operation to apply an offset to
             the units and permute the axes of air temperature data
             on a field construct. Note the use of augmented
             assignment to apply an offset to the units.*

   >>> u = t.copy()
   >>> u.transpose(inplace=True)
   >>> u.Units -= 273.15
   >>> u[0]                         
   <CF Field: air_temperature(grid_longitude(1), grid_latitude(10), atmosphere_hybrid_height_coordinate(1)) K @ 273.15>
   >>> t + u[0]
   <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>

If the physical nature of the result differs from both operands, then
the "standard_name" and "long_name" properties are removed. This is
the case if the units of the result differ from both operands, or if
they have different standard names.

.. code-block:: python
   :caption: *Applying a binary operation where the resultant field construct
             has a different physical nature to the two operands. Note the
             removal of the 'standard_name' property to account for this.*

   >>> t.identities()
   ['air_temperature',
    'Conventions=CF-1.7',
    'project=research',
    'units=K',
    'standard_name=air_temperature',
    'ncvar%ta']
   >>> u = t * cf.Data(10, 'm s-1')
   >>> u.identities()
   ['Conventions=CF-1.7',
    'project=research',
    'units=m.s-1.K',
    'ncvar%ta']
	     
The :ref:`domain <domain>` metadata constructs of the result of a
successful arithmetical operation between two field constructs are
unambiguously well defined: The domain metadata constructs of the
result of a succesful operation are copied from the left hand side
(LHS) operand, except when a coordinate construct in the LHS operand
has size 1 and the corresponding coordinate construct in right hand
side (RHS) field construct operand has size greater than 1. In this
case the coordinate construct from the RHS operand is used in the
result, to match up with the data broadcasting that will have occured
during the operation.

.. _ambiguous-result:

In circumstances when domain metadata constructs in the result can not
be inferred unambiguously then an exception will be raised. For
example, this will be the case if both operands are field constructs
with corresponding coordinate constructs of size greater than 1 *and
with different coordinate values*. In such circumstances, the field
constructs' data instances may be operated on directly, bypassing any
checks on the metadata. See :ref:`Operating-on-the-field-constructs-data`
for more details. *(This will be made easier in a future release with
a new function for combining such field constructs.)*


.. warning:: Care must be taken when combining a field construct with a
          `numpy` array or a `Data` instance, due to the ways in which
          both of these objects allow themselves to be combined with
          other types:

	  * If the field construct is on the left hand side (LHS) of
            the operation then, as expected, a field construct is
            returned whose data is the combination of the original
            field construct's data and the `numpy` array or `Data`
            instance on the right hand side (RHS).

	  * If, however, the field construct is on the RHS then a
            `numpy` array or `Data` instance (which ever type is on
            the LHS) is returned, containing the same data as in the
            first case.
    
          .. code-block:: python
             :caption: *A field construct will not be returned if the
                       left hand operand is a numpy array or a 'Data'
                       instance.*

	     >>> import numpy
	     >>> q, t = cf.read('fil.nc')
             >>> a = numpy.array(1000)
             >>> type(t * a)
	     cf.field.Field
	     >>> type(a + t)
	     numpy.ndarray
	     >>> b = numpy.random.randn(t.size).reshape(t.shape)
	     >>> type(t * b)
	     cf.field.Field
	     >>> type(b * t)
	     numpy.ndarray
	     >>> type(t - cf.Data(b))
	     cf.field.Field
	     >>> type(cf.Data(b) * t)
	     cf.data.data.Data
     
.. _Unary-operations:

Unary operations
^^^^^^^^^^^^^^^^

Python unary operators also work on the field construct's data,
returning a new field construct with modified data values. See the
:ref:`comprehensive list of available unary operations
<Field-unary-arithmetic>`.

.. code-block:: python
   :caption: *Apply some unary operations to a field construct's data.*

   >>> q, t = cf.read('file.nc')
   >>> print(q.array)  
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]   
   >>> print(-q.array)
   [[-0.007 -0.034 -0.003 -0.014 -0.018 -0.037 -0.024 -0.029]
    [-0.023 -0.036 -0.045 -0.062 -0.046 -0.073 -0.006 -0.066]
    [-0.11  -0.131 -0.124 -0.146 -0.087 -0.103 -0.057 -0.011]
    [-0.029 -0.059 -0.039 -0.07  -0.058 -0.072 -0.009 -0.017]
    [-0.006 -0.036 -0.019 -0.035 -0.018 -0.037 -0.034 -0.013]]
   >>> print(abs(-q).array)
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]

    
.. _Relational-operations:

Relational operations
^^^^^^^^^^^^^^^^^^^^^

A field construct may compared with another field construct, or any
other object that is broadcastable to its data. See the
:ref:`comprehensive list of available relational operations
<Field-comparison>`. The result is a field construct with Boolean
data values.

When comparing with another field construct, its data is actually
combined, but only after being transformed so that it is broadcastable
to the first field construct's data. This is done by using the
metadata constructs of the two field constructs to create a mapping of
physically compatible dimensions between the fields, and then
:ref:`manipulating the dimensions <Manipulating-dimensions>` of the
other field construct's data to ensure that they are broadcastable.

In any case, a field construct may appear as the left or right
operand.

Automatic units conversions are also carried out between operands
during operations, and if one operand has no units then the units of
the other are assumed.

.. code-block:: python
   :caption: *Produce field constructs of Boolean data encapsulating
             the nature of some relations between a field construct and
             another operand.*

   >>> q, t = cf.read('file.nc')
   >>> print(q.array)         
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
   >>> print((q == q).array)                                   
   [[ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True]]
   >>> print((q < 0.05).array)
   [[ True  True  True  True  True  True  True  True]
    [ True  True  True False  True False  True False]
    [False False False False False False False  True]
    [ True False  True False False False  True  True]
    [ True  True  True  True  True  True  True  True]]
   >>> print((q >= q[0]).array) 
   [[ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True False  True]
    [ True  True  True  True  True  True  True False]
    [ True  True  True  True  True  True False False]
    [False  True  True  True  True  True  True False]]

The "standard_name" and "long_name" properties are removed from the
result, which also has no units.

.. code-block:: python
   :caption: *A field construct of Boolean data created from a relational
             operation on a field construct and another operand will be
             stripped of its standard_name property (and its long_name
             property if it has been set, unlike for 'q' here).*

   >>> q.identities()
   ['specific_humidity',
    'Conventions=CF-1.7',
    'project=research',
    'units=1',
    'standard_name=specific_humidity',
    'ncvar%q']
   >>> r = q > q.mean()
   >>> r.identities()
   ['Conventions=CF-1.7',
    'project=research',
    'units=',
    'ncvar%q']

The :ref:`domain <domain>` metadata constructs of the result of a
successful relational operation between two field constructs are
unambiguously well defined: The domain metadata constructs of the
result of a succesful operation are copied from the left hand side
(LHS) operand, except when a coordinate construct in the LHS operand
has size 1 and the corresponding coordinate construct in right hand
side (RHS) field construct operand has size greater than 1. In this
case the coordinate construct from the RHS operand is used in the
result, to match up with the data broadcasting that will have occured
during the operation.

In circumstances when domain metadata constructs in the result can not
be inferred unambiguously then an exception will be raised. For
example, this will be the case if both operands are field constructs
with corresponding coordinate constructs of size greater than 1 *and
with different coordinate values*. In such circumstances, the field
constructs' data instances may be operated on directly, bypassing any
checks on the metadata. See :ref:`Operating-on-the-field-constructs-data`
for more details. *(This will be made easier in a future release with
a new function for combining such field constructs.)*

.. _Arithmetical-and-relational-operations-with-insufficient-metadata:

Arithmetical and relational operations with insufficient metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When both operands of an :ref:`arithmetical <Arithmetical-operations>`
or :ref:`relational <Relational-operations>` operation are field
constructs then the creation of the mapping of physically compatible
dimensions relies on there being sufficient metadata. By default, the
mapping relies on their being "strict" identities for the metadata
constucts with multi-valued data. The strict identity is restricted
`!standard_name` property (or `!id` attribute), and may be returned by
the `!identity` method of a construct:


.. code-block:: python
   :caption: *Find the "strict" identity of a construct.*

   >>> y = q.coordinate('Y')
   >>> y.identity(strict=True)
   'latitude'
   >>> del y.standard_name
   >>> y.identity(strict=True)
   ''

If there is insufficient metadata to create a mapping of physically
compatible dimensions, then there are various techniques that allow
the operation to proceed:

* **Option 1:** The operation may applied to the field constructs'
  data instances instead. See
  :ref:`Operating-on-the-field-constructs-data` for more details.

* **Option 2:** If the mapping is not possible due to the absence of
  `!standard_name` properties (or `!id` attributes) on metadata
  constructs that are known to correspond, then setting "relaxed
  identities" with the `cf.RELAXED_IDENTITIES` function may
  help. Setting relaxed identities to `True` allows the `!long_name`
  property and netCDF variable name (see the :ref:`netCDF interface
  <NetCDF-interface>`), to also be considered when identifying
  constructs.

* **Option 3:** Add more metadata to the field and metadata constructs.

.. _Operating-on-the-field-constructs-data:

Operating on the field constructs' data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Arithmetical <Arithmetical-operations>` and :ref:`relational
<Relational-operations>` operations between may also be carried out on
their data instances, thereby bypassing any reference to, or checks
on, the metadata constucts. This can be useful if there
:ref:`insufficient metadata
<Arithmetical-and-relational-operations-with-insufficient-metadata>`
for determining if the two field constructs are compatible; or if the
domain metadata constructs of the result can not be
:ref:`unambiguously defined <ambiguous-result>`.
     
In such cases the data instances may be operated on instead and the
result then inserted into one of the field constructs, either with the
`~cf.Field.set_data` method of the field construct, or with
:ref:`indexed assignment <Assignment-by-index>`. The former technique
is faster and more memory efficient, but the latter technique allows
broadcasting. Alternatively, for augmented assignments, the field
construct data may be changed in-place.

It is up to the user to ensure that the data instances are consistent
in terms of size 1 dimensions (to satisfy the `numpy broadcasting
rules`_), dimension order and dimension direction, and that the
resulting data is compatible with the metadata of the field constuct
which will contain it. Automatic units conversions are, however, still
accounted for when combining the data instances.


.. For **Option 1** the resulting data may then be inserted into a copy
   of one of the field constructs, either with the `~cf.Field.set_data`
   method of the field construct, or with :ref:`indexed assignment
   <Assignment-by-index>`. The former technique is faster and more memory
   efficient, but the latter technique allows
   broadcasting. Alternatively, for augmented assignments, the field
   construct data may be changed in-place.
   
   Note that it is assumed, and not checked, that the dimensions of both
   `~cf.Data` instance operands are already in the correct order for
   physically meaningful broadcasting to occur.

.. code-block:: python
   :caption: *Operate on the data and use 'set_data' to put the
             resulting data into the new field construct.*
	    
   >>> t.min()
   <CF Data(): 260.0 K>
   >>> u = t.copy()
   >>> new_data = t.data - t.data
   >>> u.set_data(new_data)
   >>> u       
   <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>
   >>> u.min()
   <CF Data(): 0.0 K>

.. code-block:: python
  :caption: *Update the data with indexed assignment*

   >>> u[...] = new_data
   >>> u.min()
   <CF Data(): 0.0 K>
   
.. code-block:: python
   :caption: *An example of augmented assignment involving the data of
            two field constructs.*

   >>> t.data -= t.data
   >>> t.min()
   <CF Data(): 0.0 K>
    
Trigonometrical and hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field construct and metadata constructs have methods to apply
trigonometric and hyperbolic functions, and their inverses,
element-wise to the data. These preserve the metadata but
change the construct's units.

The field construct and metadata constructs support the following
trigonometrical methods:

================  ========================================================
Method            Description
================  ========================================================
`~Field.arccos`   Take the inverse trigonometric cosine of the data
                  element-wise.
`~Field.arcsin`   Take the inverse trigonometric sine of the data
                  element-wise.
`~Field.arctan`   Take the inverse trigonometric tangent of the data
                  element-wise.
`~Field.cos`      Take the trigonometric cosine of the data element-wise.
`~Field.sin`      Take the trigonometric sine of the data element-wise.
`~Field.tan`      Take the trigonometric tangent of the data element-wise.
================  ========================================================

.. AT2 : As well as `~Field.arctan` there is a method available,
   `~Field.arctan2`,
   which takes the inverse trigonometric tangent of data element-wise, but
   does so instead for two constructs where the inverse tangent of the
   quotient between corresponding elements is taken, such that the signs of
   `x` and `y` values are taken into account to determine the correct quadrant
   (see `here <https://en.wikipedia.org/wiki/Atan2>`_ for further details).

The field construct and metadata constructs also support the following
hyperbolic methods:

================  ========================================================
Method            Description
================  ========================================================
`~Field.arccosh`  Take the inverse hyperbolic cosine of the data
                  element-wise.
`~Field.arcsinh`  Take the inverse hyperbolic sine of the data
                  element-wise.
`~Field.arctanh`  Take the inverse hyperbolic tangent of the data
                  element-wise.
`~Field.cosh`     Take the hyperbolic cosine of the data element-wise.
`~Field.sinh`     Take the hyperbolic sine of the data element-wise.
`~Field.tanh`     Take the hyperbolic tangent of the data element-wise.
================  ========================================================

.. code-block:: python
   :caption: *Find the sine of each latitude coordinate value.*
	     
   >>> q, t = cf.read('file.nc')
   >>> lat = q.dimension_coordinate('latitude')
   >>> lat.data
   <CF Data(5): [-75.0, ..., 75.0] degrees_north>
   >>> sin_lat = lat.sin()
   >>> sin_lat.data
   <CF Data(5): [-0.9659258262890683, ..., 0.9659258262890683] 1>

The "standard_name" and "long_name" properties are removed from the
result.

Note that a number of the inverse methods have
`mathematically restricted domains <https://mathworld.wolfram.com/InverseTrigonometricFunctions.html>`_ (see also
`here <https://mathworld.wolfram.com/InverseHyperbolicFunctions.html>`_)
and therefore may return
`"invalid" values <https://docs.scipy.org/doc/numpy/reference/constants.html>`_
(`nan` or `inf`). When applying these methods to constructs with masked
data, you may prefer to output masked values instead of invalid ones. In
this case, you can use `mask_invalid` to do the conversion afterwards:

.. code-block:: python
   :caption: *Take the `arctanh` of some masked data and then transform
             resultant invalid values into masked data values.*

   >>> d = cf.Data([2, 1.5, 1, 0.5, 0], mask=[1, 0, 0, 0, 1])
   >>> e = d.arctanh()
   >>> print(e.array)
   [-- nan inf 0.5493061443340548 --]
   >>> e.mask_invalid(inplace=True)
   >>> print(e.array)
   [-- -- -- 0.5493061443340548 --]


Exponential and logarithmic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field construct and metadata constructs have `~Field.exp` and
`~Field.log` methods for applying exponential and logarithmic
functions respectively element-wise to the data, preserving the
metadata but changing the construct's units where required.

.. code-block:: python
   :caption: *Find the logarithms and exponentials of field constructs.*
	     
   >>> q
   <CF Field: specific_humidity(latitude(5), longitude(8)) 1>
   >>> q.log()
   <CF Field: specific_humidity(latitude(5), longitude(8)) ln(re 1)>
   >>> q.exp()
   <CF Field: specific_humidity(latitude(5), longitude(8)) 1>
   >>> t   
   <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>
   >>> t.log(base=10)
   <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) lg(re 1 K)>
   >>> t.exp()  # Raises Exception
   ValueError: Can't take exponential of dimensional quantities: <Units: K>

The "standard_name" and "long_name" properties are removed from the
result.

Rounding and truncation
^^^^^^^^^^^^^^^^^^^^^^^

The field construct and metadata constructs the following methods to
round and truncate their data:

==============  ====================================================
Method          Description
==============  ====================================================
`~Field.ceil`   The ceiling of the data, element-wise.
`~Field.clip`   Limit the values in the data.
`~Field.floor`  Floor the data array, element-wise.
`~Field.rint`   Round the data to the nearest integer, element-wise.
`~Field.round`  Round the data to the given number of decimals.
`~Field.trunc`  Truncate the data, element-wise.
==============  ====================================================

Moving windows
^^^^^^^^^^^^^^

Moving window calculations along an axis may be created with the
`~Field.moving_window` method of the field construct.

Moving mean, sum, and integral calculations are possible.

By default moving means are unweighted, but weights based on the axis
cell sizes (or custom weights) may applied to the calculation.

.. code-block:: python
   :caption: *Calculate a 3-point weighted mean of the 'X' axis. Since
             the the 'X' axis is cyclic, the mean wraps by default.*

   >>> q, t = cf.read('file.nc')
   >>> print(q)
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> print(q.array)
   [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
    [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
    [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
    [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
    [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
   >>> print(q.coordinate('X').bounds.array)
   [[  0.  45.]
    [ 45.  90.]
    [ 90. 135.]
    [135. 180.]
    [180. 225.]
    [225. 270.]
    [270. 315.]
    [315. 360.]]
   >>> q.iscyclic('X')
   True
   >>> g = f.moving_window('mean', 3, axis='X', weights=True)
   >>> print(g)
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean longitude(8): mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]    
   >>> print(g.array)
   [[0.02333 0.01467 0.017   0.01167 0.023   0.02633 0.03    0.02   ]
    [0.04167 0.03467 0.04767 0.051   0.06033 0.04167 0.04833 0.03167]
    [0.084   0.12167 0.13367 0.119   0.112   0.08233 0.057   0.05933]
    [0.035   0.04233 0.056   0.05567 0.06667 0.04633 0.03267 0.01833]
    [0.01833 0.02033 0.03    0.024   0.03    0.02967 0.028   0.01767]]
   >>> print(g.coordinate('X').bounds.array)
   [[-45.  90.]
    [  0. 135.]
    [ 45. 180.]
    [ 90. 225.]
    [135. 270.]
    [180. 315.]
    [225. 360.]
    [270. 360.]]

.. note:: The `~Field.moving_window` method can not, in general, be
          emulated by the `~Field.convolution_filter` method, as the
          latter i) can not change the window weights as the filter
          passes through the axis; and ii) does not update the cell
          method constructs.
        
Convolution filters
^^^^^^^^^^^^^^^^^^^

A `convolution <https://en.wikipedia.org/wiki/Convolution>`_ of the
field construct data with a filter along a single domain axis can be
calculated, which also updates the bounds of a relevant dimension
coordinate construct to account for the width of the
filter. Convolution filters are carried with the
`~Field.convolution_filter` method of the field construct.

.. code-block:: python
   :caption: *Calculate a 5-point mean of the 'X' axis with a
             non-uniform window function. Since the the 'X' axis is
             cyclic, the convolution wraps by default.*

   >>> print(q)
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> q.iscyclic('X')
   True
   >>> r = q.convolution_filter([0.1, 0.15, 0.5, 0.15, 0.1], axis='X')
   >>> print(r)
   Field: specific_humidity (ncvar%q)
   ----------------------------------
   Data            : specific_humidity(latitude(5), longitude(8)) 1
   Cell methods    : area: mean
   Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                   : longitude(8) = [22.5, ..., 337.5] degrees_east
                   : time(1) = [2019-01-01 00:00:00]
   >>> print(q.dimension_coordinate('X').bounds.array)
   [[  0.  45.]
    [ 45.  90.]
    [ 90. 135.]
    [135. 180.]
    [180. 225.]
    [225. 270.]
    [270. 315.]
    [315. 360.]]
   >>> print(r.dimension_coordinate('X').bounds.array)
   [[-90. 135.]
    [-45. 180.]
    [  0. 225.]
    [ 45. 270.]
    [ 90. 315.]
    [135. 360.]
    [180. 405.]
    [225. 450.]]

The `~Field.convolution_filter` method of the field construct also has
options to

* Specify how the input array is extended when the filter overlaps a
  border, and

* Control the placement position of the filter window.

Note that the `scipy.signal.windows` package has suite of window
functions for creating window weights for filtering:

.. code-block:: python
   :caption: *Calculate a 3-point exponential filter of the 'Y'
             axis. Since the 'Y' axis is not cyclic, the convolution
             by default inserts missing data at points for which the
             filter window extends beyond the array.*

   >>> from scipy.signal import windows
   >>> exponential_window = windows.exponential(3)
   >>> print(exponential_window)
   [0.36787944 1.         0.36787944]
   >>> r = q.convolution_filter(exponential_window, axis='Y')
   >>> print(r.array)
   [[--      --      --      --      --      --      --      --     ]
    [0.06604 0.0967  0.09172 0.12086 0.08463 0.1245  0.0358  0.08072]
    [0.12913 0.16595 0.1549  0.19456 0.12526 0.15634 0.06252 0.04153]
    [0.07167 0.12044 0.09161 0.13659 0.09663 0.1235  0.04248 0.02583]
    [--      --      --      --      --      --      --      --     ]]

The magnitude of the integral of the filter (i.e. the sum of the
window weights defined by the *window* parameter) affects the
convolved values. For example, window weights of ``[0.2, 0.2 0.2, 0.2,
0.2]`` will produce a non-weighted 5-point running mean; and window
weights of ``[1, 1, 1, 1, 1]`` will produce a 5-point running
sum. Note that the window weights returned by functions of the
`scipy.signal.windows` package do not necessarily sum to 1.

.. note:: The `~Field.moving_window` method can not, in general, be
          emulated by the `~Field.convolution_filter` method, as the
          latter i) can not change the window weights as the filter
          passes through the axis; and ii) does not update the cell
          method constructs.
        
Derivatives
^^^^^^^^^^^

The derivative along a dimension of the field construct's data can be
calculated as a centred finite difference with the `~Field.derivative`
method. If the axis is :ref:`cyclic <Cyclic-domain-axes>` then the
derivative wraps around by default, otherwise it may be forced to wrap
around; a one-sided difference is calculated at the edges; or missing
data is inserted.

.. code-block:: python
   :caption: *Calculate for a field construct's data the derivative along both
             the 'X' and 'Y' axes, where the former (by default) uses missing
             values in the calculation, but the latter has been told to use a
             one-sided finite difference at the boundary.*

   >>> r = q.derivative('X')
   >>> r = q.derivative('Y', one_sided_at_boundary=True)

Relative vorticity
^^^^^^^^^^^^^^^^^^

The relative vorticity of the wind may be calculated on a global or
limited area domain, and in Cartesian or spherical polar coordinate
systems.

The relative vorticity of wind defined on a Cartesian domain (such as
a `plane projection`_) is defined as

.. math:: \zeta _{cartesian} = \frac{\delta v}{\delta x} -
          \frac{\delta u}{\delta y}

where :math:`x` and :math:`y` are points on along the 'X' and 'Y'
Cartesian dimensions respectively; and :math:`u` and :math:`v` denote
the 'X' and 'Y' components of the horizontal winds.

If the wind field is defined on a spherical latitude-longitude
domain then a correction factor is included:

.. math:: \zeta _{spherical} = \frac{\delta v}{\delta x} -
          \frac{\delta u}{\delta y} + \frac{u}{r}tan(\phi)

where :math:`u` and :math:`v` denote the longitudinal and latitudinal
components of the horizontal wind field; :math:`r` is the radius of
the Earth; and :math:`\phi` is the latitude at each point.

The `cf.relative_vorticity` function creates a relative vorticity
field construct from field constructs containing the wind components
using finite differences to approximate the derivatives. Dimensions
other than 'X' and 'Y' remain unchanged by the operation.

.. code-block:: python
   :caption: *Generate a relative vorticity field construct from wind
             component field constructs, then round the field's data to
             8 decimal places.*
   
   >>> u, v = cf.read('wind_components.nc')
   >>> zeta = cf.relative_vorticity(u, v)
   >>> print(zeta)
   Field: atmosphere_relative_vorticity (ncvar%va)
   -----------------------------------------------
   Data            : atmosphere_relative_vorticity(time(1), atmosphere_hybrid_height_coordinate(1), latitude(9), longitude(8)) s-1
   Dimension coords: time(1) = [1978-09-01 06:00:00] 360_day
                   : atmosphere_hybrid_height_coordinate(1) = [9.9982] m
                   : latitude(9) = [-90, ..., 70] degrees_north
                   : longitude(8) = [0, ..., 315] degrees_east
   Coord references: standard_name:atmosphere_hybrid_height_coordinate
   Domain ancils   : atmosphere_hybrid_height_coordinate(atmosphere_hybrid_height_coordinate(1)) = [9.9982] m
                   : long_name=vertical coordinate formula term: b(k)(atmosphere_hybrid_height_coordinate(1)) = [0.9989]
                   : surface_altitude(latitude(9), longitude(8)) = [[2816.25, ..., 2325.98]] m
   >>> print(zeta.array.round(8))
   [[[[--        --        --        --        --        --        --        --       ]
      [-2.04e-06  1.58e-06  5.19e-06  4.74e-06 -4.76e-06 -2.27e-06  9.55e-06 -3.64e-06]
      [-8.4e-07  -4.37e-06 -3.55e-06 -2.31e-06 -3.6e-07  -8.58e-06 -2.45e-06  6.5e-07 ]
      [ 4.08e-06  4.55e-06  2.75e-06  4.15e-06  5.16e-06  4.17e-06  4.67e-06 -7e-07   ]
      [-1.4e-07  -3.5e-07  -1.27e-06 -1.29e-06  2.01e-06  4.4e-07  -2.5e-06   2.05e-06]
      [-7.3e-07  -1.59e-06 -1.77e-06 -3.13e-06 -7.9e-07  -5.1e-07  -2.79e-06  1.12e-06]
      [-3.7e-07   7.1e-07   1.52e-06  6.5e-07  -2.75e-06 -4.3e-07   1.62e-06 -6.6e-07 ]
      [ 9.5e-07  -8e-07     6.6e-07   7.2e-07  -2.13e-06 -4.5e-07  -7.5e-07  -1.11e-06]
      [--        --        --        --        --        --        --        --       ]]]]

For axes that are not :ref:`cyclic <Cyclic-domain-axes>`, missing data
is inserted at the edges by default; otherwise it may be forced to
wrap around, or a one-sided difference is calculated at the edges. If
the longitudinal axis is :ref:`cyclic <Cyclic-domain-axes>` then the
derivative wraps around by default.
  
----

.. External links

.. _Tripolar:                 https://doi.org/10.1007%2FBF00211684

.. _numpy broadcasting rules: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

.. External links to the CF conventions (will need updating with new versions of CF)
   
.. _CF-netCDF cell methods:           http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-methods
.. _Climatological statistics:        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#climatological-statistics
.. _Latitude-longitude:               http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_latitude_longitude
.. _Rotated latitude-longitude:       http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_rotated_pole
.. _Plane projection:                 http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
.. _plane projection:                 http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
