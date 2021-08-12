.. currentmodule:: cf
.. default-role:: obj

.. _Aggregation-rules:
   
**Aggregation rules**
=====================

----

*David Hassell and Jonathan Gregory (2012)*

https://cf-trac.llnl.gov/trac/ticket/78

Version |release| for version |version| of the CF conventions.

Aggregation is the combination of two field constructs to create a new
field construct that occupies a "larger" domain. In practice, this
means combining the two field constructs so that their data are
concatenated along one or more domain axis constructs, as are the data
of their metadata constructs which span those domain axis constructs.

These rules are to be used for deciding whether or not two arbitrary
field constructs may be aggregated into one, larger field construct.
The rules are based solely on the field constructs' metadata as
recognised by the :ref:`CF-data-model` [#cfdm]_. For example, netCDF
variable names are ignored during the aggregation process, meaning
that having different netCDF variable names does not preclude the
aggregation of two field constructs.

More than two field constructs are aggregated by repeated applications
of the aggregation algorithm, and aggregations over multiple domain
axis constructs are similarly possible.

Aggregation is implemented in the `cf.aggregate` function, and is
applied by default by the `cf.read` function.

----

If all of the following statements are true for two arbitrary field
constructs then the two field constructs may be aggregated to a new
field construct.

* Both field constructs have identical standard name properties.

  - The treatment of other properties, in terms of how they should
    match between the field constructs and whether or not they should be
    included in the aggregated field, is beyond the scope of these
    rules and is at the user's discretion. This is also the case for
    all other construct properties unnamed by these rules.

.. admonition:: Coordinate construct

   A coordinate construct is either a dimension coordinate construct
   or an auxiliary coordinate construct.

.. admonition:: Pair of matching coordinate constructs

   A pair of matching coordinate constructs is a coordinate construct
   from each field construct of the same type (dimension or
   auxiliary), with equivalent calendar properties (if present) and
   identical standard names.

* Both field constructs have the same number of coordinate constructs, all of
  which have a standard name property, and each coordinate construct's
  standard name is unique within its field construct. Each coordinate construct
  in one field construct forms a pair of matching coordinate
  constructs with a unique coordinate construct in the other field
  construct.

.. admonition:: Domain axis's associated coordinate constructs

   A domain axis's associated coordinate constructs are those
   coordinate constructs which span the domain axis.

* Each domain axis in both field constructs has at least one associated 1-d
  coordinate construct.
 
.. admonition:: Pair of matching axes

   A pair of matching axes is a domain axis from each field construct chosen
   such that the two domain axes have associated coordinate constructs
   that are all matching pairs.

* Each domain axis in one field construct forms a pair of matching axes with a
  unique domain axis in the other field construct.

  - A consequence of this rule is that within each pair of matching
    coordinate constructs, the corresponding domain axes are matching
    pairs.

.. admonition:: Pair of aggregating axes

   The pair of matching axes for which one or more of the 1D matching
   coordinate constructs have different values for their coordinate
   arrays and, if present, their boundary coordinate arrays is called
   the pair of aggregating axes.
  
* There is exactly one pair of matching axes for which one or more of
  the 1D matching coordinate constructs have different values for
  their coordinate arrays and, if present, their boundary coordinate
  arrays.

  - A pair of matching non-aggregating axes must have identical sizes.
  - The aggregating axis may have either the same size or a different
    size in the two field constructs.
  - If the domain axes of the arrays of matching associated constructs
    are ordered differently or run in different senses then one of the
    arrays must be rearranged prior to comparison. This is true for
    all array comparisons in these rules.
  - If neither field construct has an aggregating axis then the field constructs are not
    aggregatable because their domains are identical.
  - If either field construct has two or more domain axes which could qualify as
    an aggregating axis then the field constructs are not aggregatable because
    it is not possible to choose the single domain axis for array
    concatenation.

.. admonition:: Pair of matching cell measure constructs 

   A pair of matching cell measure constructs is a cell measure
   construct from each field construct with equivalent units and corresponding
   domain axes that are matching pairs.

* Both field constructs have the same number of cell measure constructs, all of
  which have a units property. Each cell measure construct in either
  field construct forms a pair of matching cell measure constructs with a unique
  cell measure construct in the other field construct.
  
..
  
* Each pair of matching coordinate constructs and matching cell
  measure constructs that do not span their aggregating axes have
  identical values for their coordinate arrays and, if present, their
  boundary coordinate arrays.

..

* If the pair of matching aggregating axes has a pair of associated
  dimension coordinate constructs, then there are no common values in
  their coordinate arrays. If the matching dimension coordinate
  constructs have boundary coordinate arrays then no cells from one
  dimension coordinate construct lie entirely within any cell of the
  other dimension coordinate construct.
  
  - This does not preclude the coordinate arrays' ranges from
    overlapping.
  - The condition on the boundary coordinate arrays prevents, for
    example, a monthly mean being aggregated with a daily mean from
    the same month.
  - The condition on the boundary coordinate arrays allows, for
    example, the aggregation of overlapping running means; or the
    aggregation of a monthly mean and a daily mean from a different
    month.

* If one field construct has a cell methods construct then so does the other
  field, with the equivalent methods in the same order. Corresponding
  domain axes in each cell methods are matching pairs.

.. admonition:: Pair of matching domain ancillary constructs

   A pair of matching domain ancillary constructs is a domain
   ancillary construct from each field construct for identical standard
   coordinate conversion terms and corresponding domain axes that are
   matching pairs.

* Both field constructs have the same number of domain ancillary
  constructs. Each domain ancillary construct in either field construct forms a
  pair of matching domain ancillary constructs with a unique domain
  ancillary construct in the other field construct.

.. admonition:: Pair of matching field ancillary constructs

   A pair of matching field ancillary constructs is a field ancillary
   construct from each field construct with identical standard names and
   corresponding domain axes that are matching pairs.

* Both field constructs have the same number of field ancillary
  constructs. Each field ancillary construct in either field construct
  forms a pair of matching field ancillary constructs with a unique
  field ancillary construct in the other field construct.

..
  
* Both field constructs have the same number of coordinate reference
  constructs. For each coordinate reference construct in one field construct
  there is a coordinate reference construct in the other field construct with
  identical name and the same set of terms, taking optional terms into
  account. Corresponding terms which are scalar or vector parameters
  are identical, taking into account equivalent units. Corresponding
  terms which are domain ancillary constructs form a pair of matching
  domain ancillary constructs.

----
   
.. [#cfdm] Hassell, D., Gregory, J., Blower, J., Lawrence, B. N., and
           Taylor, K. E.: A data model of the Climate and Forecast
           metadata conventions (CF-1.6) with a software
           implementation (cf-python v2.1), Geosci. Model Dev., 10,
           4619-4646, https://doi.org/10.5194/gmd-10-4619-2017, 2017.
