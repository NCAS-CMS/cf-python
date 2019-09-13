from copy      import deepcopy
from functools import partial as functools_partial
from re        import escape  as re_escape
from re        import match   as re_match
from re        import findall as re_findall
from textwrap  import fill as textwrap_fill
from itertools import izip
from cPickle   import dumps, loads, PicklingError
from re        import search as re_search
from netCDF4   import default_fillvals as _netCDF4_default_fillvals
from operator  import truediv  as truediv
from operator  import itruediv as itruediv
from re        import compile as re_compile

from numpy import array       as numpy_array
from numpy import result_type as numpy_result_type
from numpy import vectorize   as numpy_vectorize

from .cfdatetime   import dt
from .flags        import Flags
from .functions    import RTOL, ATOL, RELAXED_IDENTITIES
from .functions    import equals     as cf_equals
from .functions    import equivalent as cf_equivalent
from .functions    import inspect    as cf_inspect
from .query        import Query
from .timeduration import TimeDuration
from .units        import Units

from .data.data import Data

_units_None = Units()

_month_units = ('month', 'months')
_year_units  = ('year', 'years', 'yr')

docstring = {
    
    # ----------------------------------------------------------------
    '{+chunksizes}': '''chunksizes: `dict` or `None`, optional
        Specify the chunk sizes for axes of the {+variable}. Axes are
        given by dictionary keys, with a chunk size for those axes as
        the dictionary values. A dictionary key may be an integer or a
        tuple of integers defining axes by position in the data
        array. In the special case of *chunksizes* being `None`, then
        chunking is set to the netCDF default.
    
          *Example:*
            To set the chunk size for first axes to 365: ``{0: 365}``.
          
          *Example:*
            To set the chunk size for the first and third data array
            axes to 100: ``{0: 100, 2: 100}``, or equivalently ``{(0,
            2): 100}``.
          
          *Example:*
            To set the chunk size for the second axis to 100 and for
            the third axis to 5: ``{1: 100, 2: 5}``.
          
          *Example:*
            To set the chunking to the netCDF default: ``None``.''',
    
    # ----------------------------------------------------------------
    '{+data-like}': '''A data-like object is any object containing array-like or
        scalar data which could be used to create a `cf.Data` object.
    
          *Example:*
            Instances, ``x``, of following types are all examples of
            data-like objects (because ``cf.Data(x)`` creates a valid
            `cf.Data` object): :py:obj:`int`, :py:obj:`float`,
            :py:obj:`str`, :py:obj:`tuple`, :py:obj:`list`,
            `numpy.ndarray`, `cf.Data`, `cf.Coordinate`,
            `cf.Field`.''',
    
    # ----------------------------------------------------------------
    '{+data-like-scalar}': '''A data-like scalar object is any object containing scalar data
        which could be used to create a `cf.Data` object.

          *Example:*
            Instances, ``x``, of following types are all examples of
            scalar data-like objects (because ``cf.Data(x)`` creates a
            valid `cf.Data` object): :py:obj:`int`, :py:obj:`float`,
            :py:obj:`str`, and scalar-valued `numpy.ndarray`,
            `cf.Data`, `cf.Coordinate`, `cf.Field`.''',
    
    # ----------------------------------------------------------------
    '{+default}': '''default: optional
        WRITE ME.''',
    
    # ----------------------------------------------------------------
    '{+atol}': '''atol: `float`, optional
        The absolute tolerance for all numerical comparisons, By
        default the value returned by the `cf.ATOL` function is
        used.''',
    
    # ----------------------------------------------------------------
    '{+rtol}': '''rtol: `float`, optional
        The relative tolerance for all numerical comparisons, By
        default the value returned by the `cf.RTOL` function is
        used.''',
    
    # ----------------------------------------------------------------
    '{+axis_selection}': '''Axes are selected with the criteria specified by the keyword
parameters. If no keyword parameters are specified then all axes are
selected.''',
    
    # ----------------------------------------------------------------
    '{+HDF_chunks}': '''Specify HDF5 chunks for the data array.
    
Chunking refers to a storage layout where the data array is
partitioned into fixed-size multi-dimensional chunks when written to a
netCDF4 file on disk. Chunking is ignored if the data array is written
to a netCDF3 format file.
    
A chunk has the same rank as the data array, but with fewer (or no
more) elements along each axis. The chunk is defined by a dictionary
in which the keys identify axes and the values are the chunk sizes for
those axes.
    
If a given chunk size for an axis is larger than the axis size, then
the size of the axis at the time of writing to disk will be used
instead.
    
If chunk sizes have been specified for some but not all axes, then the
each unspecified chunk size is assumed to be the full size of its
axis.

If no chunk sizes have been set for any axes then the netCDF default
chunk is used
(http://www.unidata.ucar.edu/software/netcdf/docs/netcdf_perf_chunking.html).

A detailed discussion of HDF chunking and I/O performance is available
at https://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html and
http://www.unidata.ucar.edu/software/netcdf/workshops/2011/nc4chunking. Basically,
you want the chunks for each dimension to match as closely as possible
the size and shape of the data block that users will read from the
file.''',
    
    # ----------------------------------------------------------------
    '{+item_definition}': '''An item of the field is one of the following field components:

  * dimension coordinate
  * auxiliary coordinate
  * cell measure
  * domain ancillary
  * field ancillary
  * coordinate reference''',

    # ----------------------------------------------------------------
    '{+item_selection}': '''Items are selected with the criteria specified by the keyword
parameters. If no parameters are set then all items are selected. If
multiple criteria are given then items that meet all of them are
selected (see the *match_and* parameter).''',
    
    # ----------------------------------------------------------------
    '{+item_criteria}': '''Item selection criteria are of the the following types:
    
==================  ==================================  ==================
Selection criteria  Description                         Keyword parameters
==================  ==================================  ==================
CF properties       Items with given CF properties      *description*
                                               
Attributes          Items with given attributes         *description*
                                               
Domain axes         Items which span given domain axes  *axes*,
                                                        *axes_all*,
                                                        *axes_subset*,
                                                        *axes_superset*,
                                                        *ndim*
==================  ==================================  ==================''',
    
    # ----------------------------------------------------------------
    '{+items_criteria}': '''Item selection criteria are of the the following types:
    
==================  ==================================  ==================
Selection criteria  Description                         Keyword parameters
==================  ==================================  ==================
CF properties       Items with given CF properties      *description*
                                               
Attributes          Items with given attributes         *description*
                                               
Domain axes         Items which span given domain axes  *axes*,
                                                        *axes_all*,
                                                        *axes_subset*,
                                                        *axes_superset*,
                                                        *ndim*
                                               
Role                Items of the given component type   *role*
==================  ==================================  ==================''',
    
    # ----------------------------------------------------------------
    '{+ndim}': '''ndim: optional
        Select the items whose number of data array dimensions satisfy
        the given condition. A range of dimensions may be selected if
        *ndim* is a `cf.Query` object.
    
          *Example:*
            ``ndim=1`` selects one-dimensional items and
            ``ndim=cf.ge(2)`` selects items which span two or more
            axes (see `cf.ge`).''',
    
    # ----------------------------------------------------------------
    '{+axes}': '''axes: optional
        Select items which span at least one of the specified axes,
        taken in any order, and possibly others. Axes are defined by
        identfiying items of the field (such as dimension coordinate
        objects) or by specifying axis sizes. In the former case the
        selected axes are those which span the identified items. The
        axes are interpreted as those that would be returned by the
        field's `~cf.Field.axes` method, i.e. by ``f.axes(axes)`` or,
        if *axes* is a dictionary, ``f.axes(**axes)``. See
        `cf.Field.axes` for details.
  
          *Example:*
            To select items which span a time axis, and possibly
            others, you could set: ``axes='T'``.
            
          *Example:*
            To select items which span a latitude and/or longitude
            axes, and possibly others, you could set: ``axes=['X',
            'Y']``.
            
          *Example:*
            To specify axes with size 19 you could set ``axes={'size':
            19}``. To specify depth axes with size 40 or more, you
            could set: ``axes={'axes': 'depth', 'size': cf.ge(40)}``
            (see `cf.ge`).''',
    
    # ----------------------------------------------------------------
    '{+axes_subset}': '''axes_subset: optional 
        Select items whose data array spans all of the specified axes,
        taken in any order, and possibly others. The axes are those
        that would be selected by this call of the field's
        `~cf.Field.axes` method: ``f.axes(axes_subset)`` or, if
        *axes_subset* is a dictionary of parameters ,
        ``f.axes(**axes_subset)``. Axes are defined by identfiying
        items of the field (such as dimension coordinate objects) or
        by specifying axis sizes. In the former case the selected axes
        are those which span the identified field items. See
        `cf.Field.axes` for details.
    
          *Example:*            
            To select items which span a time axes, and possibly
            others, you could set: ``axes_subset='T'``.
            
          *Example:*
            To select items which span latitude and longitude axes,
            and possibly others, you could set: ``axes_subset=['X',
            'Y']``.
            
          *Example:*
            To specify axes with size 19 you could set
            ``axes_subset={'size': 19}``. To specify depth axes with
            size 40 or more, you could set: ``axes_subset={'axes':
            'depth', 'size': cf.ge(40)}`` (see `cf.ge`).''',
    
    # ----------------------------------------------------------------
    '{+axes_superset}': '''axes_superset: optional
        Select items whose data arrays span a subset of the specified
        axes, taken in any order, and no others. The axes are those
        that would be selected by this call of the field's
        `~cf.Field.axes` method: ``f.axes(axes_superset)`` or, if
        *axes_superset* is a dictionary of parameters,
        ``f.axes(**axes_superset)``. Axes are defined by identfiying
        items of the field (such as dimension coordinate objects) or
        by specifying axis sizes. In the former case the selected axes
        are those which span the identified field items. See
        `cf.Field.axes` for details.
    
          *Example:*
            To select items which span a time axis, and no others, you
            could set: ``axes_superset='T'``.
            
          *Example:*
            To select items which span latitude and/or longitude axes,
            and no others, you could set: ``axes_superset=['X',
            'Y']``.
            
          *Example:*
            To specify axes with size 19 you could set
            ``axes_superset={'size': 19}``. To specify depth axes with
            size 40 or more, you could set: ``axes_superset={'axes':
            'depth', 'size': cf.ge(40)}`` (see `cf.ge`).''',
    
    
    # ----------------------------------------------------------------
    '{+axes_all}': '''axes_all: optional
        Select items whose data arrays span all of the specified axes,
        taken in any order, and no others. The axes are those that
        would be selected by this call of the field's `~cf.Field.axes`
        method: ``f.axes(axes_all)`` or, if *axes_all* is a dictionary
        of parameters, ``f.axes(**axes_all)``. Axes are defined by
        identfiying items of the field (such as dimension coordinate
        objects) or by specifying axis sizes. In the former case the
        selected axes are those which span the identified field
        items. See `cf.Field.axes` for details.
    
          *Example:*
            To select items which span a time axis, and no others, you
            could set: ``axes_all='T'``.
            
          *Example:*
            To select items which span latitude and longitude axes,
            and no others, you could set: ``axes_all=['X', 'Y']``.
            
          *Example:*
            To specify axes with size 19 you could set
            ``axes_all={'size': 19}``. To specify depth axes with size
            40 or more, you could set: ``axes_all={'axes': 'depth',
            'size': cf.ge(40)}`` (see `cf.ge`).''',
    
    # ----------------------------------------------------------------
    '{+role}': '''role: (sequence of) `str`, optional
        Select items of the given roles. Valid roles are:
    
          =======  ==========================
          Role     Items selected
          =======  ==========================
          ``'d'``  Dimension coordinate items
          ``'a'``  Auxiliary coordinate items
          ``'m'``  Cell measure items
          ``'c'``  Domain ancillary items
          ``'f'``  Field ancillary items
          ``'r'``  Coordinate reference items
          =======  ==========================
    
        Multiple roles may be specified by a sequence of role
        identifiers. By default all roles except coordinate reference
        items are considered, i.e. by default ``role=('d', 'a', 'm',
        'f', 'c')``.
    
          *Example:*
            To select dimension coordinate items: ``role='d'`` or
            ``role=['d']`.

          *Example:*
            Selecting auxiliary coordinate and cell measure items may
            be done with any of the following values of *role*:
            ``'am'``, ``'ma'``, ``('a', 'm')``, ``['m', 'a']``,
            ``set(['a', 'm'])``, etc.''',
       
    # ----------------------------------------------------------------
    '{+exact}': '''exact: `bool`, optional
        The *exact* parameter applies to the interpretation of
        string-valued conditions given by the *description*
        parameter. By default *exact* is False, which means that:
    
          * A string-valued condition is treated as a regular
            expression understood by the `re` module and an item is
            selected if its corresponding value matches the regular
            expression using the `re.match` method (i.e. if zero or
            more characters at the **beginning** of item's value match
            the regular expression pattern).
          
          * Units and calendar strings are evaluated for equivalence
            rather then equality (e.g. ``'metre'`` is equivalent to
            ``'m'`` and to ``'km'``).
    
        ..
    
          *Example:*
            To select items with with any units of pressure:
            ``f.{+name}('units:hPa')``. To select items with a
            standard name which begins with "air" and with any units
            of pressure: ``f.{+name}({'standard_name': 'air', 'units':
            'hPa'})``.
    
        If *exact* is True then:
    
          * A string-valued condition is not treated as a regular
            expression and an item is selected if its corresponding
            value equals the string.
    
          * Units and calendar strings are evaluated for exact
            equality rather than equivalence (e.g. ``'metre'`` is
            equal to ``'m'``, but not to ``'km'``).
    
        ..
    
          *Example:*
            To select items with with units of hectopascals but not,
            for example, Pascals: ``f.{+name}('units:hPa',
            exact=True)``. To select items with a standard name of
            exactly "air_pressure" and with units of exactly
            hectopascals: ``f.{+name}({'standard_name':
            'air_pressure', 'units': 'hPa'}, exact=True)``.
    
        Note that `cf.Query` objects provide a mechanism for
        overriding the *exact* parameter for individual values.
    
          *Example:*
            ``f.{+name}({'standard_name': cf.eq('air', exact=False),
            'units': 'hPa'}, exact=True)`` will select items with a
            standard name which begins "air" but with units of exactly
            hectopascals (see `cf.eq`).
    
          *Example:*
            ``f.{+name}({'standard_name': cf.eq('air_pressure'),
            'units': 'hPa'})`` will select items with a standard name
            of exactly "air_pressure" but with any units of pressure
            (see `cf.eq`).''',
    
    # ----------------------------------------------------------------
    '{+match_and}': '''match_and: `bool`, optional
        By default *match_and* is True and items are selected if they
        satisfy the all of the specified conditions.
        
        If *match_and* is False then items are selected if they
        satisfy at least one of the specified conditions.
    
          *Example:*
            To select items with identity beginning with "ocean"
            **and** two data array axes: ``f.{+name}('ocean',
            ndim=2)``.
    
          *Example:*
            To select items with identity beginning with "ocean"
            **or** two data array axes: ``f.{+name}('ocean', ndim=2,
            match_and=False)``.''',
    
    # ----------------------------------------------------------------
    '{+inverse}': '''inverse: `bool`, optional
        If True then select items other than those selected by all
        other criteria.''',
    
    # ----------------------------------------------------------------
    '{+copy}': '''copy: `bool`, optional
        If True then a returned item is a copy. By default it is not
        copied.''',
    
    # ----------------------------------------------------------------
    '{+bounds}': '''bounds: `bool`, optional
         If False then do not alter the {+variable}'s bounds, if it
         has any. By default any bounds are also altered.''',

    # ----------------------------------------------------------------
    '{+key}': '''key: `bool`, optional
        If True then return the domain's identifier for the selected
        item, rather than the item itself.''',
    
    # ----------------------------------------------------------------
    '{+description}': '''description: optional
        Select the items whose descriptive attributes or CF properties
        satisfy the given conditions. The *description* parameter may
        be one, or a sequence, of:
    
          * `None` or an empty dictionary. All items are selected. This
            is the default.
        
     ..
        
        * A string specifying one of the CF coordinate types: ``'T'``,
          ``'X'``, ``'Y'`` or ``'Z'``. An item has an attribute for
          each coordinate type and is selected if the attribute for
          the specified type is True.
        
            *Example:*
              To select CF time items: ``description='T'``.
        
      ..
        
        * A string which identifies items based on their string-valued
          metadata. The value may take one of the following forms:
        
            ==============  ========================================
            Value           Interpretation
            ==============  ========================================
            Contains ``:``  Selects on the CF property specified
                            before the first ``:``
            
            Contains ``%``  Selects on the attribute specified
                            before the first ``%``
            
            Anything else   Selects on identity as returned by an
                            item's `!identity` method
            ==============  ========================================
          
          By default the part of the string to be compared with an
          item is treated as a regular expression understood by the
          :py:obj:`re` module and an item is selected if its
          corresponding value matches the regular expression using the
          :py:obj:`re.match` method (i.e. if zero or more characters
          at the **beginning** of item's value match the regular
          expression pattern). See the *exact* parameter for details.
          
          *Example:*
            To select items with standard names which begin "lat":
            ``description='lat'``.
        
          *Example:*
            To select items with long names which begin "air":
            ``description='long_name:air'``.
          
          *Example:*
            To select items with netCDF variable names which begin
            "lon": ``description='ncvar%lon'``.
          
          *Example:*
            To select items with identities which end with the
            letter "z": ``description='.*z$'``.
          
          *Example:*
            To select items with long names which start with the
            string ".*a": ``description='long_name%\.\*a'``.

      ..
        
        * A dictionary that identifies properties of the items
          with corresponding tests on their values. An item is
          selected if **all** of the tests in the dictionary are
          passed.
        
          In general, each dictionary key is a CF property name with
          a corresponding value to be compared against the item's CF
          property value.
          
          If the dictionary value is a string then by default it is
          treated as a regular expression understood by the
          :py:obj:`re` module and an item is selected if its
          corresponding value matches the regular expression using
          the :py:obj:`re.match` method (i.e. if zero or more
          characters at the **beginning** of item's value match the
          regular expression pattern). See the *exact* parameter for
          details.
          
          *Example:*
            To select items with standard name of exactly
            "air_temperature" and long name beginning with the letter
            "a": ``description={'standard_name':
            cf.eq('air_temperature'), 'long_name': 'a'}`` (see
            `cf.eq`).
          
          Some key/value pairs have a special interpretation:
          
            ==================  ====================================
            Special key         Value
            ==================  ====================================
            ``'units'``         The value must be a string and by
                                default is evaluated for
                                equivalence, rather than equality,
                                with an item's `units` property,
                                for example a value of ``'Pa'``
                                will match units of Pascals or
                                hectopascals, etc. See the *exact*
                                parameter.
            
            ``'calendar'``      The value must be a string and by
                                default is evaluated for
                                equivalence, rather than equality,
                                with an item's `calendar`
                                property, for example a value of
                                ``'noleap'`` will match a calendar
                                of noleap or 365_day. See the
                                *exact* parameter.
          
            `None`              The value is interpreted as for a
                                string value of the *description*
                                parameter. For example,
                                ``description={None: 'air'}`` is
                                equivalent to ``description='air'``,
                                ``description={None:
                                'ncvar%pressure'}`` is equivalent to
                                ``description='ncvar%pressure'`` and
                                ``description={None: 'Y'}`` is
                                equivalent to ``description='Y'``.
            ==================  ====================================
        
            *Example:*
              To select items with standard name starting with
              "air", units of temperature and a netCDF variable name
              of "tas" you could set ``description={'standard_name':
              'air', 'units': 'K', None: 'ncvar%tas$'}``.
        
        ..
    
          * A domain item identifier (such as ``'dim1'``, ``'aux0'``,
            ``'msr2'``, ``'ref0'``, etc.). Selects the corresponding
            item.  
        
              *Example:*
                To select the item with domain identifier "dim1":
                ``description='dim1'``.
        
        If *description* is a sequence of any combination of the above then
        the selected items are the union of those selected by each
        element of the sequence. If the sequence is empty then no
        items are selected.''',
    
    # ----------------------------------------------------------------
    '{+axes, kwargs}': '''axes, kwargs: optional
        Select axes. The *axes* parameter may be one, or a sequence,
        of:
    
          * `None`. If no *kwargs* arguments have been set
            then all axes are selected. This is the default.
    
        ..
    
          * An integer. Explicitly selects the axis corresponding to
            the given position in the list of axes of the field's data
            array.
    
              *Example:*
                To select the third data array axis: ``axes=2``. To
                select the last axis: ``axes=-1``.
    
        ..
    
          * A :py:obj:`slice` object. Explicitly selects the axes
            corresponding to the given positions in the list of axes
            of the field's data array.
          
              *Example:* 
                To select the last three data array axes:
                ``axes=slice(-3, None)``
   
        ..
      
          * A domain axis identifier. Explicitly selects this axis.
      
             *Example:*
               To select axis "dim1": ``axes='dim1'``.
    
        ..
    
          * Any value accepted by the *description* parameter of the field's
            `items` method. Used in conjunction with the *kwargs*
            parameters to select the axes which span the items that
            would be identified by this call of the field's `items`
            method: ``f.items(items=axes, axes=None, **kwargs)``. See
            `cf.Field.items` for details.
          
              *Example:*
                To select the axes spanned by one dimensionsal time
                coordinates: ``f.{+name}('T', ndim=1)``.
        
        If *axes* is a sequence of any combination of the above then
        the selected axes are the union of those selected by each
        element of the sequence. If the sequence is empty then no axes
        are selected.''',
    
    # ----------------------------------------------------------------
    '{+size}': '''size: optional
        Select axes whose sizes equal *size*. Axes whose sizes lie
        within a range sizes may be selected if *size* is a `cf.Query`
        object.
          
          *Example:*        
            ``size=1`` selects size 1 axes.
        
          *Example:*
            ``size=cf.ge(2)`` selects axes with sizes greater than 1
            (see `cf.ge`).''',
    
    # ----------------------------------------------------------------
    '{+i}': '''i: `bool`, optional
        If True then update the {+variable} in place. By default a new
        {+variable} is created. In either case, a {+variable} is
        returned.''',
    
    # ----------------------------------------------------------------
}

p = re_compile('(?<=.)([A-Z])') # E.g. DimensionCoordinate or Variable
one  = re_compile('(\[\+N\].*\n|\[\+1\])')
many = re_compile('(\[\+1\].*\n|\[\+N\])')
zero = re_compile('\[\+0\]')
#first_char = re.compile('^(\s|\n)*(.)')
fef = re_compile('{\+.*?[Ff]ef,?}(.)')
fefx = re_compile('{\+(.*?)([Ff])ef,?}(.)')

def _replacement(match):
    '''Used in a `re.sub` for for replacing:

'{+Fef,}X' with 'For each field, x'
 and 
'{+,fef,}' with ', for each field,'
'''
    comma = match.group(1)
    if comma:
        comma += ' '
    return  comma+match.group(2)+"or each field, "+match.group(3).lower()     

def _replacement0(match):
    '''Used in a `re.sub` for for replacing:

'{+Fef,}X' with 'For each field, X'
 and 
'{+,fef,}' with ', for each field,'
'''
    return  match.group(1)

def _update_docstring(name, f, attr_name):
    '''
    
'''
    doc = f.__doc__
    if doc is None:
        return

    name_lc = p.sub(r' \1', name).lower()

    doc = doc.replace('{+name}'    , attr_name)
    doc = doc.replace('{+Variable}', name)
    doc = doc.replace('{+variable}', name_lc)

    kwargs = {}
    for arg in set(re_findall('{\+.*?}', doc)):
        
        if arg in ('{+Fef,}', '{+,fef,}', '{+fef}'):
            continue
        
        if arg == '{+bounds}':
            if name not in ('DimensionCoordinate',):
                doc = doc.replace('{+bounds}',
                                  'bounds: optional\n        Ignored.')
                continue

        ds = docstring[arg].replace('{+name}', attr_name)
        ds = ds.replace('{+Variable}', name)
        ds = ds.replace('{+variable}', name_lc)

#        if attr_name.endswith('s'):
#            ds = ds.replace('{+s}', 's')
#        else:
#            ds = ds.replace('{+s}', '')

        doc = doc.replace(arg, ds)
    #--- End: for

    if name == 'FieldList':
        doc = many.sub('', doc)
        doc = zero.sub('[0]', doc)
        doc = fefx.sub(_replacement, doc)
    else:
        doc = one.sub('', doc)
        doc = zero.sub('', doc)
        doc = fef.sub(_replacement0, doc)

    f.__doc__ = doc 
#--- End: def

class RewriteDocstringMeta(type):
    '''Modify docstrings.

To do this, we intercede before the class is created and modify the
docstrings of its attributes.

This will not affect inherited methods, however, so we also need to
loop through the parent classes. We cannot simply modify the
docstrings, because then the parent classes' methods will have the
wrong docstring. Instead, we must actually copy the functions, and
then modify the docstring.

    '''
 # http://www.jesshamrick.com/2013/04/17/rewriting-python-docstrings-with-a-metaclass/

 
    def __new__(cls, name, parents, attrs):
        
        for attr_name in attrs:
            # skip special methods
            if attr_name.startswith('__'):
                continue
    
            # skip non-functions
            attr = attrs[attr_name]
            if not hasattr(attr, '__call__'):                
                continue

            if not hasattr(attr, 'func_doc'):
                continue

            # update docstring
            _update_docstring(name, attr, attr_name)
 
        for parent in parents:
            for attr_name in dir(parent):
                # we already have this method
                if attr_name in attrs:
                    continue
 
                # skip special methods
                if attr_name.startswith('__'):
                    continue
 
                # get the original function and copy it
                a = getattr(parent, attr_name)
 
                # skip non-functions
                if not hasattr(a, '__call__'):
                    continue

                f = getattr(a, '__func__', None)
                if f is None:
                    continue

                # copy function
                attr = type(f)(
                    f.func_code, f.func_globals, f.func_name,
                    f.func_defaults, f.func_closure)
                doc = f.__doc__

                # update docstring and add attr
                _update_docstring(name, attr, attr_name)
                attrs[attr_name] = attr
 
        # create the class
        obj = super(RewriteDocstringMeta, cls).__new__(
            cls, name, parents, attrs)

        return obj
    #--- End: def

#--- End: class


class DeprecationError(Exception):
    '''Exception for removed methods'''
    pass


# ====================================================================
#
# Variable object
#
# ====================================================================

class Variable(object):
    '''

Base class for storing a data array with metadata.

A variable contains a data array and metadata comprising properties to
describe the physical nature of the data.

All components of a variable are optional.

'''

    __metaclass__ = RewriteDocstringMeta

#    # Do not ever change this:
#    _list = False

    # Define the reserved attributes. These are methods which can't be
    # overwritten, as well as a few attributes.
    _reserved_attrs = ('_reserved_attrs',
                       '_insert_data'
                       '_set_Data_attributes',
                       'binary_mask',
                       'chunk',
                       'clip',
                       'copy',
                       'cos',
                       'delprop',
                       'dump',
                       'equals',
                       'expand_dims',
                       'flip',
                       'getprop',
                       'hasprop',
                       'identity',
                       'match',
                       'name',
                       'override_units',
                       'select',
                       'setprop',
                       'sin',
                       'subspace',
                       'transpose',
                       'where',
                       )

    _special_properties = set(('units', 'calendar',
                               '_FillValue', 'missing_value'))

    def __init__(self, properties={}, attributes=None, data=None,
                 source=None, copy=True):
        '''**Initialization**

:Parameters:

    properties: `dict`, optional
        Initialize CF properties from the dictionary's key/value
        pairs.

    attributes: `dict`, optional
        Provide attributes from the dictionary's key/value pairs.

    data: `cf.Data`, optional
        Provide a data array.
        
    source: `cf.{+Variable}`, optional
        Take the attributes, CF properties and data array from the
        source {+variable}. Any attributes, CF properties or data
        array specified with other parameters are set after
        initialisation from the source {+variable}.

    copy: `bool`, optional
        If False then do not deep copy arguments prior to
        initialization. By default arguments are deep copied.

        '''
        self._fill_value = None

        # _hasbounds is True if and only if there are cell bounds.
        self._hasbounds = False

        self._direction = None

        # True if and only if there is a data array stored in
        # self.Data
        self._hasData = False

        # Initialize the _private dictionary with an empty Units
        # object
        self._private = {'special_attributes': {},
                         'simple_properties' : {},
        }
        
        if source is not None:
            if data is None:
                data = Data.asdata(source)

            if isinstance(source, Variable):
                p = source.properties()
                if properties:
                    p.update(properties)
                properties = p

                a = source.attributes()
                if attributes:
                    a.update(attributes) 
                attributes = a
        #--- End: if

        if properties:
            self.properties(properties, copy=copy)

        if attributes:
            self.attributes(attributes, copy=copy)

        if data is not None:
            self.insert_data(data, copy=copy)
        else:   
            # _hasData is True if and only if there is a data array
            # stored in self.Data
            self._hasData = False
    #--- End: def

    def __array__(self, *dtype):
        '''
'''
        if self._hasData:
            return self.data.__array__(*dtype)

        raise ValueError("%s has no numpy.ndarray interface'" %
                         self.__class__.__name__)
    #--- End: def

    def __contains__(self, value):
        '''

Called to implement membership test operators.

x.__contains__(y) <==> y in x

'''
        if not self._hasData:
            return False
        
        return value in self.data
    #--- End: def

    def __data__(self):
        '''
Returns a new reference to self.data.
'''
        if self._hasData:
            return self.data

        raise ValueError(
            "{0} has no Data interface".format(self.__class__.__name__))
    #--- End: def

    def __deepcopy__(self, memo):
        '''

Called by the :py:obj:`copy.deepcopy` standard library function.

'''
        return self.copy()
    #--- End: def

    def __delattr__(self, attr):
        '''

x.__delattr__(attr) <==> del x.attr

'''
        if attr in self._reserved_attrs:
            raise AttributeError("Can't delete reserved attribute %r" % attr)

        super(Variable, self).__delattr__(attr)
    #--- End: def

    def __getitem__(self, indices):
        '''

Called to implement evaluation of x[indices].

x.__getitem__(indices) <==> x[indices]

'''
        new = self.copy(_omit_Data=True)

        if self._hasData:
            new.Data = self.Data[indices]

        return new
    #--- End: def

#    def __len__(self):
#        '''
#
#Called by the :py:obj:`len` built-in function.
#
#x.__len__() <==> len(x)
#
#Always returns 1.
#
#:Examples:
#
#>>> len(f)
#1
#
#'''
#        return 1
#    #--- End: def

    def __setitem__(self, indices, value):
        '''

Called to implement assignment to x[indices]

x.__setitem__(indices, value) <==> x[indices]

'''
        if isinstance(value, Variable):
            value = value.Data

        self.Data[indices] = value
    #--- End: def

    def __add__(self, y):
        '''

The binary arithmetic operation ``+``

x.__add__(y) <==> x+y

'''        
        return self._binary_operation(y, '__add__')
    #--- End: def

    def __iadd__(self, y):
        '''

The augmented arithmetic assignment ``+=``

x.__iadd__(y) <==> x+=y

'''
        return self._binary_operation(y, '__iadd__')
    #--- End: def

    def __radd__(self, y):
        '''

The binary arithmetic operation ``+`` with reflected operands

x.__radd__(y) <==> y+x

'''
        return self._binary_operation(y, '__radd__')
    #--- End: def

    def __sub__(self, y):
        '''

The binary arithmetic operation ``-``

x.__sub__(y) <==> x-y

'''
        return self._binary_operation(y, '__sub__')
    #--- End: def

    def __isub__(self, y):
        '''

The augmented arithmetic assignment ``-=``

x.__isub__(y) <==> x-=y

'''
        return self._binary_operation(y, '__isub__')
    #--- End: def

    def __rsub__(self, y):
        '''

The binary arithmetic operation ``-`` with reflected operands

x.__rsub__(y) <==> y-x

'''    
        return self._binary_operation(y, '__rsub__')
    #--- End: def

    def __mul__(self, y):
        '''

The binary arithmetic operation ``*``

x.__mul__(y) <==> x*y

'''
        return self._binary_operation(y, '__mul__')
    #--- End: def

    def __imul__(self, y):
        '''

The augmented arithmetic assignment ``*=``

x.__imul__(y) <==> x*=y

'''
        return self._binary_operation(y, '__imul__')
    #--- End: def

    def __rmul__(self, y):
        '''

The binary arithmetic operation ``*`` with reflected operands

x.__rmul__(y) <==> y*x

'''       
        return self._binary_operation(y, '__rmul__')
    #--- End: def

    def __div__(self, y):
        '''

The binary arithmetic operation ``/``

x.__div__(y) <==> x/y

'''
        return self._binary_operation(y, '__div__')
    #--- End: def

    def __idiv__(self, y):
        '''

The augmented arithmetic assignment ``/=``

x.__idiv__(y) <==> x/=y

'''
        return self._binary_operation(y, '__idiv__')
    #--- End: def

    def __rdiv__(self, y):
        '''

The binary arithmetic operation ``/`` with reflected operands

x.__rdiv__(y) <==> y/x

'''
        return self._binary_operation(y, '__rdiv__')
    #--- End: def

    def __floordiv__(self, y):
        '''

The binary arithmetic operation ``//``

x.__floordiv__(y) <==> x//y

'''     
        return self._binary_operation(y, '__floordiv__')
    #--- End: def

    def __ifloordiv__(self, y):
        '''

The augmented arithmetic assignment ``//=``

x.__ifloordiv__(y) <==> x//=y

'''
        return self._binary_operation(y, '__ifloordiv__')
    #--- End: def

    def __rfloordiv__(self, y):
        '''

The binary arithmetic operation ``//`` with reflected operands

x.__rfloordiv__(y) <==> y//x

'''
        return self._binary_operation(y, '__rfloordiv__')
    #--- End: def

    def __truediv__(self, y):
        '''

The binary arithmetic operation ``/`` (true division)

x.__truediv__(y) <==> x/y

'''
        return self._binary_operation(y, '__truediv__')
    #--- End: def

    def __itruediv__(self, y):
        '''

The augmented arithmetic assignment ``/=`` (true division)

x.__itruediv__(y) <==> x/=y

'''
        return self._binary_operation(y, '__itruediv__')
   #--- End: def

    def __rtruediv__(self, y):
        '''

The binary arithmetic operation ``/`` (true division) with reflected
operands

x.__rtruediv__(y) <==> y/x

'''    
        return self._binary_operation(y, '__rtruediv__')
    #--- End: def

    def __pow__(self, y, modulo=None):
        '''

The binary arithmetic operations ``**`` and ``pow``

x.__pow__(y) <==> x**y

'''  
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__pow__')
    #--- End: def

    def __ipow__(self, y, modulo=None):
        '''

The augmented arithmetic assignment ``**=``

x.__ipow__(y) <==> x**=y

'''     
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__ipow__')
    #--- End: def

    def __rpow__(self, y, modulo=None):
        '''

The binary arithmetic operations ``**`` and ``pow`` with reflected
operands

x.__rpow__(y) <==> y**x

'''       
        if modulo is not None:
            raise NotImplementedError("3-argument power not supported for %r" %
                                      self.__class__.__name__)

        return self._binary_operation(y, '__rpow__')
    #--- End: def

    def __mod__(self, y):
        '''

The binary arithmetic operation ``%``

x.__mod__(y) <==> x % y

.. versionadded:: 1.0

'''
        return self._binary_operation(y, '__mod__')
    #--- End: def

    def __imod__(self, y):
        '''

The binary arithmetic operation ``%=``

x.__imod__(y) <==> x %= y

.. versionadded:: 1.0

'''
        return self._binary_operation(y, '__imod__')
    #--- End: def

    def __rmod__(self, y):
        '''

The binary arithmetic operation ``%`` with reflected operands

x.__rmod__(y) <==> y % x

.. versionadded:: 1.0

'''
        return self._binary_operation(y, '__rmod__')
    #--- End: def

    def __eq__(self, y):
        '''

The rich comparison operator ``==``

x.__eq__(y) <==> x==y

'''
        return self._binary_operation(y, '__eq__')
    #--- End: def

    def __ne__(self, y):
        '''

The rich comparison operator ``!=``

x.__ne__(y) <==> x!=y

'''
        return self._binary_operation(y, '__ne__')
    #--- End: def

    def __ge__(self, y):
        '''

The rich comparison operator ``>=``

x.__ge__(y) <==> x>=y

'''
        return self._binary_operation(y, '__ge__')
    #--- End: def

    def __gt__(self, y):
        '''

The rich comparison operator ``>``

x.__gt__(y) <==> x>y

'''
        return self._binary_operation(y, '__gt__')
    #--- End: def

    def __le__(self, y):
        '''

The rich comparison operator ``<=``

x.__le__(y) <==> x<=y

'''
        return self._binary_operation(y, '__le__')
    #--- End: def

    def __lt__(self, y):
        '''

The rich comparison operator ``<``

x.__lt__(y) <==> x<y

'''
        return self._binary_operation(y, '__lt__')
    #--- End: def

    def __and__(self, y):
        '''

The binary bitwise operation ``&``

x.__and__(y) <==> x&y

'''
        return self._binary_operation(y, '__and__')
    #--- End: def

    def __iand__(self, y):
        '''

The augmented bitwise assignment ``&=``

x.__iand__(y) <==> x&=y

'''
        return self._binary_operation(y, '__iand__')
    #--- End: def

    def __rand__(self, y):
        '''

The binary bitwise operation ``&`` with reflected operands

x.__rand__(y) <==> y&x

'''
        return self._binary_operation(y, '__rand__')
    #--- End: def

    def __or__(self, y):
        '''

The binary bitwise operation ``|``

x.__or__(y) <==> x|y

'''
        return self._binary_operation(y, '__or__')
    #--- End: def

    def __ior__(self, y):
        '''

The augmented bitwise assignment ``|=``

x.__ior__(y) <==> x|=y

'''
        return self._binary_operation(y, '__ior__')
    #--- End: def

    def __ror__(self, y):
        '''

The binary bitwise operation ``|`` with reflected operands

x.__ror__(y) <==> y|x

'''
        return self._binary_operation(y, '__ror__')
    #--- End: def

    def __xor__(self, y):
        '''

The binary bitwise operation ``^``

x.__xor__(y) <==> x^y

'''
        return self._binary_operation(y, '__xor__')
    #--- End: def

    def __ixor__(self, y):
        '''

The augmented bitwise assignment ``^=``

x.__ixor__(y) <==> x^=y

'''
        return self._binary_operation(y, '__ixor__')
    #--- End: def

    def __rxor__(self, y):
        '''

The binary bitwise operation ``^`` with reflected operands

x.__rxor__(y) <==> y^x

'''
        return self._binary_operation(y, '__rxor__')
    #--- End: def

    def __lshift__(self, y):
        '''

The binary bitwise operation ``<<``

x.__lshift__(y) <==> x<<y

'''
        return self._binary_operation(y, '__lshift__')
    #--- End: def

    def __ilshift__(self, y):
        '''

The augmented bitwise assignment ``<<=``

x.__ilshift__(y) <==> x<<=y

'''
        return self._binary_operation(y, '__ilshift__')
    #--- End: def

    def __rlshift__(self, y):
        '''

The binary bitwise operation ``<<`` with reflected operands

x.__rlshift__(y) <==> y<<x
'''
        return self._binary_operation(y, '__rlshift__')
    #--- End: def

    def __rshift__(self, y):
        '''

The binary bitwise operation ``>>``

x.__lshift__(y) <==> x>>y

'''
        return self._binary_operation(y, '__rshift__')
    #--- End: def

    def __irshift__(self, y):
        '''

The augmented bitwise assignment ``>>=``

x.__irshift__(y) <==> x>>=y
'''
        return self._binary_operation(y, '__irshift__')
    #--- End: def

    def __rrshift__(self, y):
        '''

The binary bitwise operation ``>>`` with reflected operands

x.__rrshift__(y) <==> y>>x

'''
        return self._binary_operation(y, '__rrshift__')
    #--- End: def

    def __abs__(self):
        '''

The unary arithmetic operation ``abs``

x.__abs__() <==> abs(x)

'''       
        return self._unary_operation('__abs__')
    #--- End: def

    def __neg__(self):
        '''

The unary arithmetic operation ``-``

x.__neg__() <==> -x

'''
        return self._unary_operation('__neg__')
    #--- End: def

    def __invert__(self):
        '''

The unary bitwise operation ``~``

x.__invert__() <==> ~x

'''
        return self._unary_operation('__invert__')
    #--- End: def

    def __pos__(self):
        '''

The unary arithmetic operation ``+``

x.__pos__() <==> +x

'''
        return self._unary_operation('__pos__')
    #--- End: def

    def __repr__(self):
        '''
Called by the :py:obj:`repr` built-in function.

x.__repr__() <==> repr(x)

'''
        name = self.name('')

        if self._hasData:
            dims = ', '.join([str(x) for x in self.shape])            
        else:
            dims = []
        dims = '(%s)' % dims

        # Units
        if self.Units._calendar:
            units = self.Units._calendar
        else:
            units = getattr(self, 'units', '')

        return '<CF %s: %s%s %s>' % (self.__class__.__name__,
                                    self.name(''), dims, units)
    #--- End: def

    def __str__(self):
        '''

Called by the :py:obj:`str` built-in function.

x.__str__() <==> str(x)

'''
        return self.__repr__()
    #--- End: def

    # ================================================================
    # Private methods
    # ================================================================
    def _binary_operation(self, y, method):
        '''Implement binary arithmetic and comparison operations.

The operations act on the {+variable}'s data array with the numpy
broadcasting rules.

It is intended to be called by the binary arithmetic and comparison
methods, such as `!__sub__` and `!__lt__`.

:Parameters:

    operation: `str`
        The binary arithmetic or comparison method name (such as
        ``'__imul__'`` or ``'__ge__'``).

:Returns:

    out: `cf.{+Variable}`
        A new {+variable}, or the same {+variable} if the operation
        was in-place.

:Examples:

>>> u = cf.{+Variable}(data=cf.Data([0, 1, 2, 3]))
>>> v = cf.{+Variable}(data=cf.Data([1, 1, 3, 4]))

>>> w = u._binary_operation(u, '__add__')
>>> print w.array
[1 2 5 7]

>>> w = u._binary_operation(v, '__lt__')
>>> print w.array
[ True  False  True  True]

>>> u._binary_operation(2, '__imul__')
>>> print u.array
[0 2 4 6]

        '''
        if not self._hasData:
            raise ValueError( 
                "Can't apply {} to a {} object with no Data: {!r}".format(
                    method, self.__class__.__name__, self))

        inplace = method[2] == 'i'

        xsn = self.getprop('standard_name', None)
        ysn = getattr(y, 'standard_name', None)

        x_Units = self.Units
        y_Units = getattr(y, 'Units', _units_None)

        if isinstance(y, self.__class__):
            y = y.Data

        if not inplace:
            new      = self.copy(_omit_Data=True)
            new.Data = self.Data._binary_operation(y, method)

            #if not new.Data.Units.equivalent(original_units):
            #    # this is coarse!
            #    new.delprop('standard_name')
            #
            #    if hasattr(new, 'history'):
            #        history = [new.getprop('history')]
            #    else:
            #        history = []
            #
            #    history.append(new.getprop('standard_name'))
            #    history.append(method)
            #
            #    new.setprop('history', ' '.join(history))
            ##--- End: if

#            return new

        else:
            new = self
            new.Data._binary_operation(y, method)
        #--- End: def

        new_Units = new.Data.Units
        if (not (new_Units.equivalent(x_Units) or
                 new_Units.equivalent(y_Units)) or
            xsn is not None and ysn is not None and xsn != ysn):
            try:
                new.delprop('standard_name')
            except AttributeError:
                pass
            try:
                new.delprop('long_name')
            except AttributeError:
                pass
            try:
                del new.ncvar
            except AttributeError:
                pass
            try:
                del new.id
            except AttributeError:
                pass
        #--- End: if

        return new
    #--- End: def

    def _change_axis_names(self, dim_name_map):
        '''Change the axis names of the Data object.

:Parameters:

    dim_name_map: `dict`

:Returns:

    `None`

:Examples:

>>> f._change_axis_names({'0': 'dim1', '1': 'dim2'})

        '''
        if self._hasData:
            self.Data.change_axis_names(dim_name_map)
    #--- End: def

    def _conform_for_assignment(self, other):
        return other

    def _del_special_attr(self, attr):
        '''

'''    
        d = self._private['special_attributes']
        if attr in d:
            del d[attr]
            return
            
        raise AttributeError("Can't delete non-existent %s attribute %r" %
                             (self.__class__.__name__, attr))
    #--- End: def

    def _dump_simple_properties(self, omit=(), _level=0):
        '''

:Parameters:

    omit: sequence of `str`, optional
        Omit the given CF properties from the description.

    _level: `int`, optional

:Returns:

    out: `str`

:Examples:

'''
        indent0 = '    ' * _level

        string = []

        # Simple properties
        simple = self._simple_properties()
        attrs  = sorted(set(simple) - set(omit))
        for attr in attrs:
            name   = '{}{} = '.format(indent0, attr)
            value  = repr(simple[attr])
            indent = ' ' * (len(name))
            if value.startswith("'") or value.startswith('"'):
                indent = '%(indent)s ' % locals()

            string.append(textwrap_fill(name+value, 79,
                                        subsequent_indent=indent))
        #--- End: for

        return '\n'.join(string)
    #--- End: def

    def _equivalent_data(self, other, atol=None, rtol=None, traceback=False):
        '''
:Parameters:

    transpose: `dict`, optional

    {+atol}

    {+rtol}

:Returns:

    out: `bool`
        Whether or not the two variables have equivalent data arrays.

'''
        if self._hasData != other._hasData:
            # add traceback
            return False

        if not self._hasData:
            return True

        data0 = self.data
        data1 = other.data

        if data0._shape != data1._shape:
            # add traceback
            return False              
 
        if not data0.Units.equivalent(data1.Units):
            # add traceback
            if traceback:
                print repr(data0.Units), repr(data1.Units)
                print 'BAD UNITS'
            return  False

        if atol is None:
            atol = ATOL()        
        if rtol is None:
            rtol = RTOL()
            
        if not data0.allclose(data1, rtol=rtol, atol=atol):
            # add traceback
            return False

        return True
    #--- End: def

    def _get_special_attr(self, attr):
        '''

'''
        d = self._private['special_attributes']
        if attr in d:
            return d[attr]

        raise AttributeError("%s doesn't have attribute %r" %
                             (self.__class__.__name__, attr))
    #--- End: def

#    def _list_attribute(self, attr):
#        return type(self)([getattr(f, attr) for f in self._list])
#    #--- End: def

#    def _list_method(self, method, kwargs={}):
#        if 'i' in kwargs and kwargs['i']:
#            # In-place
#            for f in self:
#                getattr(f, method)(**kwargs)
#            return self
#        else:
#            # New instance
#            return type(self)([getattr(f, method)(**kwargs) for f in self])
#    #--- End: def

    def _parameters(self, d):
        del d['self']
        if 'kwargs' in d:
            d.update(d.pop('kwargs'))
        return d
    #--- End: def

    def _parse_axes(self, axes):
        if axes is None:
            return axes

        ndim = self.ndim
        return [(i + ndim if i < 0 else i) for i in axes]
    #--- End: def
    
    def _parse_match(self, match):
        '''Called by `match`

:Parameters:

    match: 
        As for the *match* parameter of `match` method.

:Returns:

    out: `list`
        '''        
        if not match:
            return ()

        if isinstance(match, (basestring, dict, Query)):
            match = (match,)

        matches = []
        for m in match:            
            if isinstance(m, basestring):
                if ':' in m:
                    # CF property (string-valued)
                    m = m.split(':')
                    matches.append({m[0]: ':'.join(m[1:])})
                else:
                    # Identity (string-valued) or python attribute
                    # (string-valued) or axis type
                    matches.append({None: m})

            elif isinstance(m, dict):
                # Dictionary
                matches.append(m)

            else:
                # Identity (not string-valued, e.g. cf.Query).
                matches.append({None: m})
        #--- End: for

        return matches
    #--- End: def

    def _query_set(self, values, exact=True):
        '''
'''
        kwargs2 = self._parameters(locals())

#        if self._list:
#            return self._list_method('_query_set', kwargs2)

        new = self.copy(_omit_Data=True)
        new.Data = self.Data._query_set(**kwargs2)
        return new
    #--- End: def

    def _query_contain(self, value):
        '''
'''
        kwargs2 = self._parameters(locals())

#        if self._list:
#            return self._list_method('_query_contain', kwargs2)

        new = self.copy(_omit_Data=True)
        new.Data = self.Data._query_contain(**kwargs2)
        return new
    #--- End: def

    def _query_wi(self, value0, value1):
        '''
'''
#        kwargs2 = self._parameters(locals())
#
#        if self._list:
#            return self._list_method('_query_wi', kwargs2)

        new = self.copy(_omit_Data=True)
        new.Data = self.Data._query_wi(value0, value1)
        return new
    #--- End: def

    def _query_wo(self, value0, value1):
        '''
'''
#        kwargs2 = self._parameters(locals())
#
#        if self._list:
#            return self._list_method('_query_wo', kwargs2)

        new = self.copy(_omit_Data=True)
        new.Data = self.Data._query_wo(value0, value1)
        return new
    #--- End: def

    def _simple_properties(self):
        '''
        '''        
        return self._private['simple_properties']
    #--- End: def

    def _set_special_attr(self, attr, value):
        '''
        '''
        self._private['special_attributes'][attr] = value
    #--- End: def

    def _unary_operation(self, method):
        '''Implement unary arithmetic operations on the data array.

:Parameters:

    method: `str`
        The unary arithmetic method name (such as "__abs__").

:Returns:

    out: `cf.{+Variable}
        A new Variable.

:Examples:

>>> print v.array
[1 2 -3 -4 -5]

>>> w = v._unary_operation('__abs__')
>>> print w.array
[1 2 3 4 5]

>>> w = v.__abs__()
>>> print w.array
[1 2 3 4 5]

>>> w = abs(v)
>>> print w.array
[1 2 3 4 5]

        '''
        if not self._hasData:
            raise ValueError(
                "Can't apply {} to a {} with no Data".format(
                    method, self.__class__.__name__))

        new = self.copy(_omit_Data=True)

        new.Data = self.Data._unary_operation(method)
        
        return new
    #--- End: def

    def _YMDhms(self, attr):
        '''
'''
        if self._hasData:
            out = self.copy(_omit_Data=True)
            out.insert_data(getattr(self.data, attr), copy=False)
            try:
                del out.standard_name
            except AttributeError:
                pass
            out.long_name = attr
            return out
        #--- End: if
        raise ValueError(
            "ERROR: Can't get {0}s when there is no data array".format(attr))        
    #--- End: def

    def _hmmm(self, method):
        if self._hasData:
            out = self.copy(_omit_Data=True)
            out.insert_data(getattr(self.data, method)(), copy=False)
            try:
                del out.standard_name
            except AttributeError:
                pass
            out.long_name = method
            return out
        #--- End: if
        raise ValueError(
            "ERROR: Can't get {0} when there is no data array".format(method))        
    #--- End: def

#    def _forbidden(self, x, name):        
#        raise AttributeError(
#            "{} has no {} {!r}.".format(self.__class__.__name__, x, name))
#    #--- End: def
#
#    # ================================================================
#    # Forbidden methods
#    # ================================================================
#    def append(self, *args, **kwargs): self._forbidden('method', 'append')
#    def count(self, *args, **kwargs):  self._forbidden('method', 'count')
#    def extend(self, *args, **kwargs): self._forbidden('method', 'extend')
#    def index(self, *args, **kwargs):  self._forbidden('method', 'index')
#    def insert(self, *args, **kwargs): self._forbidden('method', 'insert')
#    def pop(self, *args, **kwargs):    self._forbidden('method', 'pop')
#    def remove(self, *args, **kwargs): self._forbidden('method', 'remove')

    # ================================================================
    # Attributes
    # ================================================================
    @property
    def Data(self):
        '''

The `cf.Data` object containing the data array.

The use of this attribute does not guarantee that any missing data
value that has been set is passed on to the `cf.Data` object. Use the
`data` attribute to ensure that this is the case.

:Examples:

>>> f.Data
<CF Data: >

'''
        if self._hasData:
            return self._private['Data']

        raise AttributeError("%s doesn't have attribute 'Data'" %
                             self.__class__.__name__)
    #--- End: def
    @Data.setter
    def Data(self, value):
        private = self._private
        private['Data'] = value

        # Delete Units from the variable
        private['special_attributes'].pop('Units', None)
 
        self._hasData = True
    #--- End: def
    @Data.deleter
    def Data(self):
        private = self._private
        data = private.pop('Data', None)

        if data is None:
            raise AttributeError("Can't delete non-existent %s attribute 'Data'" %
                                 self.__class__.__name__)

        # Save the Units to the variable
        private['special_attributes']['Units'] = data.Units
        
        self._hasData = False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute
    # ----------------------------------------------------------------
    @property
    def data(self):
        '''

The `cf.Data` object containing the data array.

.. seealso:: `array`, `cf.Data`, `hasdata`, `varray`

:Examples:

>>> f.hasdata
True
>>> f.data
<CF Data: [[267.3, ..., 234.5]] K>

'''       
        if self._hasData:
            data = self.Data
            data.fill_value = self._fill_value
            return data 

        raise AttributeError("%s object doesn't have attribute 'data'" %
                             self.__class__.__name__)
    #--- End: def
    @data.setter
    def data(self, value):
        self.Data = value
    @data.deleter
    def data(self):
        del self.Data

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def hasbounds(self):
        '''

True if there are cell bounds.

If present, cell bounds are stored in the `!bounds` attribute.

:Examples:

>>> if c.hasbounds:
...     b = c.bounds

'''      
        return self._hasbounds
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def hasdata(self):
        '''

True if there is a data array.

If present, the data array is stored in the `data` attribute.

.. seealso:: `data`, `hasbounds`

:Examples:

>>> if f.hasdata:
...     print f.data

'''      
        return self._hasData
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute
    # ----------------------------------------------------------------
    @property
    def reference_datetime(self):
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
"{0} doesn't have attribute 'reference_datetime'".format(
    self.__class__.__name__))
        return dt(units.reftime, calendar=units._calendar)

    @reference_datetime.setter
    def reference_datetime(self, value):
        units = self.Units
        if not units.isreftime:
            raise AttributeError(
"Can't set 'reference_datetime' for non reference date-time units".format(
    self.__class__.__name__))

        units = units.units.split(' since ')
        try:
            self.units = "{0} since {1}".format(units[0], value)
        except (ValueError, TypeError):
            raise ValueError(
"Can't override reference date-time {0!r} with {1!r}".format(
    units[1], value))
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute
    # ----------------------------------------------------------------
    @property
    def Units(self):
        '''The `cf.Units` object containing the units of the data array.

Stores the units and calendar CF properties in an internally
consistent manner. These are mirrored by the `units` and `calendar` CF
properties respectively.

:Examples:

>>> f.Units
<CF Units: K>

>>> f.Units
<CF Units: days since 2014-1-1 calendar=noleap>

        '''
        if self._hasData:
            return self.Data.Units

        try:
            return self._get_special_attr('Units')
        except AttributeError:
            self._set_special_attr('Units', _units_None)
            return _units_None
    #--- End: def

    @Units.setter
    def Units(self, value):
        if self._hasData:
            self.Data.Units = value
        else:
            self._set_special_attr('Units', value)
    #--- End: def
    @Units.deleter
    def Units(self):
        raise AttributeError(
"Can't delete %s attribute 'Units'. Use the override_units method." %
self.__class__.__name__)

    @property
    def year(self):
        '''

The year of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `month`, `day`, `hour`, `minunte`, second`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.year.array
[450 450 451]

''' 
        return self._YMDhms('year')
    #--- End: def

    @property
    def month(self):
        '''

The month of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `year`, `day`, `hour`, `minunte`, second`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.month.array
[11 12  1]

'''
        return self._YMDhms('month')
    #--- End: def

    @property
    def day(self):
        '''
The day of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `year`, `month`, `hour`, `minute`, second`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.day.array
[15 16 16]

'''
        return self._YMDhms('day')
    #--- End: def

    @property
    def hour(self):
        '''
The hour of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `year`, `month`, `day`, `minute`, second`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.hour.array
[ 0 12 12]

'''
        return self._YMDhms('hour')
    #--- End: def

    @property
    def minute(self):
        '''

The minute of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `year`, `month`, `day`, `hour`, second`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.minute.array
[ 0 30  0]

'''
        return self._YMDhms('minute')
    #--- End: def

    @property
    def second(self):
        '''
The second of each date-time data array element.

Only applicable to data arrays with reference time units.

.. seealso:: `year`, `month`, `day`, `hour`, `minute`

:Examples:

>>> f.dtarray
[ 450-11-15 00:00:00  450-12-16 12:30:00  451-01-16 12:00:45]
>>> f.second.array
[ 0  0 45]
'''
        return self._YMDhms('second')
    #--- End: def

    def mask_invalid(self, i=False):
        '''Mask the array where invalid values occur.

Note that:

* Invalid values are Nan or inf

* Invalid values in the results of arithmetic operations only occur if
  the raising of `FloatingPointError` exceptions has been suppressed
  by `cf.Data.seterr`.

* If the raising of `FloatingPointError` exceptions has been allowed
  then invalid values in the results of arithmetic operations it is
  possible for them to be automatically converted to masked values,
  depending on the setting of `cf.Data.mask_fpe`. In this case, such
  automatic conversion might be faster than calling `mask_invalid`.

.. seealso:: `cf.Data.mask_fpe`, `cf.Data.seterr`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> print f.array
[ 0.  1.]
>>> print g.array
[ 1.  2.]

>>> old = cf.Data.seterr('ignore')
>>> h = g/f
>>> print h.array
[ inf   2.]
>>> h.{+name}(i=True)
>>> print  h.array
[--  2.]

>>> h = g**12345
>>> print h.array
[ 1.  inf]
>>> h = h.{+name}()
>>> print h.array
[1.  --]

>>> old = cf.Data.seterr('raise')
>>> old = cf.Data.mask_fpe(True)
>>> print (g/f).array
[ --  2]
>>> print (g**12345).array
[1.  -- ]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data = v.Data.mask_invalid(i=True)

        return v
    #--- End: def

    def max(self):
        '''The maximum of the data array.

.. seealso:: `collapse`, `mean`, `mid_range`, `min`, `range`,
             `sample_size`, `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`    
        The maximum of the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
311.343780518
>>> f.{+name}().data
<CF Data: 311.343780518 K>
        '''
        if self._hasData:
            return self.data.max(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the maximum when there is no data array")       
    #--- End: def

    def mean(self):
        '''The unweighted mean the data array.

.. seealso:: `collapse`, `max`, `mid_range`, `min`, `range`,
             `sample_size`, `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The unweighted mean the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
280.192227593
>>> f.{+name}().data
<CF Data: 280.192227593 K>

'''
        if self._hasData:
            return self.data.mean(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the mean when there is no data array")       
    #--- End: def

    def mid_range(self):
        '''The unweighted average of the maximum and minimum of
the data array.

.. seealso:: `collapse`, `max`, `mean`, `min`, `range`,
             `sample_size`, `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The unweighted average of the maximum and minimum of the
        data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
255.08618927
>>> f.{+name}().data
<CF Data: 255.08618927 K>

'''
        if self._hasData:
            return self.data.mid_range(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the mid-range when there is no data array")       
    #--- End: def

    def min(self):
        '''The minimum of the data array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `range`,
             `sample_size`, `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The minimum of the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
198.828598022
>>> f.{+name}().data
<CF Data: 198.828598022 K>

'''
        if self._hasData:
            return self.data.min(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the minimum when there is no data array")       
    #--- End: def

    def range(self):
        '''The absolute difference between the maximum and minimum of the data
array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `min`,
             `sample_size`, `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The absolute difference between the maximum and minimum of the
        data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
112.515182495
>>> f.{+name}().data
<CF Data: 112.515182495 K>

        '''
        if self._hasData:
            return self.data.range(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the range when there is no data array")       
    #--- End:

    def remove_data(self):
        '''

Remove and return the data array.

:Returns: 

    out: `cf.Data` or `None`
        The removed data array, or `None` if there isn't one.

:Examples:

>>> f._hasData
True
>>> f.data
<CF Data: [0, ..., 9] m>
>>> f.remove_data()
<CF Data: [0, ..., 9] m>
>>> f._hasData
False
>>> print f.remove_data()
None

'''
        if not self._hasData:
            return

        data = self.data
        del self.Data

        return data
    #--- End: def

    def sample_size(self):
        '''The number of non-missing data elements in the data array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `min`, `range`,
             `sd`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The number of non-missing data elements in the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
98304.0
>>> f.{+name}().data
<CF Data: 98304.0>

'''
        if self._hasData:
            return self.data.sample_size(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the sample size when there is no data array")
    #--- End: def

    def sd(self):
        '''The unweighted sample standard deviation of the data array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `min`, `range`,
             `sample_size`, `sum`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The unweighted standard deviation of the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
22.685052535
>>> f.{+name}().data
<CF Data: 22.685052535 K>

        '''
        if self._hasData:
            return self.data.sd(squeeze=True, ddof=0)
          
        raise ValueError(
            "ERROR: Can't get the standard deviation when there is no data array")
    #--- End: def

    def sum(self):
       	'''The sum of the data array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `min`, `range`,
             `sample_size`, `sd`, `var`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The sum of the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
27544016.7413
>>> f.{+name}().data
<CF Data: 27544016.7413 K>

'''
        if self._hasData:
            return self.data.sum(squeeze=True)
          
        raise ValueError(
            "ERROR: Can't get the sum when there is no data array")       
    #--- End: def

    def var(self):
	'''The unweighted sample variance of the data array.

.. seealso:: `collapse`, `max`, `mean`, `mid_range`, `min`, `range`,
             `sample_size`, `sd`, `sum`

:Examples 1:

>>> d = f.{+name}()

:Returns: 

    out: `cf.Data`
        The unweighted variance of the data array.

:Examples 2:

>>> f.data
<CF Data: [[[236.512756348, ..., 256.93371582]]] K>
>>> f.{+name}()
514.611608515
>>> f.{+name}().data
<CF Data: 514.611608515 K2>

'''
        if self._hasData:
            return self.data.var(squeeze=True, ddof=0)
          
        raise ValueError(
            "ERROR: Can't get the variance when there is no data array")
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def T(self):
        '''

Always False.

.. seealso:: `X`, `Y`, `Z`

:Examples:

>>> print f.T
False

'''              
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def X(self):
        '''

Always False.

.. seealso:: `T`, `Y`, `Z`

:Examples:

>>> print f.X
False

'''              
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def Y(self):
        '''
Always False.

.. seealso:: `T`, `X`, `Z`

:Examples:

>>> print f.Y
False

'''              
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def Z(self):
        '''
Always False.

.. seealso:: `T`, `X`, `Y`

:Examples:

>>> print f.Z
False

'''              
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def add_offset(self):
        '''The add_offset CF property.

If present then this number is *subtracted* from the data prior to it
being written to a file. If both `scale_factor` and `add_offset`
properties are present, the offset is subtracted before the data are
scaled. See http://cfconventions.org/latest.html for details.

:Examples:

>>> f.add_offset = -4.0
>>> f.add_offset
-4.0
>>> del f.add_offset

>>> f.setprop('add_offset', 10.5)
>>> f.getprop('add_offset')
10.5
>>> f.delprop('add_offset')

        '''
        return self.getprop('add_offset')
    #--- End: def
    @add_offset.setter
    def add_offset(self, value):
        self.setprop('add_offset', value)
        self.dtype = numpy_result_type(self.dtype, numpy_array(value).dtype)
    #--- End: def
    @add_offset.deleter
    def add_offset(self):
        self.delprop('add_offset')
        if not self.hasprop('scale_factor'):
            del self.dtype
    #--- End: def

    # ----------------------------------------------------------------
    # CF property: calendar
    # ----------------------------------------------------------------
    @property
    def calendar(self):
        '''The calendar CF property.

The calendar used for encoding time data. See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.calendar = 'noleap'
>>> f.calendar
'noleap'
>>> del f.calendar

>>> f.setprop('calendar', 'proleptic_gregorian')
>>> f.getprop('calendar')
'proleptic_gregorian'
>>> f.delprop('calendar')

        '''
        value = getattr(self.Units, 'calendar', None)
        if value is None:
            raise AttributeError("%s doesn't have CF property 'calendar'" %
                                 self.__class__.__name__)
        return value
    #--- End: def

    @calendar.setter
    def calendar(self, value):
        self.Units = Units(getattr(self, 'units', None), value)
    #--- End: def

    @calendar.deleter
    def calendar(self):
        if getattr(self, 'calendar', None) is None:
            raise AttributeError("Can't delete non-existent %s CF property 'calendar'" %
                                 self.__class__.__name__)
        
        self.Units = Units(getattr(self, 'units', None))
    #--- End: def

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def comment(self):
        '''The comment CF property.

Miscellaneous information about the data or methods used to produce
it. See http://cfconventions.org/latest.html for details.

:Examples:

>>> f.comment = 'This simulation was done on an HP-35 calculator'
>>> f.comment
'This simulation was done on an HP-35 calculator'
>>> del f.comment

>>> f.setprop('comment', 'a comment')
>>> f.getprop('comment')
'a comment'
>>> f.delprop('comment')

        '''
        return self.getprop('comment')
    #--- End: def
    @comment.setter
    def comment(self, value): self.setprop('comment', value)
    @comment.deleter
    def comment(self):        self.delprop('comment')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def _FillValue(self):
        '''The _FillValue CF property.

A value used to represent missing or undefined data.

Note that this property is primarily for writing data to disk and is
independent of the missing data mask. It may, however, get used when
unmasking data array elements. See
http://cfconventions.org/latest.html for details.

The recommended way of retrieving the missing data value is with the
`fill_value` method.

.. seealso:: `fill_value`, `missing_value`

:Examples:

>>> f._FillValue = -1.0e30
>>> f._FillValue
-1e+30
>>> del f._FillValue

Mask the data array where it equals a missing data value:

>>> f.setitem(cf.masked, condition=f.fill_value()) DCH

        '''
        d = self._private['simple_properties']
        if '_FillValue' in d:
            return d['_FillValue']

        raise AttributeError("%s doesn't have CF property '_FillValue'" %
                             self.__class__.__name__)
    #--- End: def

    @_FillValue.setter
    def _FillValue(self, value):
#        self.setprop('_FillValue', value) 
        self._private['simple_properties']['_FillValue'] = value
        self._fill_value = self.getprop('missing_value', value)
    #--- End: def

    @_FillValue.deleter
    def _FillValue(self):
        self._private['simple_properties'].pop('_FillValue', None)
        self._fill_value = getattr(self, 'missing_value', None)
    #--- End: def

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def history(self):
        '''The history CF property.

A list of the applications that have modified the original data. See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.history = 'created on 2012/10/01'
>>> f.history
'created on 2012/10/01'
>>> del f.history

>>> f.setprop('history', 'created on 2012/10/01')
>>> f.getprop('history')
'created on 2012/10/01'
>>> f.delprop('history')

        '''
        return self.getprop('history')
    #--- End: def

    @history.setter
    def history(self, value): self.setprop('history', value)
    @history.deleter
    def history(self):        self.delprop('history')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def leap_month(self):
        '''The leap_month CF property.

Specifies which month is lengthened by a day in leap years for a user
defined calendar. See http://cfconventions.org/latest.html for
details.

:Examples:

>>> f.leap_month = 2
>>> f.leap_month
2
>>> del f.leap_month

>>> f.setprop('leap_month', 11)
>>> f.getprop('leap_month')
11
>>> f.delprop('leap_month')

        '''
        return self.getprop('leap_month')
    #--- End: def
    @leap_month.setter
    def leap_month(self, value): self.setprop('leap_month', value)
    @leap_month.deleter
    def leap_month(self):        self.delprop('leap_month')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def leap_year(self):
        '''The leap_year CF property.

Provides an example of a leap year for a user defined calendar. It is
assumed that all years that differ from this year by a multiple of
four are also leap years. See http://cfconventions.org/latest.html for
details.

:Examples:

>>> f.leap_year = 1984
>>> f.leap_year
1984
>>> del f.leap_year

>>> f.setprop('leap_year', 1984)
>>> f.getprop('leap_year')
1984
>>> f.delprop('leap_year')

        '''
        return self.getprop('leap_year')
    #--- End: def
    @leap_year.setter
    def leap_year(self, value): self.setprop('leap_year', value)
    @leap_year.deleter
    def leap_year(self):        self.delprop('leap_year')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def long_name(self):
        '''The long_name CF property.

A descriptive name that indicates a nature of the data. This name is
not standardized. See http://cfconventions.org/latest.html for
details.

:Examples:

>>> f.long_name = 'zonal_wind'
>>> f.long_name
'zonal_wind'
>>> del f.long_name

>>> f.setprop('long_name', 'surface air temperature')
>>> f.getprop('long_name')
'surface air temperature'
>>> f.delprop('long_name')

        '''
        return self.getprop('long_name')
    #--- End: def
    @long_name.setter
    def long_name(self, value): self.setprop('long_name', value)
    @long_name.deleter
    def long_name(self):        self.delprop('long_name')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def missing_value(self):
        '''The missing_value CF property.

A value used to represent missing or undefined data (deprecated by the
netCDF user guide). See http://cfconventions.org/latest.html for
details.

Note that this attribute is used primarily for writing data to disk
and is independent of the missing data mask. It may, however, be used
when unmasking data array elements.

The recommended way of retrieving the missing data value is with the
`fill_value` method.

.. seealso:: `_FillValue`, `fill_value`

:Examples:

>>> f.missing_value = 1.0e30
>>> f.missing_value
1e+30
>>> del f.missing_value

Mask the data array where it equals a missing data value:

>>> f.setitem(cf.masked, condition=f.fill_value()) DCH

        '''        
        d = self._private['simple_properties']
        if 'missing_value' in d:
            return d['missing_value']

        raise AttributeError("%s doesn't have CF property 'missing_value'" %
                             self.__class__.__name__)
     #--- End: def
    @missing_value.setter
    def missing_value(self, value):
        self._private['simple_properties']['missing_value'] = value
        self._fill_value = value
    #--- End: def
    @missing_value.deleter
    def missing_value(self):
        self._private['simple_properties'].pop('missing_value', None)
        self._fill_value = getattr(self, '_FillValue', None)
    #--- End: def

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def month_lengths(self):
        '''The month_lengths CF property.

Specifies the length of each month in a non-leap year for a user
defined calendar. See http://cfconventions.org/latest.html for
details.

Stored as a tuple but may be set as any array-like object.

:Examples:

>>> f.month_lengths = numpy.array([34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34])
>>> f.month_lengths
(34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34)
>>> del f.month_lengths

>>> f.setprop('month_lengths', [34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34])
>>> f.getprop('month_lengths')
(34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34)
>>> f.delprop('month_lengths')

        '''
        return self.getprop('month_lengths')
    #--- End: def

    @month_lengths.setter
    def month_lengths(self, value): self.setprop('month_lengths', tuple(value))
    @month_lengths.deleter
    def month_lengths(self):        self.delprop('month_lengths')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def scale_factor(self):
        '''The scale_factor CF property.

If present then the data are *divided* by this factor prior to it
being written to a file. If both `scale_factor` and `add_offset`
properties are present, the offset is subtracted before the data are
scaled. See http://cfconventions.org/latest.html for details.

:Examples:

>>> f.scale_factor = 10.0
>>> f.scale_factor
10.0
>>> del f.scale_factor

>>> f.setprop('scale_factor', 10.0)
>>> f.getprop('scale_factor')
10.0
>>> f.delprop('scale_factor')

        '''
        return self.getprop('scale_factor')
    #--- End: def
    @scale_factor.setter
    def scale_factor(self, value): self.setprop('scale_factor', value)
    @scale_factor.deleter
    def scale_factor(self):        self.delprop('scale_factor')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def standard_name(self):
        '''The standard_name CF property.

A standard name that references a description of a data in the
standard name table
(http://cfconventions.org/standard-names.html). See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.standard_name = 'time'
>>> f.standard_name
'time'
>>> del f.standard_name

>>> f.setprop('standard_name', 'time')
>>> f.getprop('standard_name')
'time'
>>> f.delprop('standard_name')

        '''
        return self.getprop('standard_name')
    #--- End: def
    @standard_name.setter
    def standard_name(self, value): self.setprop('standard_name', value)
    @standard_name.deleter
    def standard_name(self):        self.delprop('standard_name')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def units(self):
        '''The units CF property.

The units of the data. The value of the `units` property is a string
that can be recognized by UNIDATA's Udunits package
(http://www.unidata.ucar.edu/software/udunits). See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.units = 'K'
>>> f.units
'K'
>>> del f.units

>>> f.setprop('units', 'm.s-1')
>>> f.getprop('units')
'm.s-1'
>>> f.delprop('units')

        '''
        value = getattr(self.Units, 'units', None)
        if value is None:
            raise AttributeError("%s doesn't have CF property 'units'" %
                                 self.__class__.__name__)
        return value
    #--- End: def

    @units.setter
    def units(self, value):
        self.Units = Units(value, getattr(self, 'calendar', None))
    #--- End: def
    @units.deleter
    def units(self):
        if getattr(self, 'units', None) is None:
            self.Units = Units(None, getattr(self, 'calendar', None))
    #--- End: def

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def valid_max(self):
        '''The valid_max CF property.

The largest valid value of the data. See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.valid_max = 100.0
>>> f.valid_max
100.0
>>> del f.valid_max

>>> f.setprop('valid_max', 100.0)
>>> f.getprop('valid_max')
100.0
>>> f.delprop('valid_max')

        '''
        return self.getprop('valid_max')
    #--- End: def
    @valid_max.setter
    def valid_max(self, value): self.setprop('valid_max', value)
    @valid_max.deleter
    def valid_max(self):        self.delprop('valid_max')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def valid_min(self):
        '''The valid_min CF property.	

The smallest valid value of the data. See
http://cfconventions.org/latest.html for details.

:Examples:

>>> f.valid_min = 8.0
>>> f.valid_min
8.0
>>> del f.valid_min

>>> f.setprop('valid_min', 8.0)
>>> f.getprop('valid_min')
8.0
>>> f.delprop('valid_min')

        '''
        return self.getprop('valid_min')
    #--- End: def
    @valid_min.setter
    def valid_min(self, value): self.setprop('valid_min', value)
    @valid_min.deleter
    def valid_min(self):        self.delprop('valid_min')

    # ----------------------------------------------------------------
    # CF property
    # ----------------------------------------------------------------
    @property
    def valid_range(self):
        '''The valid_range CF property.

The smallest and largest valid values the data. See
http://cfconventions.org/latest.html for details.

Stored as a tuple but may be set as any array-like object.

:Examples:

>>> f.valid_range = numpy.array([100., 400.])
>>> f.valid_range
(100.0, 400.0)
>>> del f.valid_range

>>> f.setprop('valid_range', [100.0, 400.0])
>>> f.getprop('valid_range')
(100.0, 400.0)
>>> f.delprop('valid_range')

        '''
        return tuple(self.getprop('valid_range'))
    #--- End: def
    @valid_range.setter
    def valid_range(self, value): self.setprop('valid_range', tuple(value))
    @valid_range.deleter
    def valid_range(self):        self.delprop('valid_range')

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def subspace(self):
        '''

Return a new variable whose data is subspaced.

This attribute may be indexed to select a subspace from dimension
index values.

**Subspacing by indexing**

Subspacing by dimension indices uses an extended Python slicing
syntax, which is similar numpy array indexing. There are two
extensions to the numpy indexing functionality:

* Size 1 dimensions are never removed.

  An integer index i takes the i-th element but does not reduce the
  rank of the output array by one.

* When advanced indexing is used on more than one dimension, the
  advanced indices work independently.

  When more than one dimension's slice is a 1-d boolean array or 1-d
  sequence of integers, then these indices work independently along
  each dimension (similar to the way vector subscripts work in
  Fortran), rather than by their elements.

:Examples:

'''
        return SubspaceVariable(self)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def shape(self):
        '''

A tuple of the data array's dimension sizes.

.. seealso:: `data`, `hasdata`, `ndim`, `size`

:Examples:

>>> f.shape
(73, 96)
>>> f.ndim
2

>>> f.ndim
0
>>> f.shape
()

>>> f.hasdata
True
>>> len(f.shape) == f.dnim
True
>>> reduce(lambda x, y: x*y, f.shape, 1) == f.size
True

'''
        if self._hasData:
            return tuple(self.Data.shape)

        raise AttributeError("%s doesn't have attribute 'shape'" %
                             self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def ndim(self):
        '''

The number of dimensions in the data array.

.. seealso:: `data`, `hasdata`, `isscalar`, `shape`

:Examples:

>>> f.hasdata
True
>>> f.shape
(73, 96)
>>> f.ndim
2

>>> f.shape
()
>>> f.ndim
0

'''
        if self._hasData:
            return self.Data.ndim

        raise AttributeError("%s doesn't have attribute 'ndim'" %
                             self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def size(self):
        '''
The number of elements in the data array.

.. seealso:: `data`, `hasdata`, `ndim`, `shape`

:Examples:

>>> f.shape
(73, 96)
>>> f.size
7008

>>> f.shape
()
>>> f.ndim
0
>>> f.size
1

>>> f.shape
(1, 1, 1)
>>> f.ndim
3
>>> f.size
1

>>> f.hasdata
True
>>> f.size == reduce(lambda x, y: x*y, f.shape, 1)
True

'''
        if self._hasData:
            return self.Data.size
        
        raise AttributeError("%s doesn't have attribute 'size'" %
                             self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def dtarray(self):
        '''

An independent numpy array of date-time objects.

Only applicable for reference time units.

If the calendar has not been set then the CF default calendar will be
used and the units will be updated accordingly.

The data type of the data array is unchanged.

.. seealso:: `array`, `asdatetime`, `asreftime`, `dtvarray`, `varray`

:Examples:

'''
        if self._hasData:
            return self.data.dtarray

        raise AttributeError("%s has no data array" % self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute
    # ----------------------------------------------------------------
    @property
    def dtype(self):
        '''

The `numpy` data type of the data array.

By default this is the data type with the smallest size and smallest
scalar kind to which all sub-arrays of the master data array may be
safely cast without loss of information. For example, if the
sub-arrays have data types 'int64' and 'float32' then the master data
array's data type will be 'float64'; or if the sub-arrays have data
types 'int64' and 'int32' then the master data array's data type will
be 'int64'.

Setting the data type to a `numpy.dtype` object, or any object
convertible to a `numpy.dtype` object, will cause the master data
array elements to be recast to the specified type at the time that
they are next accessed, and not before. This does not immediately
change the master data array elements, so, for example, reinstating
the original data type prior to data access results in no loss of
information.

Deleting the data type forces the default behaviour. Note that if the
data type of any sub-arrays has changed after `dtype` has been set
(which could occur if the data array is accessed) then the reinstated
default data type may be different to the data type prior to `dtype`
being set.

:Examples:

>>> f.dtype
dtype('float64')
>>> type(f.dtype)
<type 'numpy.dtype'>

>>> print f.array
[0.5 1.5 2.5]
>>> import numpy
>>> f.dtype = numpy.dtype(int)
>>> print f.array
[0 1 2]
>>> f.dtype = bool
>>> print f.array
[False  True  True]
>>> f.dtype = 'float64'
>>> print f.array
[ 0.  1.  1.]

>>> print f.array
[0.5 1.5 2.5]
>>> f.dtype = int
>>> f.dtype = bool
>>> f.dtype = float
>>> print f.array
[ 0.5  1.5  2.5]

'''
        if self._hasData:
            return self.Data.dtype

        raise AttributeError("%s doesn't have attribute 'dtype'" %
                             self.__class__.__name__)
    #--- End: def
    @dtype.setter
    def dtype(self, value):
# DCH - allow dtype to be set before data c.f.  Units
        if self._hasData:
            self.Data.dtype = value
    #--- End: def
    @dtype.deleter
    def dtype(self):
        if self._hasData:
            del self.Data.dtype
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def dtvarray(self):
        '''

A numpy array view the data array converted to date-time objects.

Only applicable for reference time units.

If the calendar has not been set then the CF default calendar will be
used and the units will be updated accordingly.

.. seealso:: `array`, `asdatetime`, `asreftime`, `dtarray`, `varray`

:Examples:

'''
        if self._hasData:
            return self.data.dtvarray

        raise AttributeError(
            "{} has no data array".foarmat(self.__class__.__name__))
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read/write only)
    # ----------------------------------------------------------------
    @property
    def hardmask(self):
        '''

Whether the mask is hard (True) or soft (False).

When the mask is hard, masked elements of the data array can not be
unmasked by assignment, but unmasked elements may be still be masked.

When the mask is soft, masked entries of the data array may be
unmasked by assignment and unmasked entries may be masked.

By default, the mask is hard.

.. seealso:: `where`, `subspace`, `__setitem__`

:Examples:

>>> f.hardmask = False
>>> f.hardmask
False

'''
        if self._hasData:
            return self.Data.hardmask

        raise AttributeError(
"{} doesn't have attribute 'hardmask'".format(self.__class__.__name__))
    #--- End: def
    
    @hardmask.setter
    def hardmask(self, value):
        if self._hasData:
            self.Data.hardmask = value
        else:
            raise AttributeError(
"{} doesn't have attribute 'hardmask'".format(self.__class__.__name__))
    #--- End: def
    @hardmask.deleter
    def hardmask(self):
        raise AttributeError(
"Won't delete {} attribute 'hardmask'".format(self.__class__.__name__))
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def array(self):
        '''A numpy array deep copy of the data array.

Changing the returned numpy array does not change the data array.

.. seealso:: `data`, `dtarray`, `dtvarray`, `varray`

:Examples 1:

>>> f.data
<CF Data: [0, ... 4] kg m-1 s-2>
>>> a = f.array
>>> type(a)
<type 'numpy.ndarray'>
>>> print a
[0 1 2 3 4]
>>> a[0] = 999
>>> print a
[999 1 2 3 4]
>>> print f.array
[0 1 2 3 4]
>>> f.data
<CF Data: [0, ... 4] kg m-1 s-2>

        '''
        if self._hasData:
            return self.data.array

        raise AttributeError(
            "{} has no data array".format(self.__class__.__name__))
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def unsafe_array(self):
        '''

'''
        if self._hasData:
            return self.data.unsafe_array

        raise AttributeError("%s has no data array" % self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def varray(self):
        '''

A numpy array view of the data array.

Changing the elements of the returned view changes the data array.

.. seealso:: `array`, `data`, `dtarray`, `dtvarray`

:Examples 1:

>>> f.data
<CF Data: [0, ... 4] kg m-1 s-2>
>>> a = f.array
>>> type(a)
<type 'numpy.ndarray'>
>>> print a
[0 1 2 3 4]
>>> a[0] = 999
>>> print a
[999 1 2 3 4]
>>> print f.array
[999 1 2 3 4]
>>> f.data
<CF Data: [999, ... 4] kg m-1 s-2>

'''
        if self._hasData:
            return self.data.varray

        raise AttributeError("%s has no data array" % self.__class__.__name__)
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def isauxiliary(self): 
        '''True if the variable is an auxiliary coordinate object.

.. seealso:: `isdimension`, `isdomainancillary`, `isfieldancillary`,
             `ismeasure`

:Examples: 

>>> f.isauxiliary
False

        '''
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def isdimension(self): 
        '''True if the variable is a dimension coordinate object.

.. seealso:: `isauxiliary`, `isdomainancillary`, `isfieldancillary`,
             `ismeasure`

:Examples: 

>>> f.isdimension
False

        '''
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def isdomainancillary(self): 
        '''True if the variable is a domain ancillary object.

.. versionadded:: DCH 

.. seealso:: `isauxiliary`, `isdimension`, `isfieldancillary`,
             `ismeasure`

:Examples: 

>>> f.isdomainancillary
False

        '''
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def isfieldancillary(self): 
        '''True if the variable is a field ancillary object.

.. versionadded:: DCH 

.. seealso:: `isauxiliary`, `isdimension`, `isdomainancillary`,
             `ismeasure`

:Examples: 

>>> f.isfieldancillary
False

        '''
        return False
    #--- End: def

    # ----------------------------------------------------------------
    # Attribute (read only)
    # ----------------------------------------------------------------
    @property
    def ismeasure(self): 
        '''True if the variable is a cell measure object.

.. seealso:: `isauxiliary`, `isdimension`, `isdomainancillary`,
             `isfieldancillary`

:Examples: 

>>> f.ismeasure
False

        '''
        return False
    #--- End: def

    @property
    def isscalar(self):
        '''True if the data array is scalar.

.. seealso:: `hasdata`, `ndim`

:Examples:

>>> f.ndim
0
>>> f.isscalar
True

>>> f.ndim >= 1
True
>>> f.isscalar
False

>>> f.hasdata
False
>>> f.isscalar
False

        '''
        if not self._hasData:
            return False

        return self.Data.isscalar
    #--- End: def

    def ceil(self, i=False):
        '''The ceiling of the data array.

The ceiling of the scalar ``x`` is the smallest integer ``i``, such
that ``i >= x``.

.. versionadded:: 1.0

.. seealso:: `floor`, `rint`, `trunc`

:Examples 1:

Create a new {+variable} with the ceiling of the data:

>>> g = f.{+name}()

:Parameters:

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the ceiling of data array values.

:Examples 2:

>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}().array
[-1. -1. -1. -1.  0.  1.  2.  2.  2.]
>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}(i=True).array
[-1. -1. -1. -1.  0.  1.  2.  2.  2.]
>>> print f.array
[-1. -1. -1. -1.  0.  1.  2.  2.  2.]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.ceil(i=True)

        return v
    #--- End: def

    def chunk(self, chunksize=None):
        '''Partition the data array.

:Parameters:

    chunksize: `int`

:Returns:

    `None`

        '''
        if self._hasData:
            self.Data.chunk(chunksize)

        # Partition the data of the bounds, if they exist.
        if self._hasbounds:
            self.bounds.chunk(chunksize)
    #--- End: def

    def clip(self, a_min, a_max, units=None, bounds=True, i=False):
        '''Limit the values in the data array.

Given an interval, values outside the interval are clipped to the
interval edges.

:Examples 1:

>>> g = f.clip(-90, 90)

:Parameters:
 
    a_min: scalar

    a_max: scalar

    units: `str` or `cf.Units`

    {+bounds}

    {+i}

:Returns: 

    out: `cf.{+Variable}`

:Examples 2:

>>> 
        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.clip(a_min, a_max, units=units, i=True)

        if bounds and v._hasbounds:
            v.bounds.clip(a_min, a_max, units=units, i=True)

        return v
    #--- End: def

    def close(self):
        '''Close all files referenced by the {+variable}.

Note that a closed file will be automatically reopened if its contents
are subsequently required.

.. seealso:: `files`

:Examples 1:

>>> v.{+name}()

:Returns:

    `None`
        
        '''
        if self._hasData:
            self.Data.close()
            
        if self._hasbounds:
            self.bounds.close()
    #--- End: def

    @classmethod
    def concatenate(cls, variables, axis=0, _preserve=True):
        '''Join a sequence of variables together.

:Parameters:

    variables: sequence of `cf.{+Variable}`

    axis: `int`, optional

    _preserve: `bool`, optional

:Returns:

    out: `cf.{+Variable}`

        '''
        variable0 = variables[0]

        if len(variables) == 1:
            return variable0.copy()

        out = variable0.copy(_omit_Data=True)
        out.Data = Data.concatenate([v.Data for v in variables], axis=axis,
                                    _preserve=_preserve)

        if variable0._hasbounds:
            bounds = Variable.concatenate(
                [v.bounds for v in variables],
                axis=axis, _preserve=_preserve)
            out.insert_bounds(bounds, copy=False)

        return out
    #--- End: def

    def copy(self, _omit_Data=False, _only_Data=False,
             _omit_special=None, _omit_properties=False,
             _omit_attributes=False):
        '''Return a deep copy.

``f.{+name}()`` is equivalent to ``copy.deepcopy(f)``.

:Examples 1:

>>> g = f.{+name}()

:Returns:

    out: `cf.{+Variable}`
        The deep copy.

:Examples 2:

>>> g = f.{+name}()
>>> g is f
False
>>> f.equals(g)
True
>>> import copy
>>> h = copy.deepcopy(f)
>>> h is f
False
>>> f.equals(g)
True

        '''
        new = type(self)()
#        ts = type(self)
#        new = ts.__new__(ts)

        if _only_Data:
            if self._hasData:
                new.Data = self.Data.copy()

            return new
        #--- End: if

        self_dict = self.__dict__.copy()
        
        self_private = self_dict.pop('_private')
            
        del self_dict['_hasData']
        new.__dict__['_fill_value'] = self_dict.pop('_fill_value')
        new.__dict__['_hasbounds']  = self_dict.pop('_hasbounds')
            
        if self_dict and not _omit_attributes:        
            try:
                new.__dict__.update(loads(dumps(self_dict, -1)))
            except PicklingError:
                new.__dict__.update(deepcopy(self_dict))
                
        private = {}

        if not _omit_Data and self._hasData:
            private['Data'] = self_private['Data'].copy()
            new._hasData = True
 
        # ------------------------------------------------------------
        # Copy special attributes. These attributes are special
        # because they have a copy() method which return a deep copy.
        # ------------------------------------------------------------
        special = self_private['special_attributes'].copy()
        if _omit_special:            
            for prop in _omit_special:
                special.pop(prop, None)

        for prop, value in special.iteritems():
            special[prop] = value.copy()

        private['special_attributes'] = special

        if not _omit_properties:
            try:
                private['simple_properties'] = loads(dumps(self_private['simple_properties'], -1))
            except PicklingError:
                private['simple_properties'] = deepcopy(self_private['simple_properties'])
        else:
            private['simple_properties'] = {}

        new._private = private

        if self._hasbounds:
            bounds = self.bounds.copy(_omit_Data=_omit_Data,
                                      _only_Data=_only_Data)
            new._set_special_attr('bounds', bounds)        

        return new
    #--- End: def

    def cos(self, bounds=True, i=False):
        '''Take the trigonometric cosine of the data array.

Units are accounted for in the calculation, so that the the cosine of
90 degrees_east is 0.0, as is the cosine of 1.57079632 radians. If the
units are not equivalent to radians (such as Kelvin) then they are
treated as if they were radians.

The output units are '1' (nondimensionsal).

.. seealso:: `sin`, `tan`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the cosine of data array values.

:Examples 2:

>>> f.Units
<CF Units: degrees_east>
>>> print f.array
[[-90 0 90 --]]
>>> f.{+name}()
>>> f.Units
<CF Units: 1>
>>> print f.array
[[0.0 1.0 0.0 --]]

>>> f.Units
<CF Units: m s-1>
>>> print f.array
[[1 2 3 --]]
>>> f.{+name}()
>>> f.Units
<CF Units: 1>
>>> print f.array
[[0.540302305868 -0.416146836547 -0.9899924966 --]]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.cos(i=True)

        if bounds and v._hasbounds:
            v.bounds.cos(i=True)

        return v
    #--- End: def

    def count(self):
        '''Count the non-masked elements of the array.

:Examples 1:

>>> n = f.count()

:Returns:

    out: `int`
     
'''
        if not self._hasData:
            return 0
            
        return self.Data.count()
    #--- End: def

    def cyclic(self, axes=None, iscyclic=True):
        '''Set the cyclicity of axes of the data array.

.. seealso:: `iscyclic`

:Parameters:

    axes: (sequence of) `int`
        The axes to be set. Each axis is identified by its integer
        position. By default no axes are set.
        
    iscyclic: `bool`, optional
        If False then the axis is set to be non-cyclic. By default the
        axis is set to be cyclic.

:Returns:

    out: `!set`

:Examples:

>>> f.{+name}()
{}
>>> f.{+name}(1)
{}
>>> f.{+name}()
{1}

        '''
        if self._hasData:
            out = self.Data.cyclic(axes, iscyclic)

            if axes is not None and self._hasbounds:
                axes = self._parse_axes(axes)
                self.bounds.cyclic(axes, iscyclic)

            return out
        else:
            return {}
    #--- End: def
            
    def datum(self, *index):
        '''

Return an element of the data array as a standard Python scalar.

The first and last elements are always returned with ``f.datum(0)``
and ``f.datum(-1)`` respectively, even if the data array is a scalar
array or has two or more dimensions.

:Parameters:

    index: optional
        Specify which element to return. When no positional arguments
        are provided, the method only works for data arrays with one
        element (but any number of dimensions), and the single element
        is returned. If positional arguments are given then they must
        be one of the following:

          * An integer. This argument is interpreted as a flat index
            into the array, specifying which element to copy and
            return.
         
              Example: If the data aray shape is ``(2, 3, 6)`` then:
                * ``f.{+name}(0)`` is equivalent to ``f.{+name}(0, 0, 0)``.
                * ``f.{+name}(-1)`` is equivalent to ``f.{+name}(1, 2, 5)``.
                * ``f.{+name}(16)`` is equivalent to ``f.{+name}(0, 2, 4)``.

            If *index* is ``0`` or ``-1`` then the first or last data
            array element respecitively will be returned, even if the
            data array is a scalar array or has two or more
            dimensions.
        ..
         
          * Two or more integers. These arguments are interpreted as a
            multidimensionsal index to the array. There must be the
            same number of integers as data array dimensions.
        ..
         
          * A tuple of integers. This argument is interpreted as a
            multidimensionsal index to the array. There must be the
            same number of integers as data array dimensions.
         
              Example: ``f.datum((0, 2, 4))`` is equivalent to
              ``f.datum(0, 2, 4)``; and ``f.datum(())`` is equivalent
              to ``f.datum()``.

:Returns:

    out:
        A copy of the specified element of the array as a suitable
        Python scalar.

:Examples:

>>> print f.array
2
>>> f.{+name}()
2
>>> 2 == f.{+name}(0) == f.{+name}(-1) == f.{+name}(())
True

>>> print f.array
[[2]]
>>> 2 == f.{+name}() == f.{+name}(0) == f.{+name}(-1)
True
>>> 2 == f.{+name}(0, 0) == f.{+name}((-1, -1)) == f.{+name}(-1, 0)
True

>>> print f.array
[[4 -- 6]
 [1 2 3]]
>>> f.{+name}(0)
4     
>>> f.{+name}(-1)
3     
>>> f.{+name}(1)
masked
>>> f.{+name}(4)
2     
>>> f.{+name}(-2)
2     
>>> f.{+name}(0, 0)
4     
>>> f.{+name}(-2, -1)
6     
>>> f.{+name}(1, 2)
3     
>>> f.{+name}((0, 2))
6

'''
        if not self._hasData:
            raise ValueError(
                "ERROR: Can't return an element when there is no data array")
        
        return self.Data.datum(*index)
    #--- End: def

    def dump(self, display=True, prefix=None, omit=(), field=None,
             key=None, _title=None, _level=0):
        '''

Return a string containing a full description of the instance.

:Parameters:

    display: `bool`, optional
        If False then return the description as a string. By default
        the description is printed, i.e. ``f.dump()`` is equivalent to
        ``print f.dump(display=False)``.

    omit: sequence of `str`, optional
        Omit the given CF properties from the description.

    prefix: optional
        Ignored.

:Returns:

    out: `None` or `str`
        A string containing the description.

:Examples:

>>> f.{+name}()
Data(1, 2) = [[2999-12-01 00:00:00, 3000-12-01 00:00:00]] 360_day
axis = 'T'
standard_name = 'time'

>>> f.{+name}(omit=('axis',))
Data(1, 2) = [[2999-12-01 00:00:00, 3000-12-01 00:00:00]] 360_day
standard_name = 'time'

'''
        indent0 = '    ' * _level
        indent1 = '    ' * (_level+1)

        if _title is None:
            string = ['{0}Variable: {1}'.format(indent0, self.name(''))]
        else:
            string = [indent0 + _title]

        string.append(self._dump_simple_properties(omit=omit, _level=_level+1))

        if self._hasData:
            if field and key:
                x = ['{0}({1})'.format(field.axis_name(axis), field.axis_size(axis))
                     for axis in field.item_axes(key)]
            else:
                x = [str(s) for s in self.shape]

            string.append('{0}Data({1}) = {2}'.format(indent1,
                                                      ', '.join(x),
                                                      str(self.Data)))
#            data = self.Data
#            string.append('{0}Data{1} = {2}'.format(indent1,
#                                                    data.shape,
#                                                    data))

        string = '\n'.join(string)
       
        if display:
            print string
        else:
            return string
    #--- End: def

    def equals(self, other, rtol=None, atol=None, ignore_fill_value=False,
               traceback=False, ignore=()): #, _set=False):
        '''

True if two {+variable}s are equal, False otherwise.

:Parameters:

    other: 
        The object to compare for equality.

    {+atol}

    {+rtol}

    ignore_fill_value: `bool`, optional
        If True then data arrays with different fill values are
        considered equal. By default they are considered unequal.

    traceback: `bool`, optional
        If True then print a traceback highlighting where the two
        {+variable}s differ.

    ignore: `tuple`, optional
        The names of CF properties to omit from the comparison.

:Returns: 

    out: `bool`
        Whether or not the two {+variable}s are equal.

:Examples:

>>> f.equals(f)
True
>>> g = f + 1
>>> f.equals(g)
False
>>> g -= 1
>>> f.equals(g)
True
>>> f.setprop('name', 'name0')
>>> g.setprop('name', 'name1')
>>> f.equals(g)
False
>>> f.equals(g, ignore=['name'])
True

'''
        # Check for object identity
        if self is other:
            return True

        # Check that each instance is of the same type
        if not isinstance(other, self.__class__):
            if traceback:
                print("{0}: Incompatible types: {0}, {1}".format(
			self.__class__.__name__,
			other.__class__.__name__))
	    return False
        #--- End: if

#        # Check that there are equal numbers of variables
#        len_self = len(self)
#        if len_self != len(other): 
#            if traceback:
#                print("{}: Different numbers of ppp: {}, {}".format(
#			self.__class__.__name__,
#			len_self,
#			len(other)))
#            return False
#        #--- End: if
#
#        if len_self != 1:
#	    if not _set:
#		# ----------------------------------------------------
#		# Two lists, each with more than one element: Check
#		# the lists pair-wise.
#		# ----------------------------------------------------
#		for i, (f, g) in enumerate(zip(self, other)):
#		    if not f.equals(g, rtol=rtol, atol=atol,
#				    ignore_fill_value=ignore_fill_value,
#				    traceback=traceback, ignore=ignore):
#			if traceback:
#			    print("{0}: Different element {1}: {2!r}, {3!r}".format(
#				    self.__class__.__name__, i, f, g))
#                        return False
#	    else:
#		# ----------------------------------------------------
#		# Two lists, each with more than one element: Check
#		# the lists set-wise.
#		# ----------------------------------------------------
#
#		# Group the variables by identity
#		self_identity  = {}
#                for f in self:
#                    self_identity.setdefault(f.identity(), []).append(f)
#
#		other_identity  = {}
#                for f in other:
#                    other_identity.setdefault(f.identity(), []).append(f)
#
#		# Check that there are the same identities
#		if set(self_identity) != set(other_identity):
#		    if traceback:
#			print("{0}: Different sets of identities: {1}, {2}".format(
#				self.__class__.__name__,
#				set(self_identity),
#				set(other_identity)))
#		    return False
#                #--- End: if
#
#                # Check that there are the same number of variables
#		# for each identity
#		for identity, fl in self_identity.iteritems():
#		    gl = other_identity[identity]
#		    if len(fl) != len(gl):
#			if traceback:
#			    print("{0}: Different numbers of {1!r} {2}s: {3}, {4}".format(
#				    self.__class__.__name__,
#				    identity,
#                                    fl[0].__class__.__name__,
#				    len(fl),
#                                    len(gl)))
#                        return False
#                #--- End: fo
#
#		# For each identity, check that there are matching
#		# pairs of equal fields.
#		for identity, fl in self_identity.iteritems():
#		    gl = other_identity[identity]
#
#                    for f in fl:
#			found_match = False
#                        for i, g in enumerate(gl):
#                            if f.equals(g, rtol=rtol, atol=atol,
#                                        ignore_fill_value=ignore_fill_value,
#					ignore=ignore, traceback=False):
#                                found_match = True
#				del gl[i]
#                                break
#                        #--- End: for
#			if not found_match:
#			    if traceback:                        
#				print("{0}: No {1} equal to: {2!r}".format(
#					self.__class__.__name__,
#					g.__class__.__name__,
#					f))
#                            return False
#	    #--- End: if
#
#	    # --------------------------------------------------------
#	    # Lists are equal
#	    # --------------------------------------------------------
#	    return True	    
#        #--- End: if
#
#	# Still here?
#	if self._list:
#	    self = self[0]
#	if other._list:
#	    other = other[0]

        # ------------------------------------------------------------
        # Check the simple properties
        # ------------------------------------------------------------
        if ignore_fill_value:
            ignore += ('_FillValue', 'missing_value')

        self_simple  = self._private['simple_properties']
        other_simple = other._private['simple_properties']

        if (set(self_simple).difference(ignore) != 
            set(other_simple).difference(ignore)):
            if traceback:
                print("{}: Different properties: {}, {}".format( 
                    self.__class__.__name__, self_simple, other_simple))
            return False
        #--- End: if

        if rtol is None:
            rtol = RTOL()
        if atol is None:
            atol = ATOL()

        for attr, x in self_simple.iteritems():
            if attr in ignore:
                continue
            y = other_simple[attr]
            if not cf_equals(x, y, rtol=rtol, atol=atol,
                             ignore_fill_value=ignore_fill_value,
                             traceback=traceback):
                if traceback:
                    print("{}: Different {}: {!r}, {!r}".format(
                        self.__class__.__name__, attr, x, y))
                return False
        #--- End: for

        # ------------------------------------------------------------
        # Check the special attributes
        # ------------------------------------------------------------
        self_special  = self._private['special_attributes']
        other_special = other._private['special_attributes']
        if set(self_special) != set(other_special):
            if traceback:
                print("{}: Different attributes: {}".format(
                    self.__class__.__name__,
                    set(self_special).symmetric_difference(other_special)))
            return False
        #--- End: if

        for attr, x in self_special.iteritems():
            y = other_special[attr]
            result = cf_equals(x, y, rtol=rtol, atol=atol,
                               ignore_fill_value=ignore_fill_value,
                               traceback=traceback)
               
            if not result:
                if traceback:
                    print("{}: Different {}: {!r}, {!r}".format(
                          self.__class__.__name__, attr, x, y))
                return False
        #--- End: for

        # ------------------------------------------------------------
        # Check the data
        # ------------------------------------------------------------
        self_hasData = self._hasData
        if self_hasData != other._hasData:
            if traceback:
                print("{}: Different data".format(self.__class__.__name__))
            return False

        if self_hasData:
            if not self.data.equals(other.data, rtol=rtol, atol=atol,
                                    ignore_fill_value=ignore_fill_value,
                                    traceback=traceback):
                if traceback:
                    print("{}: Different data".format(self.__class__.__name__))
                return False
        #--- End: if

        return True
    #--- End: def

    def equivalent(self, other, rtol=None, atol=None, traceback=False):
        '''True if two {+variable}s are equal, False otherwise.

:Parameters:

    other: 
        The object to compare for equality.

    {+atol}

    {+rtol}

        '''     
        if self is other:
            return True

        # Check that each instance is the same type
        if type(self) != type(other):
            print("{}: Different types: {}, {}".format(
                self.__class__.__name__,
                self.__class__.__name__,
                other.__class__.__name__))
            return False
        #--- End: if
       
        identity0 = self.identity()
        identity1 = other.identity()

        if identity0 is None or identity1 is None or identity0 != identity1:
            # add traceback
            return False
                  
        # ------------------------------------------------------------
        # Check the special attributes
        # ------------------------------------------------------------
        self_special  = self._private['special_attributes']
        other_special = other._private['special_attributes']
        if set(self_special) != set(other_special):
            if traceback:
                print("%s: Different attributes: %s" %
                      (self.__class__.__name__,
                       set(self_special).symmetric_difference(other_special)))
            return False
        #--- End: if

        for attr, x in self_special.iteritems():
            y = other_special[attr]

            result = cf_equivalent(x, y, rtol=rtol, atol=atol,
                                   traceback=traceback)
               
            if not result:
                if traceback:
                    print("{}: Different {} attributes: {!r}, {!r}".format(
                        self.__class__.__name__, attr, x, y))
                return False
        #--- End: for

        # ------------------------------------------------------------
        # Check the data
        # ------------------------------------------------------------
        if not self._equivalent_data(other, rtol=rtol, atol=atol,
                                     traceback=traceback):
            # add traceback
            return False
            
        return True
    #--- End: def

    def convert_reference_time(self, units=None,
                               calendar_months=False,
                               calendar_years=False, i=False):
        '''Convert reference time data values to have new units.

Conversion is done by decoding the reference times to date-time
objects and then re-encoding them for the new units.

Any conversions are possible, but this method is primarily for
conversions which require a change in the date-times originally
encoded. For example, use this method to reinterpret data values in
units of "months" since a reference time to data values in "calendar
months" since a reference time. This is often necessary when when
units of "calendar months" were intended but encoded as "months",
which have special definition. See the note and examples below for
more details.

For conversions which do not require a change in the date-times
implied by the data values, this method will be considerably slower
than a simple reassignment of the units. For example, if the original
units are ``'days since 2000-12-1'`` then ``c.Units = cf.Units('days
since 1901-1-1')`` will give the same result and be considerably
faster than ``c.convert_reference_time(cf.Units('days since
1901-1-1'))``

.. note::
   It is recommended that the units "year" and "month" be used
   with caution, as explained in the following excerpt from the CF
   conventions: "The Udunits package defines a year to be exactly
   365.242198781 days (the interval between 2 successive passages of
   the sun through vernal equinox). It is not a calendar year. Udunits
   includes the following definitions for years: a common_year is 365
   days, a leap_year is 366 days, a Julian_year is 365.25 days, and a
   Gregorian_year is 365.2425 days. For similar reasons the unit
   ``month``, which is defined to be exactly year/12, should also be
   used with caution.

:Examples 1:

>>> g = f.convert_reference_time()
    
:Parameters:

    units: `cf.Units`, optional
        The reference time units to convert to. By default the units
        days since the original reference time in the the original
        calendar.

          *Example:*
            If the original units are ``'months since 2000-1-1'`` in
            the Gregorian calendar then the default units to convert
            to are ``'days since 2000-1-1'`` in the Gregorian
            calendar.

    calendar_months: `bool`, optional
        If True then treat units of ``'months'`` as if they were
        calendar months (in whichever calendar is originally
        specified), rather than a 12th of the interval between 2
        successive passages of the sun through vernal equinox
        (i.e. 365.242198781/12 days).

    calendar_years: `bool`, optional
        If True then treat units of ``'years'`` as if they were
        calendar years (in whichever calendar is originally
        specified), rather than the interval between 2 successive
        passages of the sun through vernal equinox (i.e. 365.242198781
        days).
        
    {+i}

:Returns: 
 
    out: `cf.{+Variable}` 
        The {+variable} with converted reference time data values.

:Examples 2:

>>> print f.array
[1 2 3 4]
>>> print f.Units
months since 2000-1-1
>>> print f.dtarray
[datetime.datetime(2000, 1, 31, 10, 29, 3, 831197)
 datetime.datetime(2000, 3, 1, 20, 58, 7, 662441)
 datetime.datetime(2000, 4, 1, 7, 27, 11, 493645)
 datetime.datetime(2000, 5, 1, 17, 56, 15, 324889)]
>>> f.convert_reference_time(calendar_months=True, i=True)
>>> print f.dtarray
[datetime.datetime(2000, 2, 1, 0, 0)
 datetime.datetime(2000, 3, 1, 0, 0)
 datetime.datetime(2000, 4, 1, 0, 0)
 datetime.datetime(2000, 5, 1, 0, 0)]
>>> print f.array
[  31.   60.   91.  121.]
>>> print f.Units
days since 2000-1-1

        '''
        def _convert_reftime_units(value, units, reftime, calendar):
            '''sads

    :Parameters:

        value: number

        units: `cf.Units`

    :Returns:

        out: `datetime.datetime` or `cf.Datetime`

            '''
            t = TimeDuration(value, units=units)
            if value > 0:
                return t.interval(reftime, calendar=calendar, end=False)[1]
            else:
                return t.interval(reftime, calendar=calendar, end=True)[0]
        #--- End: def

#        # List functionality
#        if self._list:
#            kwargs2 = self._parameters(locals())
#            return self._list_method('convert_reference_time', kwargs2)

        if not self.Units.isreftime:
            raise ValueError(
"{} must have reference time units, not {!r}".format(
    self.__class__.__name__, self.Units))

        if i:
            v = self
        else:
            v = self.copy()

        units0 = self.Units
        
        if units is None:
            # By default, set the target units to "days since
            # <reference time of self.Units>,
            # calendar=<self.calendar>"
            units = Units('days since '+units0.units.split(' since ')[1],
                          calendar=units0._calendar)
        elif not getattr(units, 'isreftime', False):
            raise ValueError(
"New units must be reference time units, not {0!r}".format(units))
           
        if units0._utime.units in _month_units:
            if calendar_months:
                units0 = Units('calendar_'+units0.units, calendar=units0._calendar)
            else:
                units0 = Units('days since '+units0.units.split(' since ')[1],
                                calendar=units0._calendar)
                v.Units = units0
        elif units0._utime.units in _year_units:
            if calendar_years:
                units0 = Units('calendar_'+units0.units, calendar=units0._calendar)
            else:
                units0 = Units('days since '+units0.units.split(' since ')[1],
                                calendar=units0._calendar)
                v.Units = units0
        #--- End: def

        # Not LAMAed!
        v.Data = Data(
            numpy_vectorize(
                functools_partial(_convert_reftime_units,
                                  units=units0._utime.units,
                                  reftime=units0.reftime, #.timetuple()[:6],
                                  calendar=units0._calendar),
                otypes=[object])(v),
            units=units)

        return v    
    #--- End: def

    def floor(self, bounds=True, i=False):
        '''Floor the data array.

The floor of the scalar ``x`` is the largest integer ``i``, such that
``i <= x``.

.. versionadded:: 1.0

.. seealso:: `ceil`, `rint`, `trunc`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the floor of data array values.

:Examples 2:

>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}().array
[-2. -2. -2. -1.  0.  1.  1.  1.  1.]
>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}(i=True).array
[-2. -2. -2. -1.  0.  1.  1.  1.  1.]
>>> print f.array
[-2. -2. -2. -1.  0.  1.  1.  1.  1.]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.floor(i=True)

        if bounds and v._hasbounds:
            v.bounds.floor(i=True)

        return v
    #--- End: def

    def match(self, description=None, ndim=None, exact=False,
              match_and=True, inverse=False, _Flags=False,
              _CellMethods=False):
        '''Determine whether or not a variable satisfies conditions.

Conditions may be specified on the variable's attributes and CF
properties.

:Parameters:

:Returns:

    out: `bool`
        Whether or not the variable matches the given criteria.

:Examples:

        '''
        conditions_have_been_set = False
        something_has_matched    = False

        if ndim is not None:
            conditions_have_been_set = True
            try:
                found_match = self.ndim == ndim
            except AttributeError:
                found_match = False

            if match_and and not found_match:
                return bool(inverse)

            something_has_matched = True
        #--- End: if
            
        matches = self._parse_match(description)

        if matches:
            conditions_have_been_set = True

        found_match = True
        for match in matches:
            found_match = True

            # ----------------------------------------------------
            # Try to match cf.Units
            # ----------------------------------------------------
            if 'units' in match or 'calendar' in match:
                match = match.copy()
                units = Units(match.pop('units', None), match.pop('calendar', None))
                
                if not exact:
                    found_match = self.Units.equivalent(units)
                else:
                    found_match = self.Units.equals(units)
    
                if not found_match:
                    continue
            #--- End: if

            # ----------------------------------------------------
            # Try to match cell methods
            # ----------------------------------------------------
            if _CellMethods and 'cell_methods' in match:
                f_cell_methods = self.getprop('cell_methods', None)
                
                if not f_cell_methods:
                    found_match = False
                    continue

                match = match.copy()
                cell_methods = match.pop('cell_methods')

                if not exact:
                    n = len(cell_methods)
                    if n > len(f_cell_methods):
                        found_match = False
                    else:
                        found_match = f_cell_methods[-n:].equivalent(cell_methods)
                else:
                    found_match = f_cell_methods.equals(cell_methods)
                                    
                if not found_match:
                    continue
            #--- End: if

            # ---------------------------------------------------
            # Try to match cf.Flags
            # ---------------------------------------------------
            if _Flags and ('flag_masks'    in match or 
                           'flag_meanings' in match or
                           'flag_values'   in match):
                f_flags = getattr(self, Flags, None)
                
                if not f_flags:
                    found_match = False
                    continue

                match = match.copy()
                found_match = f_flags.equals(
                    Flags(flag_masks=match.pop('flag_masks', None),
                          flag_meanings=match.pop('flag_meanings', None),
                          flag_values=match.pop('flag_values', None)))
            
                if not found_match:
                    continue
            #--- End: if
             
            for prop, value in match.iteritems():
                if prop is None: 
                    if value is None:
                        continue

                    if isinstance(value, basestring):
                        if value in ('T', 'X', 'Y', 'Z'):
                            # Axis type
                            x = getattr(self, value)
                            value = True
                        else:
                            value = value.split('%')
                            if len(value) == 1:
                                value = value[0].split(':')
                                if len(value) == 1:
                                    # Identity
                                    # (string-valued). E.g. 'air_temperature'
                                    x = self.identity(None)
                                    value = value[0]
                                else:
                                    # CF property
                                    # (string-valued). E.g. 'long_name:rain'
                                    x = self.getprop(value[0], None)
                                    value = ':'.join(value[1:])
                            else:
                                # Python attribute
                                # (string-valued). E.g. 'ncvar%tas'
                                x = getattr(self, value[0], None)
                                value = '%'.join(value[1:])
                    else:   
                        # Identity (not string-valued, e.g. cf.Query)
                        x = self.identity(None)
                else:                    
                    # CF property
                    x = self.getprop(prop, None)
    
                if x is None:
                    found_match = False
                elif not exact and isinstance(x, basestring) and isinstance(value, basestring):
#                    if exact:
#                        found_match = (value == x) #re_match('^'+value+'$', x)
#                    else:
                    found_match = re_match(value, x)
                else:	
                    found_match = (value == x)
                    try:
                        found_match == True
                    except ValueError:
                        found_match = False
                #--- End: if
     
                if not found_match:
                    break
            #--- End: for

            if found_match:
                something_has_matched = True
                break
        #--- End: for

        if match_and and not found_match:
            return bool(inverse)

        if conditions_have_been_set:
            if something_has_matched:            
                return not bool(inverse)
            else:
                return bool(inverse)
        else:
            return not bool(inverse)
    #--- End: def

    @property
    def mask(self):
        '''The mask of the data array.

Values of True indicate masked elements.

.. seealso:: `binary_mask`

:Examples:

>>> f.shape
(12, 73, 96)
>>> m = f.mask
>>> m.long_name
'mask'
>>> m.shape
(12, 73, 96)
>>> m.dtype
dtype('bool')
>>> m.data
<CF Data: [[[True, ..., False]]] >

        '''
        if not self._hasData:
            raise ValueError(
                "ERROR: Can't get mask when there is no data array")

        out = self.copy(_omit_Data=True, _omit_properties=True)

        out.Data = self.data.mask

        out.long_name = 'mask'

        out.ncvar = None
        del out.ncvar

        return out
    #--- End: def

#    # ----------------------------------------------------------------
#    # Attribute (read only)
#    # ----------------------------------------------------------------
#    @property
#    def attributes(self):
#        '''
#
#A dictionary of the attributes which are not CF properties.
#
#:Examples:
#
#>>> f.attributes
#{}
#>>> f.foo = 'bar'
#>>> f.attributes
#{'foo': 'bar'}
#>>> f.attributes.pop('foo')
#'bar'
#>>> f.attributes
#{'foo': 'bar'}
#
#'''
#        attributes = self.__dict__.copy()
#
#        del attributes['_hasbounds']
#        del attributes['_hasData']
#        del attributes['_private']
#
#        return attributes
#    #--- End: def
#
#    # ----------------------------------------------------------------
#    # Attribute (read only)
#    # ----------------------------------------------------------------
#    @property
#    def properties(self):
#        '''
#
#A dictionary of the CF properties.
#
#:Examples:
#
#>>> f.properties
#{'_FillValue': 1e+20,
# 'foo': 'bar',
# 'long_name': 'Surface Air Temperature',
# 'standard_name': 'air_temperature',
# 'units': 'K'}
#
#'''
#        return self._simple_properties().copy()
#    #--- End: def

    # ================================================================
    # Methods
    # ================================================================
    def all(self):
        '''Test whether all data array elements evaluate to True.

Performs a logical and over the data array and returns the
result. Masked values are considered as True during computation.

.. seealso:: `any`

:Examples 1:
        
>>> x = f.{+name}()

:Returns:

    out: `bool`
        Whether ot not all data array elements evaluate to True.

:Examples:

>>> print f.array
[[0 3 0]]
>>> f.{+name}()
False

>>> print f.array
[[1 3 --]]
>>> f.{+name}()
True

        '''
        if self._hasData:
            return self.Data.all()

        return False
    #--- End: def

    def allclose(self, y, atol=None, rtol=None):
        '''Returns True if two broadcastable {+variable}s have equal array
values to within numerical tolerance.

For numeric data arrays ``f.allclose(y, atol, rtol)`` is equivalent to
``(abs(f - y) <= atol + rtol*abs(y)).all()``; for other data types
it is equivalent to ``(f == y).all()``.

.. seealso:: `~cf.{+Variable}.equals`

:Examples 1:
        
>>> x = f.{+name}(g)

:Parameters:

    y: data-like object
        The object to be compared with the data array. *y* must be
        broadcastable to the data array and if *y* has units then they
        must be compatible.

        {+data-like}
    
    {+atol}

    {+rtol}

:Returns:

    out: `bool`
        Whether or not the two data arrays are equivalent.

:Examples 2:

        '''
        if not self._hasData:
            return False

        if isinstance(y, self.__class__):
            if not y._hasData:
                return False

            y = self._conform_for_assignment(y)
        #-- End: if

        y = getattr(y, 'Data', y)
            
        return self.Data.allclose(y, rtol=rtol, atol=atol)
    #--- End: def

    def any(self):
        '''Return True if any data array elements evaluate to True.

Performs a logical or over the data array and returns the
result. Masked values are considered as False during computation.

.. seealso:: `all`

:Examples 1:

>>> x = f.any()

:Returns:

    out: `bool`
        Whether ot not any data array elements evaluate to True.

:Examples 2:

>>> print f.array
[[0 0 0]]
>>> f.{+name}()
False

>>> print f.array
[[-- 0 0]]
>>> d.{+name}()
False

>>> print f.array
[[-- 3 0]]
>>> f.{+name}()
True

        '''
        if self._hasData:
            return self.Data.any()

        return False
    #--- End: def

    def asdatetime(self, i=False):
        '''Convert the internal representation of data array elements to
date-time objects.

Only applicable to {+variable}s with reference time units.

If the calendar has not been set then the CF default calendar will be
used and the units will be updated accordingly.

.. seealso:: `asreftime`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> t.{+name}().dtype
dtype('float64')
>>> t.{+name}().dtype
dtype('O')

        '''
        raise NotImplementedError("asdatetime is dead. Consider {0}.dtarray instead".format(self.__class__.__name__))
#        # List functionality
#        if self._list:
#            kwargs2 = self._parameters(locals())
#            return self._list_method('asdatetime', kwargs2)
#
#        if not self._hasData:
#            raise AttributeError(
#                "{0} has no data array".format(self.__class__.__name__))
#
#        if i:
#            v = self
#        else:
#            v = self.copy()
#
#        v.data.asdatetime(i=True)
#        return v
    #--- End: def

    def asreftime(self, i=False):
        '''Convert the internal representation of data array elements
to numeric reference times.

Only applicable to {+variable}s with reference time units.

If the calendar has not been set then the CF default calendar will be
used and the units will be updated accordingly.

.. seealso:: `asdatetime`

:Examples 1:

>>> g = f.asreftime()

:Parameters:

    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> t.asdatetime().dtype
dtype('O')
>>> t.asreftime().dtype
dtype('float64')

        '''
        raise NotImplementedError("asreftime is dead. Consider {0}.array instead".format(self.__class__.__name__))   
#        # List functionality
#        if self._list:
#            kwargs2 = self._parameters(locals())
#            return self._list_method('asreftime', kwargs2)
#
#        if not self._hasData:
#            raise AttributeError(
#                "{0} has no data array".format(self.__class__.__name__))
#
#        if i:
#            v = self
#        else:
#            v = self.copy()
#
#        v.data.asreftime(i=True)
#        return v
    #--- End: def

    def files(self):
        '''Return the names of any files containing parts of the data array.

.. seealso:: `close`

:Examples 1:

>>> f.{+name}()

:Returns:

    out: `!set`
        The file names in normalized, absolute form.

:Examples 2:

>>> f = cf.read_field('../file*')
>>> f.{+name}()
{'/data/user/file1',
 '/data/user/file2',
 '/data/user/file3'}
>>> a = f.array
>>> f.{+name}()
set()

        '''
        if self._hasData:
           out = self.Data.files()
        else:
           out = set()

        if self._hasbounds:
            out.update(self.bounds.files())

        return out
    #--- End: def

    def fill_value(self, default=None):
        '''Return the data array missing data value.

This is the value of the `missing_value` CF property, or if that is
not set, the value of the `_FillValue` CF property, else if that is
not set, ``None``. In the last case the default `numpy` missing data
value for the array's data type is assumed if a missing data value is
required.

:Parameters:

    default: optional
        If the missing value is unset then return this value. By
        default, *default* is `None`. If *default* is the special
        value ``'netCDF'`` then return the netCDF default value
        appropriate to the data array's data type is used. These may
        be found as follows:

        >>> import netCDF4
        >>> print netCDF4.default_fillvals    

:Returns:

    out:
        The missing data value, or the value specified by *default* if
        one has not been set.

:Examples:

>>> f.{+name}()
None
>>> f._FillValue = -1e30
>>> f.{+name}()
-1e30
>>> f.missing_value = 1073741824
>>> f.{+name}()
1073741824
>>> del f.missing_value
>>> f.{+name}()
-1e30
>>> del f._FillValue
>>> f.{+name}()
None
>>> f,dtype
dtype('float64')
>>> f.{+name}(default='netCDF')
9.969209968386869e+36
>>> f._FillValue = -999
>>> f.{+name}(default='netCDF')
-999

        '''
        fillval = self._fill_value

        if fillval is None:
            if default == 'netCDF':
                d = self.dtype
                fillval = _netCDF4_default_fillvals[d.kind + str(d.itemsize)]
            else:
                fillval = default 
        #--- End: if

        return fillval
#        return self._fill_value
    #--- End: def

    def flip(self, axes=None, i=False):
        '''Flip dimensions of the data array in place.

.. seealso:: `expand_dims`, `squeeze`, `transpose`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    axes: (sequence of) `int`
        Flip the dimensions whose positions are given. By default all
        dimensions are flipped.

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> f.{+name}()
>>> f.{+name}(1)
>>> f.{+name}([0, 1])

>>> g = f[::-1, :, ::-1]
>>> f.{+name}([2, 0]).equals(g)
True

        '''
        kwargs2 = self._parameters(locals())

        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.flip(axes, i=True)
        
        return v
    #--- End: def

    def select(self, *args, **kwargs):
        '''`cf.{+Variable}.select` has been deprecated.

Use `cf.{+Variable}.match` to see if an individual {+variable} meets
given criteria.

        '''
        raise DeprecationError(
"select has been deprecated. Use match to see if criteria are met")
    #--- End: def

    @property
    def binary_mask(self):
        '''

A binary (0 and 1) missing data mask of the data array.

The binary mask's data array comprises dimensionless 32-bit integers
and has 0 where the data array has missing data and 1 otherwise.

:Examples:

>>> print f.mask.array
[[ True  False  True False]]
>>> b = f.binary_mask()
>>> print b.array
[[0 1 0 1]]

'''
        return type(self)(properties={'long_name': 'binary_mask'},
                          data=self.Data.binary_mask(),
                          copy=False)
    #--- End: def

    def exp(self, bounds=True, i=False):
        '''The exponential of the data array.

.. seealso:: `log`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the exponential of data array values.

:Examples 2:

>>> f.data
<CF Data: [[1, 2]]>
>>> f.{+name}().data            
<CF Data: [[2.71828182846, 7.38905609893]]>

>>> f.data
<CF Data: [[1, 2]] 2>
>>> f.{+name}().data            
<CF Data: [[7.38905609893, 54.5981500331]]>

>>> g = f.{+name}(i=True)
>>> g is f
True

>>> f.data
<CF Data: [[1, 2]] kg m-1 s-2>
>>> f.+name}()          
ValueError: Can't take exponential of dimensional quantities: <CF Units: kg m-1 s-2>

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.exp(i=True)

        if bounds and v._hasbounds:
            v.bounds.exp(i=True)

        return v
    #--- End: def

    def expand_dims(self, position=0, i=False):
        '''Insert a size 1 axis into the data array.

.. seealso:: `flip`, `squeeze`, `transpose`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    position: `int`, optional    
        Specify the position amongst the data array axes where the new
        axis is to be inserted. By default the new axis is inserted at
        position 0, the slowest varying position.

    {+bounds}

    {+i}

:Returns:

    `None`

:Examples:

>>> v.{+name}(2)
>>> v.{+name}(-1)

        '''       
        if i:
            v = self
        else:
            v = self.copy()

        if self._hasData:
            v.Data.expand_dims(position, i=True)
        
        if v._hasbounds:
            position = self._parse_axes([position])[0]
            v.bounds.expand_dims(position, i=True)

        return v
    #--- End: def

    def sin(self, bounds=True, i=False):
        '''The trigonometric sine of the data array.

Units are accounted for in the calculation. For example, the the sine
of 90 degrees_east is 1.0, as is the sine of 1.57079632 radians. If
the units are not equivalent to radians (such as Kelvin) then they are
treated as if they were radians.

The Units are changed to '1' (nondimensionsal).

.. seealso:: `cos`, `tan`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the sine of data array values.

:Examples 2:

>>> f.Units
<CF Units: degrees_north>
>>> print f.array
[[-90 0 90 --]]
>>> f.{+name}()
>>> f.Units
<CF Units: 1>
>>> print f.array
[[-1.0 0.0 1.0 --]]

>>> f.Units
<CF Units: m s-1>
>>> print f.array
[[1 2 3 --]]
>>> f.{+name}()
>>> f.Units
<CF Units: 1>
>>> print f.array
[[0.841470984808 0.909297426826 0.14112000806 --]]

'''
        if i:
            v = self
        else:
            v = self.copy()
    
        if v._hasData:
            v.Data.sin(i=True)

        if bounds and v._hasbounds:
            v.bounds.sin(i=True)

        return v
    #--- End: def

    def tan(self, bounds=True, i=False):
        '''The trigonometric tangent of the data array.

Units are accounted for in the calculation, so that the the tangent of
180 degrees_east is 0.0, as is the sine of 3.141592653589793
radians. If the units are not equivalent to radians (such as Kelvin)
then they are treated as if they were radians.

The Units are changed to '1' (nondimensionsal).

.. seealso:: `cos`, `tan`

:Examples 1:	

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the tangent of data array values.

:Examples 2:

>>> 
        '''     
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.tan(i=True)

        if bounds and v._hasbounds:
            v.bounds.tan(i=True)

        return v
    #--- End: def

    def log(self, base=None, bounds=True, i=False):
        '''The logarithm of the data array.

By default the natural logarithm is taken, but any base may be
specified.

.. seealso:: `exp`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    base: number, optional
        The base of the logiarthm. By default a natural logiarithm is
        taken.

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with the logarithm of data array values.

:Examples 2:

>>> f.data
<CF Data: [[1, 2]]>
>>> f.{+name}().data
<CF Data: [[0.0, 0.69314718056]] ln(re 1)>

>>> f.data
<CF Data: [[1, 2]] 2>
>>> f.{+name}().data
<CF Data: [[0.0, 0.69314718056]] ln(re 2 1)>

>>> f.data
<CF Data: [[1, 2]] kg s-1 m-2>
>>> f.{+name}().data
<CF Data: [[0.0, 0.69314718056]] ln(re 1 m-2.kg.s-1)>

>>> g = f.{+name}(i=True)
>>> g is f
True

>>> f.Units
<CF Units: >
>>> f.log()
ValueError: Can't take the logarithm to the base 2.718281828459045 of <CF Units: >

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.log(base, i=True)

        if bounds and v._hasbounds:
            v.bounds.log(i=True)

        return v
    #--- End: def

    def squeeze(self, axes=None, i=False):
        '''Remove size 1 dimensions from the data array

.. seealso:: `expand_dims`, `flip`, `transpose`

:Examples 1:

>>> f.{+name}()

:Parameters:

    axes: (sequence of) `int`, optional
        The size 1 axes to remove. By default, all size 1 axes are
        removed. Size 1 axes for removal are identified by their
        integer positions in the data array.
    
    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples:

>>> f.{+name}(1)
>>> f.{+name}([1, 2])

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.squeeze(axes, i=True)

        if v._hasbounds:
            axes = self._parse_axes(axes)
            v.bounds.squeeze(axes, i=True)

        return v
    #--- End: def
    
    def transpose(self, axes=None, i=False):
        '''Permute the dimensions of the data array.

.. seealso:: `expand_dims`, `flip`, `squeeze`

:Examples 1:

g = f.{+name}()

:Parameters:

    axes: (sequence of) `int`
        The new axis order of the data array. By default the order is
        reversed. Each axis of the new order is identified by its
        original integer position.
        
    {+i}
        
:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> f.shape
(2, 3, 4)
>>> f.{+name}()
>>> f.shape
(4, 3, 2)
>>> f.{+name}([1, 2, 0])
>>> f.shape
(3, 2, 4)
>>> f.{+name}((1, 0, 2))
>>> f.shape
(2, 3, 4)

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.transpose(axes, i=True)
        
        return v
    #--- End: def

    def trunc(self, bounds=True, i=False):
        '''Truncate the data array.

The truncated value of the scalar ``x``, is the nearest integer ``i``
which is closer to zero than ``x`` is. I.e. the fractional part of the
signed number ``x`` is discarded.

.. versionadded:: 1.0

.. seealso:: `ceil`, `floor`, `rint`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with truncated data array values.

:Examples 2:

>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}().array
[-1. -1. -1. -1.  0.  1.  1.  1.  1.]
>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}(i=True).array
[-1. -1. -1. -1.  0.  1.  1.  1.  1.]
>>> print f.array
[-1. -1. -1. -1.  0.  1.  1.  1.  1.]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.trunc(i=True)

        if bounds and v._hasbounds:
            v.bounds.trunc(i=True)

        return v
    #--- End: def

    def unique(self):
        '''The unique elements of the data array.

:Examples 1:

>>> f.{+name}()

:Returns:

    out: `cf.Data`
        Returns the unique data array values in a one dimensional
        `cf.Data` object.

:Examples 2:

>>> print f.array
[[4 2 1]
 [1 2 3]]
>>> print f.{+name}().array
[1 2 3 4]
>>> f[1, -1] = cf.masked
>>> print f.array
[[4 2 1]
 [1 2 --]]
>>> print f.{+name}().array
[1 2 4]

        '''        
        if self._hasData:
            return self.data.unique()

        raise ValueError(
            "ERROR: Can't get unique values when there is no data array")
    #--- End: def

    def setprop(self, prop, value):
        '''

Set a CF property.

.. seealso:: `delprop`, `getprop`, `hasprop`

:Examples 1:

>>> f.setprop('standard_name', 'time')
>>> f.setprop('foo', 12.5)

:Parameters:

    prop: `str`
        The name of the CF property.

    value:
        The value for the property.

:Returns:

     `None`

'''
#        # List functionality
#        if self._list:
#            for f in self:
#                f.setprop(prop, value)
#            return

        # Set a special attribute
        if prop in self._special_properties:
            try:
                setattr(self, prop, value)
            except AttributeError as error:
                raise AttributeError("{} {!r}".format(error, prop))

            return

        # Still here? Then set a simple property
        self._private['simple_properties'][prop] = value
    #--- End: def

    def hasprop(self, prop):
        '''

Return True if a CF property exists, otherise False.

.. seealso:: `delprop`, `getprop`, `setprop`

:Examples 1:

>>> x = f.{+name}('standard_name')

:Parameters:

    prop: `str`
        The name of the property.

:Returns:

     out: `bool`
         True if the CF property exists, otherwise False.

'''
        # Has a special property? # DCH 
        if prop in self._special_properties:
            return hasattr(self, prop)

        # Still here? Then has a simple property?
        return prop in self._private['simple_properties']
    #--- End: def

    def identity(self, default=None, relaxed_identity=None):
        '''Return the identity of the {+variable}.

The identity is, by default, the first found of the following:

* The `standard_name` CF property.

* The `!id` attribute.

* If the *relaxed* parameter is True, the `standard_name` CF property.

* The `!id` attribute.

* The value of the *default* parameter.

This is altered if the *relaxed* parameter is True.

.. seealso:: `name`

:Examples 1:

>>> i = f.{+name}()

:Parameters:

    default: optional
        The identity if one could not otherwise be found. By default,
        *default* is `None`.
        
:Returns:

    out:
        The identity.

:Examples 2:

>>> f.standard_name = 'Kelvin'
>>> f.id = 'foo'
>>> f.{+name}()
'Kelvin'
>>> del f.standard_name
>>> f.{+name}()
'foo'
>>> del f.id
>>> f.{+name}()
None
>>> f.{+name}('bar')
'bar'
>>> print f.{+name}()
None

        '''
        return self.name(default, identity=True, relaxed_identity=relaxed_identity)
    #--- End: def

    def index(self, x, start=0, stop=None):
        '''
L.index(value, [start, [stop]]) -- return first index of value.

Each field in the {+variable} is compared with the field's
`~cf.Field.equals` method (as aopposed to the ``==`` operator).

It is an error if there is no such field.

.. seealso:: :py:obj:`list.index`

:Examples:

>>> 

'''      
        try:
            if stop is None:
                (None,).index(x, start)
            else:
                (None,).index(x, start, stop)
        except ValueError:
            raise ValueError("{0} is not equal to {1!r}".format(
                    self.__class__.__name__, x))

        if not self.equals(x):
            raise ValueError("{0} is not equal to {1!r}".format(
                    self.__class__.__name__, x))

        return 0
    #--- End: def

    def insert_data(self, data, copy=True):
        '''Insert a new data array into the variable in place.

:Parameters:

    data: `cf.Data`

    copy: `bool`, optional

:Returns:

    `None`

        '''
        if not copy:
            self.Data = data
        else:
            self.Data = data.copy()
    #--- End: def

    def inspect(self):
        '''Inspect the object for debugging.

.. seealso:: `cf.inspect`

:Examples 1:

>>> f.{+name}()

:Returns: 

    `None`

        '''
        print cf_inspect(self)
    #--- End: def

#    def getattr(self, attr, *default):
#         '''
#
#Get a named attribute.
#
#``f.getattr(attr, *default)`` is equivalent to ``getattr(f, attr,
#*default)``.
#
#.. seealso:: `delattr`, `hasattr`, `setattr`
#
#:Parameters:
#
#    attr: `str`
#        The attribute's name.
#
#    default: optional
#        When a default argument is given, it is returned when the
#        attribute doesn't exist; without it, an exception is raised in
#        that case.
#
#:Returns:
#
#    out: 
#        The attribute's value.
#
#:Examples:
#
#>>> f.foo = -99
#>>> fl.getattr('foo')
#-99
#>>> del f.foo
#>>> fl.getattr('foo', 'bar')
#'bar'
#
#'''         
#         return getattr(self, attr, *default)
#    #--- End: def
#
#    def hasattr(self, attr):
#         '''
#
#Return whether an attribute exists.
#
#``f.hasattr(attr)`` is equivalent to ``hasattr(f, attr)``.
#
#.. seealso:: `delattr`, `getattr`,  `setattr`
#
#:Parameters:
#
#    attr: `str`
#        The attribute's name.
#
#:Returns:
#
#    out: `bool`
#        Whether the object has the attribute.
#
#:Examples:
#
#>>> f.foo = -99
#>>> fl.hasattr('foo')
#True
#>>> del f.foo
#>>> fl.hasattr('foo')
#False
#
#'''
#         return hasattr(self, attr)
#    #--- End: def

    def getprop(self, prop, *default):
        '''

Get a CF property.

When a default argument is given, it is returned when the attribute
doesn't exist; without it, an exception is raised in that case.

.. seealso:: `delprop`, `hasprop`, `setprop`

:Examples 1:

>>> f.{+name}('standard_name')

:Parameters:

    prop: `str`
        The name of the CF property.

    default: optional
        Return *default* if and only if the variable does not have the
        named property.

:Returns:

    out:
        The value of the named property or the default value, if set.

:Examples 2:

>>> f.{+name}('standard_name')
>>> f.{+name}('standard_name', None)
>>> f.{+name}('foo')
AttributeError: Field doesn't have CF property 'foo'
>>> f.{+name}('foo', 'bar')
'bar'

'''        
        # Get a special attribute
        if prop in self._special_properties:
            return getattr(self, prop, *default)

        # Still here? Then get a simple attribute
        d = self._private['simple_properties']
        if default:
            return d.get(prop, default[0])
        elif prop in d:
            return d[prop]

        raise AttributeError("%s doesn't have CF property %r" %
                             (self.__class__.__name__, prop))
    #--- End: def

#    def delattr(self, attr):
#         '''
#
#Delete an attribute.
#
#[+1]Note that ``f.delattr(attr)`` is equivalent to ``delattr(f, attr)``.
#
#.. seealso:: `getattr`, `hasattr`, `setattr`
#
#:Examples 1:
#
#>>> f.delattr('foo')
#
#:Parameters:
# 
#    attr: `str`
#        The attribute's name.
#
#:Returns:
#
#    `None`
#
#:Examples 2:
#
#>>> f.getattr('foo')
#'bar'
#
#'''
#         # List functionality
#         if self._list:
#             for f in self:
#                 f.delattr(attr)
#             return
#                 
#         delattr(self, attr)
#    #--- End: def

    def delprop(self, prop):
        '''Delete a CF property.

.. seealso:: `getprop`, `hasprop`, `setprop`

:Examples 1:

>>> f.{+name}('standard_name')

:Parameters:

    prop: `str`
        The name of the CF property.

:Returns:

     `None`

:Examples 2:

>>> f.foo = 'bar'
>>> f.{+name}('foo')
>>> f.{+name}('foo')
AttributeError: Can't delete non-existent Field CF property 'foo'

        '''
#        # List functionality
#        if self._list:
#            for f in self:
#                f.delprop(prop)
#            return
                 
        # Delete a special attribute
        if prop in self._special_properties:
            delattr(self, prop)
            return

        # Still here? Then delete a simple attribute
        d = self._private['simple_properties']
        if prop in d:
            del d[prop]
        else:
            raise AttributeError("Can't delete non-existent %s CF property %r" %
                                 (self.__class__.__name__, prop))
    #--- End: def

    def name(self, default=None, identity=False, ncvar=False,
             relaxed_identity=None):
        '''Return a name for the {+variable}.

By default the name is the first found of the following:

  1. The `standard_name` CF property.
  
  2. The `long_name` CF property, preceeded by the string
     ``'long_name:'``.

  3. The `!id` attribute.

  4. The `!ncvar` attribute, preceeded by the string ``'ncvar%'``.
  
  5. The value of the *default* parameter.

Note that ``f.{+name}(identity=True)`` is equivalent to ``f.identity()``.

.. seealso:: `identity`

:Examples 1:

>>> n = f.{+name}()
>>> n = f.{+name}(default='NO NAME')

:Parameters:

    default: optional
        If no name can be found then return the value of the *default*
        parameter. By default the default is `None`.

    identity: `bool`, optional
        If True then only 1., 3. and 5. are considered as possible
        names.
 
    ncvar: `bool`, optional
        If True then only 4. and 5. are considered as possible names.

:Returns:

    out:
        The name.

:Examples 2:

>>> f.standard_name = 'air_temperature'
>>> f.long_name = 'temperature of the air'
>>> f.ncvar = 'tas'
>>> f.{+name}()
'air_temperature'
>>> del f.standard_name
>>> f.{+name}()
'long_name:temperature of the air'
>>> del f.long_name
>>> f.{+name}()
'ncvar:tas'
>>> del f.ncvar
>>> f.{+name}()
None
>>> f.{+name}('no_name')
'no_name'
>>> f.standard_name = 'air_temperature'
>>> f.{+name}('no_name')
'air_temperature'

        '''

        if relaxed_identity is None:
            relaxed_identity = RELAXED_IDENTITIES()

        if ncvar:
            if identity:
                raise ValueError(
"Can't find identity/ncvar: ncvar and identity parameters can't both be True")

            if relaxed_identity:
                raise ValueError(
"Can't find identity/ncvar: ncvar and relaxed_identity parameters can't both be True")

            n = getattr(self, 'ncvar', None)
            if n is not None:
                return 'ncvar%{0}'.format(n)
            
            return default
        #--- End: if

        n = self.getprop('standard_name', None)
        if n is not None:
            return n

        if identity or relaxed_identity:
            n = getattr(self, 'id', None)
            if n is not None:
#                return 'id%{0}'.format(n) #n
                return n
            if not relaxed_identity:
                return default

        n = self.getprop('long_name', None)
        if n is not None:
            return 'long_name:{0}'.format(n)

        if not relaxed_identity:
            n = getattr(self, 'id', None)
            if n is not None:
                return 'id%{0}'.format(n) #n

        n = getattr(self, 'ncvar', None)
        if n is not None:
            return 'ncvar%{0}'.format(n)

        return default
    #--- End: def

    def HDF_chunks(self, *chunksizes):
        '''{+HDF_chunks}
        
.. versionadded:: 1.1.13

:Examples 1:
        
To define chunks which are the full size for each axis except for the
first axis which is to have a chunk size of 12:

>>> old_chunks = f.{+name}({0: 12})

:Parameters:

    {+chunksizes}

:Returns:

    out: `dict`
        The chunk sizes prior to the new setting, or the current
        current sizes if no new values are specified.

        '''
        if self._hasData:
            old_chunks = self.Data.HDF_chunks(*chunksizes)
        else:
            old_chunks = None

        if self._hasbounds:
            self.bounds.HDF_chunks(*chunksizes)

        return old_chunks
    #--- End: def

    def override_calendar(self, calendar, i=False):
        '''Override the calendar of date-time units.

The new calendar **need not** be equivalent to the original one and
the data array elements will not be changed to reflect the new
units. Therefore, this method should only be used when it is known
that the data array values are correct but the calendar has been
incorrectly encoded.

Not to be confused with setting the `calendar` or `Units` attributes
to a calendar which is equivalent to the original calendar

.. seealso:: `calendar`, `override_units`, `units`, `Units`

:Examples 1:

>>> g = f.{+name}('noleap')

:Parameters:

    calendar: `str`
        The new calendar.

    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

        '''
#        # List functionality
#        if self._list:
#            kwargs2 = self._parameters(locals())
#            return self._list_method('override_calendar', kwargs2)

        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.override_calendar(calendar, i=True)
        else:
            if not v.Units.isreftime:
                raise ValueError(
"Can't override the calender of non-reference-time units: {0!r}".format(
    self.Units))
                
            v.Units = Units(getattr(v.Units, 'units', None), calendar=calendar)
        #--- End: if

        return v
    #--- End: def

    def override_units(self, units, i=False):
        '''Override the units.

The new units **need not** be equivalent to the original ones and the
data array elements will not be changed to reflect the new
units. Therefore, this method should only be used when it is known
that the data array values are correct but the units have incorrectly
encoded.

Not to be confused with setting `units` or `Units` attributes to units
which are equivalent to the original units.

.. seealso:: `calendar`, `override_calendar`, `units`, `Units`

:Examples 1:

>>> g = f.{+name}('m')

:Parameters:

    units: `str` or `cf.Units`
        The new units for the data array.

    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples 2:

>>> f.Units
<CF Units: hPa>
>>> f.datum(0)
100000.0
>>> f.{+name}('km')
>>> f.Units
<CF Units: km>
>>> f.datum(0)
100000.0
>>> f.{+name}(cf.Units('watts'))
>>> f.Units
<CF Units: watts>
>>> f.datum(0)
100000.0

        '''
#        # List functionality
#        if self._list:
#            kwargs2 = self._parameters(locals())
#            return self._list_method('override_units', kwargs2)

        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.override_units(units, i=True)
        else:
            v.Units = Units(units)

        return v
    #--- End: def

    def properties(self, props=None, clear=False, copy=True):
        '''Inspect or change the CF properties.

:Examples 1:

>>> f.{+name}()

:Parameters:

    props: `dict`, optional   
        Set {+variable} attributes from the dictionary of values. If
        the *copy* parameter is True then the values in the *attrs*
        dictionary are deep copied

    clear: `bool`, optional
        If True then delete all CF properties.

    copy: `bool`, optional
        If True then the values in the returned dictionary are deep
        copies of the {+variable}'s attribute values. By default they
        are not copied.

:Returns:

    out: `dict`
        The CF properties prior to being changed, or the current CF
        properties if no changes were specified.

:Examples 2:

        '''
        if copy:            
            out = deepcopy(self._simple_properties())
        else:
            out = self._simple_properties().copy()
            
        if clear:
            self._simple_properties().clear()
            return out

        if not props:
            return out

        setprop = self.setprop
        delprop = self.delprop
        if copy:
            for prop, value in props.iteritems():
                if value is None:
                    # Delete this property
                    delprop(prop)
                else:
                    setprop(prop, deepcopy(value))
        else:
            for prop, value in props.iteritems():
                if value is None:
                    # Delete this property
                    delprop(prop)
                else:
                    setprop(prop, value)

        return out
    #--- End: def

    def attributes(self, attrs=None, copy=True):
        '''Inspect or change attributes which are not CF properties.

:Examples 1:

>>> f.{+name}()

:Parameters:

    attrs: `dict`, optional
        Set {+variable} attributes from the dictionary of values. If
        the *copy* parameter is True then the values in the *attrs*
        dictionary are deep copied

    clear: `bool`, optional
        If True then delete all attributes.

    copy: `bool`, optional
        If True then the values in the returned dictionary are deep
        copies of the {+variable}'s attribute values. By default they
        are not copied.

:Returns:

    out: `dict`

:Examples:

>>> f.{+name}()
{}
>>> f.foo = 'bar'
>>> f.{+name}()
{'foo': 'bar'}
>>> f.{+name}().pop('foo')
'bar'
>>> f.{+name}()
{'foo': 'bar'}

>>> f.{+name}({'name': 'value'})
{'foo': 'bar', 'name': 'value'}

        ''' 
 
        if copy:
            out = deepcopy(self.__dict__)
        else:
            out = self.__dict__.copy()

        del out['_hasbounds']
        del out['_hasData']
        del out['_private']
        del out['_direction']
        
        if not attrs:
            return out

        if copy:
            for attr, value in attrs.iteritems():
                setattr(self, attr, deepcopy(value))
        else:
            for attr, value in attrs.iteritems():
                setattr(self, attr, value)

        return out
    #--- End: def

    def rint(self, bounds=True, i=False):
        '''Round data array.

The scalar ``x`` is rounded to the nearest integer ``i``.

.. versionadded:: 1.0

.. seealso:: `ceil`, `floor`, `trunc`

:Examples 1:

>>> g = f.{+name}()

:Parameters:

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with rounded data array values.

:Examples 2:

>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}().array
[-2. -2. -1. -1.  0.  1.  1.  2.  2.]
>>> print f.array
[-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
>>> print f.{+name}(i=True).array
[-2. -2. -1. -1.  0.  1.  1.  2.  2.]
>>> print f.array
[-2. -2. -1. -1.  0.  1.  1.  2.  2.]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.rint(i=True)

        if bounds and v._hasbounds:
            v.bounds.rint(i=True)

        return v
    #--- End: def

    def round(self, decimals=0, bounds=True, i=False):
        '''Round the data array.

Data elements are evenly rounded to the given number of decimals.

.. note:: Values exactly halfway between rounded decimal values are
          rounded to the nearest even value. Thus 1.5 and 2.5 round to
          2.0, -0.5 and 0.5 round to 0.0, etc. Results may also be
          surprising due to the inexact representation of decimal
          fractions in the IEEE floating point standard and errors
          introduced when scaling by powers of ten.
 
.. versionadded:: 1.1.4

.. seealso:: `ceil`, `floor`, `rint`, `trunc`

:Examples 1:

>>> g = f.round(2)

:Parameters:
	
    decimals: `int`, optional
        Number of decimal places to round to (0 by default). If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.

    {+bounds}

    {+i}

:Returns:

    out: `cf.{+Variable}`
        The {+variable} with rounded data array values.

:Examples 2:

>>> print f.array
[-1.81, -1.41, -1.01, -0.91,  0.09,  1.09,  1.19,  1.59,  1.99])
>>> print f.{+name}().array
[-2., -1., -1., -1.,  0.,  1.,  1.,  2.,  2.]
>>> print f.{+name}(1).array
[-1.8, -1.4, -1. , -0.9,  0.1,  1.1,  1.2,  1.6,  2. ]
>>> print f.{+name}(-1).array
[-0., -0., -0., -0.,  0.,  0.,  0.,  0.,  0.]

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.round(decimals=decimals, i=True)

        if bounds and v._hasbounds:
            v.bounds.round(decimals=decimals, i=True)

        return v
    #--- End: def


    def roll(self, iaxis, shift, i=False):
        '''Roll the {+variable} along an axis.

.. seealso:: `expand_dims`, `flip`, `squeeze`, `transpose`

:Parameters:

    iaxis: `int`
        
    {+i}

:Returns:

    out: `cf.{+Variable}`

:Examples:

        '''
        if i:
            v = self
        else:
            v = self.copy()

        if v._hasData:
            v.Data.roll(iaxis, shift, i=True)

        # Roll the bounds, if there are any
        if v._hasbounds:
            iaxis = self._parse_axes([iaxis])[0]
            v.bounds.roll(iaxis, shift, i=True)

        return v
    #--- End: def

#    def setattr(self, attr, value):
#         '''
#
#Set a named attribute.
#
#``f.setattr(attr, value)`` is equivalent to ``setattr(f, attr,
#value)``.
#
#.. seealso:: `delattr`, `hasattr`, `getattr`
#
#:Parameters:
#
#    attr: `str`
#        The attribute's name.
#
#    value:
#        The value to set the attribute.
#
#:Returns:
#
#    `None`
#
#:Examples:
#
#>>> f.setattr('foo', -99)
#>>> f.foo
#-99
#
#'''
#     
#         setattr(self, attr, value)
#    #--- End: def

    def where(self, condition, x=None, y=None, i=False):
        '''Set data array elements depending on a condition.

.. seealso:: `cf.masked`, `hardmask`, `subspace`

:Returns:

    out: `cf.{+Variable}`

        '''
        if not self._hasData:
            raise ValueError(
                "ERROR: Can't set data in nonexistent data array")
        
        if isinstance(condition, Variable):
            if not condition._hasData:
                raise ValueError(
                    "ERROR: Can't set data when %r condition has no data array" %
                    condition.__class__.__name__)
            condition = condition.Data
        #--- End: if

        if x is not None and isinstance(x, Variable):
            if not x._hasData:
                raise ValueError(
                    "ERROR: Can't set data from %r with no data array" % 
                    x.__class__.__name__)
            x = x.Data
        #--- End: if

        if y is not None and isinstance(y, Variable):
            if not y._hasData:
                raise ValueError(
"ERROR: Can't set data from {!r} with no data array".format(
    y.__class__.__name__))
            y = y.Data
        #--- End: if
        
        if i:
            v = self
        else:
            v = self.copy()

        v.Data.where(condition, x, y, i=True)

        return v
    #--- End: def

#--- End: class


# ====================================================================
#
# SubspaceVariable object
#
# ====================================================================

class SubspaceVariable(object):

    __slots__ = ('variable',)

    def __init__(self, variable):
        '''

Set the contained variable.

'''
        self.variable = variable
    #--- End: def

    def __getitem__(self, indices):
        '''

Called to implement evaluation of x[indices].

x.__getitem__(indices) <==> x[indices]

'''
        return self.variable[indices]
    #--- End: def

    def __setitem__(self, indices, value):
        '''

Called to implement assignment to x[indices]

x.__setitem__(indices, value) <==> x[indices]

'''
        if isinstance(value, Variable):
            value = value.Data

        self.variable[indices] = value
    #--- End: def

#--- End: class
