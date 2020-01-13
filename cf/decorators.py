from functools import wraps


# Use a generic attribute name as this is applicable to various objects:
INPLACE_ENABLED_PLACEHOLDER = '_x'


def _inplace_enabled(operation_func):
    '''A decorator enabling operations to be applied in-place.

    If the decorated function has keyword argument 'inplace' being equal to
    True, the function will be performed on 'self' and return None, otherwise
    it will operate on a copy of 'self' & return the processed copy.
    '''
    @wraps(operation_func)

    def inplace_wrapper(self, *args, **kwargs):
        is_inplace = kwargs.get('inplace')
        if is_inplace:
            # create an attribute equal to 'self'
            setattr(self, INPLACE_ENABLED_PLACEHOLDER, self)
        else:
            # create an attribute equal to a copy of 'self'
            setattr(self, INPLACE_ENABLED_PLACEHOLDER, self.copy())

        x_res = operation_func(self, *args, **kwargs)

        if is_inplace:
            return  # decorated function returns None in this case
        else:
            return x_res

    return inplace_wrapper


def _inplace_enabled_define_and_cleanup(obj):
    '''Delete attribute set by inable_enabled but store and return its value.

    Designed as a convenience function for use at the start of methods
    decorated by inplace_enabled; the core construct variable used throughout
    the decorated method should first be assigned to this function with the
    class instance as the input. For example:

    d = _inplace_enabled_define_and_cleanup(self)

    should be set initially for a method operating inplace or otherwise
    (via inplace_enabled) on a data array, d.

    In doing so, the relevant construct variable can be defined appropriately
    and the internal attribute created for that purpose by inplace_enabled
    (which is no longer required) can be cleaned up, all in one line.'''
    x = getattr(obj, INPLACE_ENABLED_PLACEHOLDER)
    delattr(obj, INPLACE_ENABLED_PLACEHOLDER)

    return x


def _deprecation_error_kwargs(operation_func):
    '''A decorator for adding a keyword argument deprecation check.'''
    @wraps(operation_func)

    def precede_with_kwargs_deprecation_check(self, *args, **kwargs):
        i_kwarg = kwargs.get('i')

        if i_kwarg:
            _DEPRECATION_ERROR_KWARGS(
                self, operation_func.__name__, i=i_kwarg) # pragma: no cover

        # Call, but save the output to return afterwards:
        func = operation_func(self, *args, **kwargs)

        return func  # decorated function returns None in this case

    return precede_with_kwargs_deprecation_check
