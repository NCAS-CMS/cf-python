from functools import wraps

from .functions import _DEPRECATION_ERROR_KWARGS


# Identifier for 'inplace_enabled' to use as internal '_custom' dictionary key,
# or directly as a (temporary) attribute name if '_custom' is not provided:
INPLACE_ENABLED_PLACEHOLDER = '_to_assign'


def _inplace_enabled(operation_method):
    '''A decorator enabling operations to be applied in-place.

    If the decorated method has keyword argument `inplace` being equal to
    True, the function will be performed on `self` and return None, otherwise
    it will operate on a copy of `self` & return the processed copy.

    Note that methods decorated with this should assign the core variable
    storing the relevant instance for use throughout the method to
    `_inplace_enabled_define_and_cleanup(self)`.

    '''
    @wraps(operation_method)
    def inplace_wrapper(self, *args, **kwargs):
        is_inplace = kwargs.get('inplace')
        try:
            if is_inplace:
                # create an attribute equal to 'self'
                self._custom[INPLACE_ENABLED_PLACEHOLDER] = self
            else:
                # create an attribute equal to a (shallow) copy of 'self'
                self._custom[INPLACE_ENABLED_PLACEHOLDER] = self.copy()
        # '_custom' not available for object so have to use a direct attribute
        # for the storage, which is not as desirable since it is more exposed:
        except AttributeError:
            if is_inplace:
                self.INPLACE_ENABLED_PLACEHOLDER = self
            else:
                self.INPLACE_ENABLED_PLACEHOLDER = self.copy()

        processed_copy = operation_method(self, *args, **kwargs)

        if is_inplace:
            return  # decorated function returns None in this case
        else:
            return processed_copy

    return inplace_wrapper


def _inplace_enabled_define_and_cleanup(instance):
    '''Delete attribute set by inable_enabled but store and return its value.

    Designed as a convenience function for use at the start of methods
    decorated by inplace_enabled; the core variable used throughout for the
    instance in the decorated method should first be assigned to this
    function with the class instance as the input. For example:

    d = _inplace_enabled_define_and_cleanup(self)

    should be set initially for a method operating inplace or otherwise
    (via inplace_enabled) on a data array, d.

    In doing so, the relevant construct variable can be defined appropriately
    and the internal attribute created for that purpose by inplace_enabled
    (which is no longer required) can be cleaned up, all in one line.

    '''
    try:
        x = instance._custom.pop(INPLACE_ENABLED_PLACEHOLDER)
    except (AttributeError, KeyError):
        x = instance.INPLACE_ENABLED_PLACEHOLDER
        del instance.INPLACE_ENABLED_PLACEHOLDER

    return x


# @_deprecated_kwarg_check('i') -> example usage for decorating, using i kwarg
def _deprecated_kwarg_check(*depr_kwargs):
    '''A wrapper for provision of positional arguments to the decorator.'''
    def deprecated_kwarg_check_decorator(operation_method):
        '''A decorator for a deprecation check on given kwargs.

        For a specified list `deprecated_kwargs`, check if the decorated
        method has been supplied with any of the elements as keyword arguments
        and if so, call _DEPRECATION_ERROR_KWARGS on them, optionally
        providing a custom message to raise inside it.

        '''
        @wraps(operation_method)
        def precede_with_kwarg_deprecation_check(self, *args, **kwargs):

            # If there is only one input deprecated kwarg, form a list so the
            # following loop does not iterate over characters. This means we
            # do not have to write 'dkwarg,' instead of 'dkwarg' each time.
            if isinstance(depr_kwargs, str):
                # Note: can't simply reassign to depr_kwarg and use that in the
                # following loop to iterate over, as this leads to an
                # UnboundLocalError. Easiest way is to use another variable.
                depr_kwargs_list = [depr_kwargs]
            else:
                depr_kwargs_list = depr_kwargs

            for depr_kwarg in depr_kwargs_list:
                if kwargs.get(depr_kwarg):
                    pass_in_kwarg = {depr_kwarg: True}
                    _DEPRECATION_ERROR_KWARGS(
                        self, operation_method.__name__, pass_in_kwarg
                    )  # pragma: no cover

            operation_method_result = operation_method(self, *args, **kwargs)

            # Decorated method has same return signature as if undecorated:
            return operation_method_result

        return precede_with_kwarg_deprecation_check
    return deprecated_kwarg_check_decorator
