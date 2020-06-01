from functools import wraps

from .functions import _DEPRECATION_ERROR_KWARGS

import cfdm


# Decorators (and helper functions for these) inherited from cfdm:
_inplace_enabled = cfdm._inplace_enabled
_inplace_enabled_define_and_cleanup = cfdm._inplace_enabled_define_and_cleanup


# @_deprecated_kwarg_check('i') -> example usage for decorating, using i kwarg
def _deprecated_kwarg_check(*depr_kwargs):
    '''A wrapper for provision of positional arguments to the decorator.'''
    def deprecated_kwarg_check_decorator(operation_method):
        '''A decorator for a deprecation check on given kwargs.

        To specify deprecated kwargs, supply them as string arguments, e.g:

            @_deprecated_kwarg_check('i')
            @_deprecated_kwarg_check('i', 'traceback')

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
