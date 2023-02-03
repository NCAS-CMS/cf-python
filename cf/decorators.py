from functools import wraps

import cfdm

from .constants import ValidLogLevels
from .functions import (
    _DEPRECATION_ERROR_KWARGS,
    _disable_logging,
    _is_valid_log_level_int,
    _reset_log_emergence_level,
    log_level,
)

# Decorators (and helper functions for these) inherited from cfdm:
_inplace_enabled = cfdm._inplace_enabled
_inplace_enabled_define_and_cleanup = cfdm._inplace_enabled_define_and_cleanup
_manage_log_level_via_verbosity = cfdm._manage_log_level_via_verbosity
_display_or_return = cfdm._display_or_return


# @_deprecated_kwarg_check('i') -> example usage for decorating, using i kwarg
def _deprecated_kwarg_check(*depr_kwargs, version=None, removed_at=None):
    """A wrapper for provision of positional arguments to the
    decorator."""

    def deprecated_kwarg_check_decorator(operation_method):
        """A decorator for a deprecation check on given kwargs.

        To specify deprecated kwargs, supply them as string arguments,
        with the version at which they were deprecated, and optionally
        the version at which the kwargs will be completely
        removed. E.g:

            @_deprecated_kwarg_check('i', version="3.0.0")
            @_deprecated_kwarg_check('i', 'traceback', version="3.0.0")
            @_deprecated_kwarg_check('i', version="3.0.0")
            @_deprecated_kwarg_check('i', version="3.0.0", removed_at="4.0.0")

        For a specified list `depr_kwargs`, check if the
        decorated method has been supplied with any of the elements as
        keyword arguments and if so, call _DEPRECATION_ERROR_KWARGS on
        them, optionally providing a custom message to raise inside
        it.

        """

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
                        self,
                        operation_method.__name__,
                        pass_in_kwarg,
                        version=version,
                        removed_at=removed_at,
                    )  # pragma: no cover

            operation_method_result = operation_method(self, *args, **kwargs)

            # Decorated method has same return signature as if undecorated:
            return operation_method_result

        return precede_with_kwarg_deprecation_check

    return deprecated_kwarg_check_decorator


# TODO(?): we could instead define a metaclass to apply this logic to all
# methods in a class with a self.verbose attibute to avoid decorating perhaps
# most methods of the class (any that contain log calls should have it).
# But this (explicit) approach may be better?
def _manage_log_level_via_verbose_attr(method_using_verbose_attr, calls=[0]):
    """A decorator for managing log message filtering by verbose
    attribute.

    Note this has identical purpose to _manage_log_level_via_verbosity except
    it adapts the log level based on a verbose attribute of the class
    instance, rather than a verbose keyword argument to the function.

    This enables overriding of the log severity level such that an integer
    value (lying in the valid range) of self.verbose on the decorated
    method will ignore the global cf.log_level() to configure a custom
    verbosity for the individual method call, applying to its logic and any
    functions it calls internally and lasting only the duration of the call.

    If self.verbose=None, as is the default, the log_level() determines
    which log messages are shown, as standard.

    Only use this to decorate methods which make log calls directly and are
    in a class with a 'verbose' instance attribute set to None by default.

    Note that the 'calls' keyword argument is to automatically track the number
    of decorated functions that are being (or about to be) executed, with the
    purpose of preventing resetting of the effective log level at the
    completion of decorated functions that are called inside other decorated
    functions (see comments in 'finally' statement for further explanation).
    Note (when it is of concern) that this approach may not be thread-safe.

    """

    @wraps(method_using_verbose_attr)
    def verbose_override_wrapper(self, *args, **kwargs):
        # Increment indicates that one decorated function has started execution
        calls[0] += 1

        lvls = ", ".join(
            [val.name + " = " + str(val.value) for val in ValidLogLevels]
        )
        invalid_attr_msg = (
            f"Invalid value '{self.verbose}' for the 'self.verbose' "
            "attribute. Accepted values are integers corresponding in "
            f"positive cases to increasing verbosity (namely {lvls}), or "
            "None, to configure the verbosity according to the global "
            "log_level setting."
        )
        # Convert Boolean cases for backwards compatibility. Need 'is' identity
        # rather than '==' (value) equivalency test, since 1 == True, etc.
        # Note: it is safe (and desirable) to change these (to standard form)
        if self.verbose is True:
            self.verbose = 3  # max verbosity excluding debug levels
        elif self.verbose is False:
            self.verbose = 0  # corresponds to disabling logs i.e. no verbosity

        # May change during decorator, whereas we don't want to amend the attr:
        verbose_attr = self.verbose
        # First convert valid string inputs to the enum-mapped int constant:
        if isinstance(verbose_attr, str):
            uppercase_attr = verbose_attr.upper()
            if hasattr(ValidLogLevels, uppercase_attr):
                verbose_attr = getattr(ValidLogLevels, uppercase_attr).value
            else:
                raise ValueError(invalid_attr_msg)

        # Override log levels for the function & all it calls (to reset at end)
        if verbose_attr is not None:  # None as default; exclude True & False
            if _is_valid_log_level_int(verbose_attr):
                _reset_log_emergence_level(verbose_attr)
            else:
                raise ValueError(invalid_attr_msg)

        # First need to (temporarily) re-enable global logging if disabled
        # in the cases where you do not want to disable it anyway:
        if log_level() == "DISABLE" and verbose_attr not in (0, None):
            _disable_logging(at_level="NOTSET")  # enables all logging again

        # After method completes, re-set any changes to log level or enabling
        try:
            return method_using_verbose_attr(self, *args, **kwargs)
        except Exception:
            raise
        finally:  # so that crucial 'teardown' code runs even if method errors
            # Decrement indicates one decorated function has finished execution
            calls[0] -= 1
            # Due to the incrementing and decrementing of 'calls', it will
            # only be zero when the outermost decorated method has finished,
            # so the following condition prevents resetting occurring once
            # inner functions complete (which would mean any subsequent code
            # in the outer function would undesirably regain the global level):
            if calls[0] == 0:
                if verbose_attr == 0:
                    _disable_logging(at_level="NOTSET")  # lift deactivation
                elif verbose_attr is not None and _is_valid_log_level_int(
                    verbose_attr
                ):
                    _reset_log_emergence_level(log_level())
                if log_level() == "DISABLE" and verbose_attr != 0:
                    _disable_logging()  # disable again after re-enabling

    return verbose_override_wrapper
