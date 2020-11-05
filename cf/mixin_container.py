'''.. note:: This class is not in the cf.mixin package because it
             needs to be imported by cf.Data, and some of the other
             mixin classes in cf.mixin themsleves import cf.Data,
             which would lead to a circular import situation.

'''

import logging

from .docstring import _docstring_substitution_definitions

logger = logging.getLogger(__name__)


class Container:
    '''Mixin class for storing components.

    .. versionadded:: 3.7.0

    '''
    def __docstring_substitutions__(self):
        '''Define docstring substitutions that apply to this class and all of
    its subclasses.

    These are in addtion to, and take precendence over, docstring
    substitutions defined by the base classes of this class.

    See `_docstring_substitutions` for details.

    .. versionadded:: 3.7.0

    .. seealso:: `_docstring_substitutions`

    :Returns:

        `dict`
            The docstring substitutions that have been applied.

        '''
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        '''Return the package depth for {{package}} docstring substitutions.

    See `_docstring_package_depth` for details.

        '''
        return 0

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _log_call(self, method, kwargs, log_level='info'):
        '''TODO

    .. versionadded:: 1.8.9.0

    :Parameters:
        
        method: str
        
        kwargs: dict

        log_level: str, optional

    :Returns:

        None

        '''
        kwargs = ["{}={!r}".format(k, v) for k, v in kwargs.items()]

        f = "{}.{}(".format(
            self.__class__.__name__, method
        )

        getattr(logger, log_level)("{}{})".format(
            f, (',\n' + ' ' * len(f)).join(kwargs))
        )
    
# --- End: class
