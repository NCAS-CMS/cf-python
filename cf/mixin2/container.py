"""This class is not in the cf.mixin package because it needs to be
imported by cf.Data, and some of the other mixin classes in cf.mixin
themselves import cf.Data, which would lead to a circular import
situation.

"""

from ..docstring import _docstring_substitution_definitions
from ..functions import atol, rtol


class Container:
    """Mixin class for storing components.

    .. versionadded:: 3.7.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        .. versionadded:: 3.16.0

        """
        return super().__repr__().replace("<", "<CF ", 1)

    def __docstring_substitutions__(self):
        """Define docstring substitutions that apply to this class and
        all of its subclasses.

        These are in addtion to, and take precendence over, docstring
        substitutions defined by the base classes of this class.

        See `_docstring_substitutions` for details.

        .. versionadded:: 3.7.0

        .. seealso:: `_docstring_substitutions`

        :Returns:

            `dict`
                The docstring substitutions that have been applied.

        """
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        """Return the package depth for {{package}} docstring
        substitutions.

        See `_docstring_package_depth` for details.

        """
        return 0

    @property
    def _atol(self):
        """Internal alias for `{{package}}.atol`.

        An alias is necessary to avoid a name clash with the keyword
        argument of identical name (`atol`) in calling functions.

        """
        return atol().value

    @property
    def _rtol(self):
        """Internal alias for `{{package}}.rtol`.

        An alias is necessary to avoid a name clash with the keyword
        argument of identical name (`rtol`) in calling functions.

        """
        return rtol().value
