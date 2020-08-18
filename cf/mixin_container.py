from .docstring import _docstring_substitution_definitions


# Note: This class is not in the cf.mixin package because it needs to
#       be imported by cf.Data, and some of the other mixin classes in
#       cf.mixin themsleves import cf.Data, which would lead to a
#       circular import situation.


class Container:
    '''Mixin class for storing components.

    .. versionadded:: 3.7.0

    '''
    def __docstring_substitution__(self):
        '''Define docstring substitutions that apply to this class and all of
    its subclasses.

    These are in addtion to, and take precendence over, docstring
    substitutions defined by the base classes of this class.

    Text to be replaced is specified as a key in the returned
    dictionary, with the replacement text defined by the corresponding
    value.

    Keys must be `str` or `re.Pattern` objects.

    If a key is a `str` then the corresponding value must be a string.

    If a key is a `re.Pattern` object then the corresponding value
    must be a string or a callable, as accepted by the
    `re.Pattern.sub` method.

    Special docstring subtitutions, as defined by
    `_special_docstring_substitutions`, may be used in the replacement
    text, and will be substituted as ususal.

    .. versionaddedd:: 3.7.0

    .. seealso:: `_docstring_substitutions`,
                 `_special_docstring_substitutions`

    :Returns:

        `dict`
            The docstring substitutions that have been applied.

        '''
        return _docstring_substitution_definitions

# --- End: class
