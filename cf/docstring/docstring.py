'''Define docstring substitutions.

Text to be replaced is specified as a key in the returned dictionary,
with the replacement text defined by the corresponding value.

Special docstring subtitutions, as defined by a class's
`_docstring_special_substitutions` method, may be used in the
replacement text, and will be substituted as ususal.

Replacement text may contain other non-special substitutions.

.. note:: The values are only checked once for embedded non-special
          substitutions, so if the embedded substitution itself
          contains a non-special substitution then the latter will
          *not* be replaced. This restriction is to prevent the
          possibility of infinite recursion.

Keys must be `str` or `re.Pattern` objects:

* If a key is a `str` then the corresponding value must be a string.

* If a key is a `re.Pattern` object then the corresponding value must
  be a string or a callable, as accepted by the `re.Pattern.sub`
  method.

.. versionaddedd:: 3.7.0

'''
_docstring_substitution_definitions = {
    '{{repr}}':
    'CF ',

    # i
    '{{i: deprecated at version 3.0.0}}':
    """i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.""",
}
