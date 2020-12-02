'''Define docstring substitutions.

Text to be replaced is specified as a key in the returned dictionary,
with the replacement text defined by the corresponding value.

Special docstring subtitutions, as defined by a class's
`_docstring_special_substitutions` method, may be used in the
replacement text, and will be substituted as usual.

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

.. versionadded:: 3.7.0

'''
_docstring_substitution_definitions = {
    '{{repr}}':
    'CF ',

    # i
    '{{i: deprecated at version 3.0.0}}':
    """i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.""",

    # List comparison
    '{{List comparison}}':
    '''Each construct in the list is compared with its `!equals` method,
    rather than the ``==`` operator.''',

    # construct selection identity
    '{{construct selection identity}}':
    '''A construct has a number of string-valued identities
              (defined by its `!identities` method) and is selected if
              any of them match the *identity* parameter. *identity*
              may be a string or a `Query` object that equals one of a
              construct's identities; or a `re.Pattern` object that
              matches one of a construct's identities via `re.search`.

              Note that in the output of a `print` call or `!dump`
              method, a construct is always described by one of its
              identities, and so this description may always be used
              as an *identity* argument.''',

    # domain axis selection identity
    '{{domain axis selection identity}}':
    '''A domain axis construct has a number of string-valued
              identities (defined by its `!identities` method) and is
              selected if any of them match the *identity* parameter.
              *identity* may be a string or a `Query` object that
              equals one of a construct's identities; or a
              `re.Pattern` object that matches one of a construct's
              identities via `re.search`.''',

}
