"""Define docstring substitutions.

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

"""
_docstring_substitution_definitions = {
    # ----------------------------------------------------------------
    # General docstring susbstitutions
    # ----------------------------------------------------------------
    "{{repr}}": "CF ",
    "{{formula terms links}}": """See the parametric vertical coordinate sections of the CF
    conventions for more details:

    `4.3.3. Parametric Vertical Coordinate
    <https://cfconventions.org/Data/cf-conventions/cf-conventions-{{VN}}/cf-conventions.html#parametric-vertical-coordinate>`_

    `Appendix D: Parametric Vertical Coordinates
    <https://cfconventions.org/Data/cf-conventions/cf-conventions-{{VN}}/cf-conventions.html#parametric-v-coord>`_""",
    # ----------------------------------------------------------------
    # Parameter description substitutions
    # ----------------------------------------------------------------
    "{{i: deprecated at version 3.0.0}}": """i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.""",
    "{{default_to_zero: `bool`, optional}}": """default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero.""",
    # ----------------------------------------------------------------
    # Returns substitutions
    # ----------------------------------------------------------------
    "{{Returns formula}}": """5-`tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple.""",
}
