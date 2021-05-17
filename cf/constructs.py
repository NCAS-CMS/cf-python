import cfdm

from .query import Query


class Constructs(cfdm.Constructs):
    """A container for metadata constructs.

    The following metadata constructs can be included:

    * auxiliary coordinate constructs
    * coordinate reference constructs
    * cell measure constructs
    * dimension coordinate constructs
    * domain ancillary constructs
    * domain axis constructs
    * cell method constructs
    * field ancillary constructs

    The container may be used by `Field` and `Domain` instances. In
    the latter case cell method and field ancillary constructs must be
    flagged as "ignored" (see the *_ignore* parameter).

    The container is like a dictionary in many ways, in that it stores
    key/value pairs where the key is the unique construct key with
    corresponding metadata construct value, and provides some of the
    usual dictionary methods.

    **Calling**

    Calling a `Constructs` instance selects metadata constructs by
    identity and is an alias for the `filter_by_identity` method. For
    example, to select constructs that have an identity of
    'air_temperature': ``d = c('air_temperature')``.

    .. versionadded:: 3.0.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @classmethod
    def _matching_values(cls, value0, construct, value1, basic=False):
        """Whether two values match according to equality on a given
        construct.

        The definition of "match" depends on the types of *value0* and
        *value1*.

        :Parameters:

            value0:
                The first value to be matched.

            construct:
                The construct whose `_equals` method is used to
                determine whether values can be considered to match.

            value1:
                The second value to be matched.

            basic: `bool`
                If True then value0 and value1 will be compared with
                the basic ``==`` operator.

        :Returns:

            `bool`
                Whether or not the two values match.

        """
        if isinstance(value0, Query):
            return value0.evaluate(value1)

        return super()._matching_values(value0, construct, value1, basic=basic)

    #     def domain_axis_key(self, identity, default=ValueError()):
    #         '''Return the key of the domain axis construct that is spanned by 1-d
    # coordinate constructs.
    #
    # .. versionadded:: 3.0.0
    #
    # :Parameters:
    #
    #     identity:
    #
    #         Select the 1-d coordinate constructs that have the given
    #         identity.
    #
    #         An identity is specified by a string (e.g. ``'latitude'``,
    #         ``'long_name=time'``, etc.); or a compiled regular expression
    #         (e.g. ``re.compile('^atmosphere')``), for which all constructs
    #         whose identities match (via `re.search`) are selected.
    #
    #         Each coordinate construct has a number of identities, and is
    #         selected if any of them match any of those provided. A
    #         construct's identities are those returned by its `!identities`
    #         method. In the following example, the construct ``x`` has four
    #         identities:
    #
    #            >>> x.identities()
    #            ['time', 'long_name=Time', 'foo=bar', 'ncvar%T']
    #
    #         In addition, each construct also has an identity based its
    #         construct key (e.g. ``'key%dimensioncoordinate2'``)
    #
    #         Note that in the output of a `print` call or `!dump` method, a
    #         construct is always described by one of its identities, and so
    #         this description may always be used as an *identity* argument.
    #
    #     default: optional
    #         Return the value of the *default* parameter if a domain axis
    #         construct can not be found. If set to an `Exception` instance
    #         then it will be raised instead.
    #
    # :Returns:
    #
    #     `str`
    #         The key of the domain axis construct that is spanned by the
    #         data of the selected 1-d coordinate constructs.
    #
    # **Examples:**
    #
    # TODO
    #
    #         '''
    #         # Try for index
    #         try:
    #             da_key = self.get_data_axes(default=None)[identity]
    #         except TypeError:
    #             pass
    #         except IndexError:
    #             return self._default(
    #                 default,
    #                 "Index does not exist for field construct data dimenions")
    #         else:
    #             identity = da_key
    #
    #         domain_axes = self.domain_axes(identity)
    #         if len(domain_axes) == 1:
    #             # identity is a unique domain axis construct identity
    #             da_key = domain_axes.key()
    #         else:
    #             # identity is not a unique domain axis construct identity
    #             da_key = self.domain_axis_key(identity, default=default)
    #
    #         if key:
    #             return da_key
    #
    #         return self.constructs[da_key]

    def _filter_by_identity(self, arg, identities, todict, _config):
        """Worker function for `filter_by_identity` and `filter`.

        See `filter_by_identity` for details.

        .. versionadded:: 3.9.0

        """
        ctypes = [i for i in "XTYZ" if i in identities]

        config = {"identities_kwargs": {"ctypes": ctypes}}
        if _config:
            config.update(_config)

        return super()._filter_by_identity(arg, identities, todict, config)

    #    def _filter_by_coordinate_type(self, arg, ctypes, todict):
    #        """Worker function for `filter_by_identity` and `filter`.
    #
    #        See `filter_by_identity` for details.
    #
    #        .. versionadded:: 3.9.0
    #
    #        """
    #        out, pop = self._filter_preprocess(
    #            arg,
    #            filter_applied={"filter_by_identity": ctypes},
    #            todict=todict,
    #        )
    #
    #        if not ctypes:
    #            # Return all constructs if no coordinate types have been
    #            # provided
    #            return out
    #
    #        for cid, construct in tuple(out.items()):
    #            ok = False
    #            for ctype in ctypes:
    #                if getattr(construct, ctype, False):
    #                    ok = True
    #                    break
    #
    #            if not ok:
    #                pop(cid)
    #
    #        return out

    @classmethod
    def _short_iteration(cls, x):
        """The default short circuit test.

        If this method returns True then only the first identity
        returned by the construct's `!identities` method will be
        checked.

        See `_filter_by_identity` for details.

        .. versionadded:: (cfdm) 1.8.9.0

        :Parameters:

            x: `str`
                The value against which the construct's identities are
                being compared.

        :Returns:

            `bool`
                 Returns `True` if a construct's `identities` method
                 is to short circuit after the first identity is
                 computed, otherwise `False`.

        """
        if not isinstance(x, str):
            return False

        if x in "XTYZ" or x.startswith("measure:") or x.startswith("id%"):
            return True

        return "=" not in x and ":" not in x and "%" not in x
