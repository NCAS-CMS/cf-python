from . import mixin


class SubspaceField(mixin.Subspace):
    """Create a subspace of the field construct.

    Creation of a new field construct which spans a subspace of the
    domain of an existing field construct is achieved either by
    identifying indices based on the metadata constructs (subspacing
    by metadata) or by indexing the field construct directly
    (subspacing by index).

    The subspacing operation, in either case, also subspaces any
    metadata constructs of the field construct (e.g. coordinate
    metadata constructs) which span any of the domain axis constructs
    that are affected. The new field construct is created with the
    same properties as the original field construct.

    **Subspacing by metadata**

    Subspacing by metadata, signified by the use of round brackets,
    selects metadata constructs and specifies conditions on their
    data. Indices for subspacing are then automatically inferred from
    where the conditions are met.

    Metadata constructs and the conditions on their data are defined
    by keyword parameters.

    * Any domain axes that have not been identified remain unchanged.

    * Multiple domain axes may be subspaced simultaneously, and it
      doesn't matter which order they are specified in.

    * Subspace criteria may be provided for size 1 domain axes that
      are not spanned by the field construct's data.

    * Explicit indices may also be assigned to a domain axis
      identified by a metadata construct, with either a Python `slice`
      object, or a sequence of integers or booleans.

    * For a dimension that is cyclic, a subspace defined by a slice or
      by a `Query` instance is assumed to "wrap" around the edges of
      the data.

    * Conditions may also be applied to multi-dimensional metadata
      constructs. The "compress" mode is still the default mode (see
      the positional arguments), but because the indices may not be
      acting along orthogonal dimensions, some missing data may still
      need to be inserted into the field construct's data.

    **Subspacing by index**

    Subspacing by indexing, signified by the use of square brackets,
    uses rules that are very similar to the numpy indexing rules, the
    only differences being:

    * An integer index i specified for a dimension reduces the size of
      this dimension to unity, taking just the i-th element, but keeps
      the dimension itself, so that the rank of the array is not
      reduced.

    * When two or more dimensions’ indices are sequences of integers
      then these indices work independently along each dimension
      (similar to the way vector subscripts work in Fortran). This is
      the same indexing behaviour as on a Variable object of the
      netCDF4 package.

    * For a dimension that is cyclic, a range of indices specified by
      a `slice` that spans the edges of the data (such as ``-2:3`` or
      ``3:-2:-1``) is assumed to "wrap" around, rather then producing
      a null result.

    **Halos**

    If a halo is defined via a positional argument, then each
    subspaced axis will be extended to include that many extra
    elements at each "side" of the axis. The number of extra elements
    will be automatically reduced if including the full amount defined
    by the halo would extend the subspace beyond the axis limits.

    For instance, ``f.subspace(X=slice(10, 20))`` will give identical
    results to each of ``f.subspace(0, X=slice(10, 20))``,
    ``f.subspace(1, X=slice(11, 19))``, ``f.subspace(2, X=slice(12,
    18))``, etc.

    .. seealso:: `cf.Field.indices`, `cf.Field.where`,
                 `cf.Field.__getitem__`, `cf.Field.__setitem__`,
                 `cf.Domain.subspace`

    :Parameters:

        config: optional
            Configure the subspace by specifying the mode of operation
            (``mode``) and any halo to be added to the subspaced axes
            (``halo``), with positional arguments in the format
            ``mode``, or ``halo``, or ``mode, halo``, or with no
            positional arguments at all.

            A mode of operation is given as a `str`, and a halo as a
            non-negative `int` (or any object that can be converted to
            one):

            ==============  ======================================
            *mode*          Description
            ==============  ======================================
            Not provided     If no positional arguments are
                            provided then assume the
                            ``'compress'`` mode of operation with
                            no halo added to the subspaced axes.

            ``mode``        Define the mode of operation with no
                            halo added to the subspaced axes.

            ``mode, halo``  Define a mode of operation, as well as
                            a halo to be added to the subspaced
                            axes.

            ``halo``        Assume the ``'compress'`` mode of
                            operation and define a halo to be
                            added to the subspaced axes.
            ==============  ======================================

            Valid modes are:

            * ``'compress'`` This is the default.
                 Unselected locations are removed to create the
                 subspace. If the result is not hyperrectangular then
                 the minimum amount of unselected locations required
                 to make it so will also be specially
                 selected. Missing data is inserted at the specially
                 selected locations, unless a halo has been defined
                 (of any size, including 0).

            * ``'envelope'``
                 The subspace is the smallest hyperrectangular
                 subspace that contains all of the selected
                 locations. Missing data is inserted at unselected
                 locations within the envelope, unless a halo has been
                 defined (of any size, including 0).

            * ``'full'``
                 The subspace has the same domain as the original
                 construct. Missing data is inserted at unselected
                 locations, unless a halo has been defined (of any
                 size, including 0).

            .. note:: Setting a halo size of `0` differs from not not
                      defining a halo at all. The shape of the
                      returned field will always be the same, but in
                      the former case missing data will not be
                      inserted at unselected locations (if any) within
                      the output domain.

            In addition, an extra positional argument of ``'test'`` is
            allowed. When provided, the subspace is not returned,
            instead `True` or `False` is returned depending on whether
            or not it is possible for the requested subspace to be
            created.

        keyword parameters: optional
            A keyword name is an identity of a metadata construct, and
            the keyword value provides a condition for inferring
            indices that apply to the dimension (or dimensions)
            spanned by the metadata construct's data. Indices are
            created that select every location for which the metadata
            construct's data satisfies the condition.

    :Returns:

        `Field` or `bool`
            An independent field construct containing the subspace of
            the original field. If the ``'test'`` positional argument
            has been set then return `True` or `False` depending on
            whether or not it is possible to create specified
            subspace.

    **Examples**

    There are further worked examples
    :ref:`in the tutorial <Subspacing-by-metadata>`.

    >>> g = f.subspace(X=112.5)
    >>> g = f.subspace(X=112.5, latitude=cf.gt(-60))
    >>> g = f.subspace(latitude=cf.eq(-45) | cf.ge(20))
    >>> g = f.subspace(X=[1, 2, 4], Y=slice(None, None, -1))
    >>> g = f.subspace(X=cf.wi(-100, 200))
    >>> g = f.subspace(X=slice(-2, 4))
    >>> g = f.subspace(Y=[True, False, True, True, False])
    >>> g = f.subspace(T=410.5)
    >>> g = f.subspace(T=cf.dt('1960-04-16'))
    >>> g = f.subspace(T=cf.wi(cf.dt('1962-11-01'), cf.dt('1967-03-17 07:30')))
    >>> g = f.subspace('compress', X=[1, 2, 4, 6])
    >>> g = f.subspace('envelope', X=[1, 2, 4, 6])
    >>> g = f.subspace('full', X=[1, 2, 4, 6])
    >>> g = f.subspace(latitude=cf.wi(51, 53))

    >>> g = f.subspace[::-1, 0]
    >>> g = f.subspace[:, :, 1]
    >>> g = f.subspace[:, 0]
    >>> g = f.subspace[..., 6:3:-1, 3:6]
    >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
    >>> g = t.subspace[0, :, -2]
    >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
    >>> g = f.subspace[:, -2:3]
    >>> g = f.subspace[:, 3:-2:-1]
    >>> g = f.subspace[..., [True, False, True, True, False]]

    """

    __slots__ = []

    def __call__(self, *config, **kwargs):
        """Create a subspace of a field construct.

        Creation of a new field construct which spans a subspace of the
        domain of an existing field construct is achieved by identifying
        indices based on the metadata constructs.

        The subspacing operation also subspaces any metadata constructs of
        the field construct (e.g. coordinate metadata constructs) which
        span any of the domain axis constructs that are affected. The new
        field construct is created with the same properties as the
        original field construct.

        Metadata constructs and the conditions on their data are defined
        by keyword parameters.

        The subspace is defined by identifying indices based on the
        metadata constructs.

        * Any domain axes that have not been identified remain unchanged.

        * Multiple domain axes may be subspaced simultaneously, and it
          doesn't matter which order they are specified in.

        * Subspace criteria may be provided for size 1 domain axes that
          are not spanned by the field construct's data.

        * Explicit indices may also be assigned to a domain axis
          identified by a metadata construct, with either a Python `slice`
          object, or a sequence of integers or booleans.

        * For a dimension that is cyclic, a subspace defined by a slice or
          by a `Query` instance is assumed to "wrap" around the edges of
          the data.

        * Conditions may also be applied to multi-dimensional metadata
          constructs. The "compress" mode is still the default mode (see
          the positional arguments), but because the indices may not be
          acting along orthogonal dimensions, some missing data may still
          need to be inserted into the field construct's data.

        **Halos**

        {{subspace halos}}

        .. seealso:: `cf.Field.indices`, `cf.Field.squeeze`,
                     `cf.Field.where`

        :Parameters:

            positional arguments: *optional*
                There are three modes of operation, each of which provides
                a different type of subspace, plus a testing mode:

                ==============  ==========================================
                *argument*      Description
                ==============  ==========================================
                ``'compress'``  This is the default mode. Unselected
                                locations are removed to create the
                                returned subspace. Note that if a
                                multi-dimensional metadata construct is
                                being used to define the indices then some
                                missing data may still be inserted at
                                unselected locations.

                ``'envelope'``  The returned subspace is the smallest that
                                contains all of the selected
                                indices. Missing data is inserted at
                                unselected locations within the envelope.

                ``'full'``      The returned subspace has the same domain
                                as the original field construct. Missing
                                data is inserted at unselected locations.

                ``'test'``      May be used on its own or in addition to
                                one of the other positional arguments. Do
                                not create a subspace, but return `True`
                                or `False` depending on whether or not it
                                is possible to create the specified
                                subspace.
                ==============  ==========================================

            Keyword parameters: *optional*
                A keyword name is an identity of a metadata construct, and
                the keyword value provides a condition for inferring
                indices that apply to the dimension (or dimensions)
                spanned by the metadata construct's data. Indices are
                created that select every location for which the metadata
                construct's data satisfies the condition.

        :Returns:

            `Field` or `bool`
                An independent field construct containing the subspace of
                the original field. If the ``'test'`` positional argument
                has been set then return `True` or `False` depending on
                whether or not it is possible to create specified
                subspace.

        **Examples**

        There are further worked examples
        :ref:`in the tutorial <Subspacing-by-metadata>`.

        >>> g = f.subspace(X=112.5)
        >>> g = f.subspace(X=112.5, latitude=cf.gt(-60))
        >>> g = f.subspace(latitude=cf.eq(-45) | cf.ge(20))
        >>> g = f.subspace(X=[1, 2, 4], Y=slice(None, None, -1))
        >>> g = f.subspace(X=cf.wi(-100, 200))
        >>> g = f.subspace(X=slice(-2, 4))
        >>> g = f.subspace(Y=[True, False, True, True, False])
        >>> g = f.subspace(T=410.5)
        >>> g = f.subspace(T=cf.dt('1960-04-16'))
        >>> g = f.subspace(T=cf.wi(cf.dt('1962-11-01'), cf.dt('1967-03-17 07:30')))
        >>> g = f.subspace('compress', X=[1, 2, 4, 6])
        >>> g = f.subspace('envelope', X=[1, 2, 4, 6])
        >>> g = f.subspace('full', X=[1, 2, 4, 6])
        >>> g = f.subspace(latitude=cf.wi(51, 53))

        """
        field = self.variable

        test = False
        if "test" in config:
            config = list(config)
            config.remove("test")
            test = True

        if not config and not kwargs:
            if test:
                return True

            return field.copy()

        try:
            indices = field.indices(*config, **kwargs)
            out = field[indices]
        except (ValueError, IndexError) as error:
            if test:
                return False

            raise error
        else:
            if test:
                return True

            return out
