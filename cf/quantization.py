import cfdm


class Quantization(cfdm.Quantization):
    """A quantization variable.

    A quantization variable describes a quantization algorithm via a
    collection of parameters.

    The ``algorithm`` parameter names a specific quantization
    algorithm via one of the keys in the `algorithm_parameters`
    dictionary.

    The ``implementation`` parameter contains unstandardised text that
    concisely conveys the algorithm provenance including the name of
    the library or client that performed the quantization, the
    software version, and any other information required to
    disambiguate the source of the algorithm employed. The text must
    take the form ``software-name version version-string
    [(optional-information)]``.

    The retained precision of the algorithm is defined with either the
    ``quantization_nsb`` or ``quantization_nsd`` parameter.

    For instance, the following parameters describe quantization via
    the BitRound algorithm, retaining 6 significant bits, and
    implemented by libnetcdf::

       >>> q = {{package}}.{{class}}(
       ...         parameters={'algorithm': 'bitround',
       ...                     'quantization_nsb': 6,
       ...                     'implementation': 'libnetcdf version 4.9.4'}
       ... )
       >>> q.parameters()
       {'algorithm': 'bitround',
        'quantization_nsb': 6,
        'implementation': 'libnetcdf version 4.9.4'}

    See CF section 8.4. "Lossy Compression via Quantization".

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF group attributes}}

    .. versionadded:: 3.18.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        .. versionadded:: 3.18.0

        """
        return super().__repr__().replace("<", "<CF ", 1)
