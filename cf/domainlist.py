from . import ConstructList, mixin


class DomainList(mixin.FieldDomainList, ConstructList):
    """An ordered sequence of domain constructs.

    A domain list supports the python list-like operations (such as
    indexing and methods like `!append`). These methods provide
    functionality similar to that of a built-in list. The main
    difference is that when a domain construct element needs to be
    assesed for equality its `~cf.Domain.equals` method is used,
    rather than the ``==`` operator.

    .. versionadded:: 3.11.0

    """

    def __init__(self, domains=None):
        """Initialisation.

        :Parameters:

            domains: (sequence of) `Domain`, optional
                 Create a new list with these domain constructs.

        """
        super().__init__(constructs=domains)
