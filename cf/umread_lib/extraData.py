import sys

import numpy as np


def cmp(a, b):
    """Workaround to get a Python-2-like `cmp` function in Python 3."""
    return (a > b) - (a < b)


_codes = {
    1: ("x", float),
    2: ("y", float),
    3: ("y_domain_lower_bound", float),
    4: ("x_domain_lower_bound", float),
    5: ("y_domain_upper_bound", float),
    6: ("x_domain_upper_bound", float),
    7: ("z_domain_lower_bound", float),
    8: ("z_domain_upper_bound", float),
    10: ("title", str),
    11: ("domain_title", str),
    12: ("x_lower_bound", float),
    13: ("x_upper_bound", float),
    14: ("y_lower_bound", float),
    15: ("y_upper_bound", float),
}


class ExtraData(dict):
    """Extends dictionary class with a comparison method between extra
    data for different records."""

    _key_to_type = dict([(key, typ) for key, typ in _codes.values()])

    def sorted_keys(self):
        k = self.keys()
        k.sort()
        return k

    _tolerances = {np.dtype(np.float32): 1e-5, np.dtype(np.float64): 1e-13}

    def _cmp_floats(self, a, b, tolerance):
        if a == b:
            return 0

        delta = abs(b * tolerance)
        if a < b - delta:
            return -1

        if a > b + delta:
            return 1

        return 0

    def _cmp_float_arrays(self, avals, bvals):
        n = len(avals)
        c = cmp(n, len(bvals))
        if c != 0:
            return c

        tolerance = self._tolerances[avals.dtype]
        for i in range(n):
            c = self._cmp_floats(avals[i], bvals[i], tolerance)
            if c != 0:
                return c

        return 0

    def __cmp__(self, other):
        """Compare two extra data dictionaries returned by unpacker."""
        if other is None:
            return 1
        ka = self.sorted_keys()
        kb = other.sorted_keys()
        c = cmp(ka, kb)
        if c != 0:
            return c

        for key in ka:
            valsa = self[key]
            valsb = other[key]
            typ = self._key_to_type[key]
            if typ == float:
                c = self._cmp_float_arrays(valsa, valsb)
            elif type == str:
                c = cmp(valsa, valsb)
            else:
                assert False

            if c != 0:
                return c

        return 0


class ExtraDataUnpacker:
    _int_types = {4: np.int32, 8: np.int64}
    _float_types = {4: np.float32, 8: np.float64}

    def __init__(self, raw_extra_data, word_size, byte_ordering):
        self.rdata = raw_extra_data
        self.ws = word_size
        self.itype = self._int_types[word_size]
        self.ftype = self._float_types[word_size]
        # byte_ordering is 'little_endian' or 'big_endian'
        # sys.byteorder is 'little' or 'big'
        self.is_swapped = not byte_ordering.startswith(sys.byteorder)

    def next_words(self, n):
        """return next n words as raw data string, and pop them off the
        front of the string."""
        pos = n * self.ws
        rv = self.rdata[:pos]
        assert len(rv) == pos
        self.rdata = self.rdata[pos:]
        return rv

    def convert_bytes_to_string(self, st):
        """Convert bytes to string.

        :Returns:

            `str`

        """
        if self.is_swapped:
            indices = slice(None, None, -1)
        else:
            indices = slice(None)

        st = "".join(
            [
                st[pos : pos + self.ws][indices].decode("utf-8")
                for pos in range(0, len(st), self.ws)
            ]
        )

        while st.endswith("\x00"):
            st = st[:-1]

        return st

    def get_data(self):
        """Get the (key, value) pairs for extra data.

        :Returns:

            `ExtraData`

        """
        d = {}
        while self.rdata:
            i = np.frombuffer(self.next_words(1), self.itype)[0]
            if i == 0:
                break

            ia, ib = divmod(i, 1000)
            key, etype = _codes[ib]

            rawvals = self.next_words(ia)
            if etype == float:
                vals = np.frombuffer(rawvals, self.ftype)
            elif etype == str:
                vals = np.array([self.convert_bytes_to_string(rawvals)])

            if key not in d:
                d[key] = vals
            else:
                d[key] = np.append(d[key], vals)

        return ExtraData(d)
