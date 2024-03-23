import os
from functools import cmp_to_key

import numpy

from . import cInterface
from .extraData import ExtraDataUnpacker


class UMFileException(Exception):
    pass


# Lookup header pointers
LBLREC = 14  # Length of data record (including any extra data)
LBPACK = 20  # Packing method indicator
LBEGIN = 28  # Disk address/Start Record


class File:
    """A class for a UM file that gives a view of the file including
    sets of PP records combined into variables."""

    def __init__(
        self, path, byte_ordering=None, word_size=None, fmt=None, parse=True
    ):
        """Open and parse a UM file.

        The optional *byte_ordering*, *word_size* and *fmt* arguments
        specify the file type. If all three are set, then this forces the
        file type; otherwise, the file type is autodetected and any of
        them that are set are ignored.

        :Parameters:

            path: `str`
                The name of the UM file.

            byte_ordering: `str`, optional
                'little_endian' or 'big_endian'

            word_size: `int`, optional
                The size in bytes of one word. Either ``4`` or ``8``.

            fmt: `str`, optional
                'FF' or 'PP'

            parse: `bool`, optional
                The default action is to open the file, store the file
                type from the arguments or autodetection as described
                above, and then parse the contents, giving a tree of
                variables and records under the `File` object. However, if
                *parse* is False, then an object is returned in which the
                last step is omitted, so only the file type is stored, and
                there are no variables under it. Such an object can be
                passed when instantiating Rec objects, and contains
                sufficient info about the file type to ensure that the
                `get_data` method of those `Rec` objects will work.

        """
        c = cInterface.CInterface()
        self._c_interface = c

        self.path = path
        self.fd = None
        self.open_fd()

        if byte_ordering and word_size and fmt:
            self.fmt = fmt
            self.byte_ordering = byte_ordering
            self.word_size = word_size
        else:
            self._detect_file_type()

        self.path = path
        file_type_obj = c.create_file_type(
            self.fmt, self.byte_ordering, self.word_size
        )

        # Set the word size used to interpret file pointers
        c.set_word_size(file_type_obj)

        if parse:
            # --------------------------------------------------------
            # Work out information from the file and store it in the
            # `vars` attribute.
            #
            # Note that the word size used to interpret file pointers
            # needs to have been previously set.
            # --------------------------------------------------------
            info = c.parse_file(self.fd, file_type_obj)
            self.vars = info["vars"]
            self._add_back_refs()

    def open_fd(self):
        """(Re)open the low-level file descriptor.

        :Returns:

            `int`
                The file descriptor.

        """
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)

        return self.fd

    def close_fd(self):
        """Close the low-level file descriptor.

        :Returns:

            `None`

        """
        if self.fd:
            os.close(self.fd)

        self.fd = None

    def _detect_file_type(self):
        """Store string values describing the auto-detected file type.

        :Returns:

            `None`

        """
        c = self._c_interface
        try:
            file_type_obj = c.detect_file_type(self.fd)
        except Exception:
            self.close_fd()
            raise IOError(f"File {self.path} has unsupported format")

        d = c.file_type_obj_to_dict(file_type_obj)
        self.fmt = d["fmt"]
        self.byte_ordering = d["byte_ordering"]
        self.word_size = d["word_size"]

    def _add_back_refs(self):
        """Add file attribute to `Var` objects, and both `!file` and
        `!var` attributes to `Rec` objects.

        The important one is the file attribute in the `Rec` object, as
        this is used when reading data. The others are provided for extra
        convenience.

        :Returns:

            `None`

        """
        for var in self.vars:
            var.file = self
            for rec in var.recs:
                rec.var = var
                rec.file = self


class Var:
    """Container for some information about variables."""

    def __init__(self, recs, nz, nt, supervar_index=None):
        self.recs = recs
        self.nz = nz
        self.nt = nt
        self.supervar_index = supervar_index

    @staticmethod
    def _compare(x, y):
        """Method equivalent to the Python 2 'cmp'.

        Note that (x > y) - (x < y) is equivalent but not as performant
        since it would not short-circuit.

        :Returns:

            `int`

        """
        if x == y:
            return 0
        elif x > y:
            return 1
        else:
            return -1

    def _compare_recs_by_extra_data(self, a, b):
        """Compare records with respect to their extra data.

        :Returns:

            `int`

        """
        return self._compare(a.get_extra_data(), b.get_extra_data())

    def _compare_recs_by_orig_order(self, a, b):
        """Compare records with respect to their original order.

        :Returns:

            `int`

        """
        return self._compare(self.recs.index(a), self.recs.index(b))

    def group_records_by_extra_data(self):
        """Group records by matching extra data.

        Returns a list of (sub)lists of records where each record
        within each sublist has matching extra data (if
        any). Therefore, if the whole variable has consistent extra
        data then the returned value will be a list of length 1.

        Within each group, the ordering of returned records is the
        same as in the `!recs` attribute.

        :Returns:

            `list`

        """
        compare = self._compare_recs_by_extra_data
        recs = self.recs[:]
        n = len(recs)
        if n == 0:
            # shouldn't have a var without records, but...
            return []

        #        recs.sort(compare) #python2
        recs.sort(key=cmp_to_key(compare))

        # optimise simple case - if two ends of a sorted list match,
        # the whole list matches
        if not compare(recs[0], recs[-1]):
            return [self.recs[:]]

        groups = []
        this_grp = []
        for i, rec in enumerate(recs):
            this_grp.append(rec)
            if i == n - 1 or compare(rec, recs[i + 1]):
                this_grp.sort(key=self._compare_recs_by_orig_order)
                groups.append(this_grp)
                this_grp = []

        return groups


class Rec:
    """Container for some information about records."""

    def __init__(
        self,
        int_hdr,
        real_hdr,
        hdr_offset,
        data_offset,
        disk_length,
        file=None,
    ):
        """Default instantiation, which stores the supplied headers and
        offsets.

        :Parameters:

            file: `File`, optional
                Used to set the `!file` attribute. Does not need to be
                supplied, but if it is not then it will have to be set on
                the returned `Rec` object before calling `get_data` will
                work. If set it should be set to the `File` object that
                contains the returned `Rec` object. Normally this would be
                done by the calling code instantiating via `File` rather
                than directly.

        """
        self.int_hdr = int_hdr
        self.real_hdr = real_hdr
        self.hdr_offset = hdr_offset
        self.data_offset = data_offset
        self.disk_length = disk_length
        self._extra_data = None
        if file:
            self.file = file

    @classmethod
    def from_file_and_offsets(
        cls, file, hdr_offset, data_offset=None, disk_length=None
    ):
        """Instantiate a `Rec` object from the `File` object and the
        header and data offsets.

        The lookup header is read from disk immediately, and the
        returned record object is ready for calling `get_data`.

        :Parameters:

            file: `File`
                A view of a file including sets of PP records combined
                into variables.

            hdr_offset: `int`
                The file start address of the header, in bytes.

            data_offset: `int`, optional
                The file start address of the data, in bytes. If
                `None`, the default, then the data offset will be
                calculated from the integer header.

            disk_length: `int`
                The length in bytes of the data in the file. If
                `None`, the default, then the disk length will be
                calculated from the integer header.

        :Returns:

             `Rec`

        """
        c = file._c_interface
        word_size = file.word_size
        int_hdr, real_hdr = c.read_header(
            file.fd, hdr_offset, file.byte_ordering, word_size
        )

        if data_offset is None:
            # Calculate the data offset from the integer header
            if file.fmt == "PP":
                # We only support 64-word headers, so the data starts
                # 66 words after the header_offset, i.e. after 64
                # words of the header, plus 2 block control words.
                data_offset = hdr_offset + 66 * word_size
            else:
                # Fields file
                data_offset = int_hdr[LBEGIN] * word_size

        if disk_length is None:
            # Calculate the disk length from the integer header
            disk_length = int_hdr[LBLREC]
            if int_hdr[LBPACK] % 10 == 2:
                # Cray 32-bit packing
                disk_length = disk_length * 4
            else:
                disk_length = disk_length * word_size

        return cls(
            int_hdr,
            real_hdr,
            hdr_offset,
            data_offset,
            disk_length,
            file=file,
        )

    def read_extra_data(self):
        """Read the extra data associated with the record.

        :Returns:

            `numpy.ndarray`

        """
        file = self.file
        c = file._c_interface

        (
            extra_data_offset,
            extra_data_length,
        ) = c.get_extra_data_offset_and_length(
            self.int_hdr, self.data_offset, self.disk_length
        )

        raw_extra_data = c.read_extra_data(
            file.fd,
            extra_data_offset,
            extra_data_length,
            file.byte_ordering,
            file.word_size,
        )

        edu = ExtraDataUnpacker(
            raw_extra_data, file.word_size, file.byte_ordering
        )

        return edu.get_data()

    def get_extra_data(self):
        """Get extra data associated with the record.

        This is done either by reading or using cached read.

        :Returns:

            `numpy.ndarray`

        """
        if self._extra_data is None:
            self._extra_data = self.read_extra_data()

        return self._extra_data

    def get_type_and_num_words(self):
        """Get the data type and number of words.

        :Returns:

            `numpy.dtype`, `int`

        """
        c = self.file._c_interface
        ntype, num_words = c.get_type_and_num_words(self.int_hdr)
        if ntype == "integer":
            dtype = numpy.dtype(c.file_data_int_type)
        elif ntype == "real":
            dtype = numpy.dtype(c.file_data_real_type)

        return dtype, num_words

    def get_data(self):
        """Get the data array associated with the record.

        :Returns:

            `numpy.ndarray`

        """
        file = self.file
        c = file._c_interface
        int_hdr = self.int_hdr
        data_type, nwords = c.get_type_and_num_words(int_hdr)

        return c.read_record_data(
            file.fd,
            self.data_offset,
            self.disk_length,
            file.byte_ordering,
            file.word_size,
            int_hdr,
            self.real_hdr,
            data_type,
            nwords,
        )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    f = File(path)
    print(f.fmt, f.byte_ordering, f.word_size)
    print("num variables: %s" % len(f.vars))
    for varno, var in enumerate(f.vars):
        print()
        print("var %s: nz = %s, nt = %s" % (varno, var.nz, var.nt))
        for recno, rec in enumerate(var.recs):
            print("var %s record %s" % (varno, recno))
            print("hdr offset: %s" % rec.hdr_offset)
            print("data offset: %s" % rec.data_offset)
            print("disk length: %s" % rec.disk_length)
            print("int hdr: %s" % rec.int_hdr)
            print("real hdr: %s" % rec.real_hdr)
            print("data: %s" % rec.get_data())
            print("extra_data: %s" % rec.get_extra_data())
            print("type %s, num words: %s" % rec.get_type_and_num_words())
            # if recno == 1:
            #     rec._extra_data['y'] += .01
            #     print("massaged_extra_data: %s" % rec.get_extra_data())
            print("-----------------------")

        print("all records", var.recs)
        print(
            "records grouped by extra data ", var.group_records_by_extra_data()
        )
        print("===============================")

    f.close_fd()

    # also read a record using saved metadata
    if f.vars:
        fmt = f.fmt
        byte_ordering = f.byte_ordering
        word_size = f.word_size
        myrec = f.vars[0].recs[0]
        hdr_offset = myrec.hdr_offset
        data_offset = myrec.data_offset
        disk_length = myrec.disk_length

        del f

        fnew = File(
            path,
            fmt=fmt,
            byte_ordering=byte_ordering,
            word_size=word_size,
            parse=False,
        )

        rnew = Rec.from_file_and_offsets(
            fnew, hdr_offset, data_offset, disk_length
        )
        print("record read using saved file type and offsets:")
        print("int hdr: %s" % rnew.int_hdr)
        print("real hdr: %s" % rnew.real_hdr)
        print("data: %s" % rnew.get_data())
        print("extra data: %s" % rnew.get_extra_data())
        print("nx = %s" % rnew.int_hdr[18])
        print("ny = %s" % rnew.int_hdr[17])
        rdata = open("recdata0.txt", "w")
        for value in rnew.get_data():
            rdata.write("%s\n" % value)
        rdata.close()
