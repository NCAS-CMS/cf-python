import os
import numpy
import numpy.ctypeslib
import ctypes as CT

from . import umfile


_len_real_hdr = 19
_len_int_hdr = 45


class File_type(CT.Structure):
    _fields_ = [
        ("format", CT.c_int),
        ("byte_ordering", CT.c_int),
        ("word_size", CT.c_int)
    ]


def _get_ctypes_array(dtype, size=None):
    """
    get ctypes corresponding to a numpy array of a given type;
    the size should not be necessary unless the storage for the array
    is allocated in the C code
    """
    kwargs = {
        'dtype': dtype,
        'ndim': 1,
        'flags': ('C_CONTIGUOUS', 'WRITEABLE')
    }
    if size:
        kwargs['shape'] = (size,)
    return numpy.ctypeslib.ndpointer(**kwargs)


def _gen_rec_class(int_type, float_type):
    class Rec(CT.Structure):
        """
        ctypes object corresponding to the Rec object in the C code,
        """
        _fields_ = [
            ("int_hdr", _get_ctypes_array(int_type, _len_int_hdr)),
            ("real_hdr", _get_ctypes_array(float_type, _len_real_hdr)),
            ("header_offset", CT.c_size_t),
            ("data_offset", CT.c_size_t),
            ("disk_length", CT.c_size_t),
            ("_internp", CT.c_void_p)
        ]
    return Rec


Rec32 = _gen_rec_class(numpy.int32, numpy.float32)
Rec64 = _gen_rec_class(numpy.int64, numpy.float64)


def _gen_var_class(rec_class):
    class Var(CT.Structure):
        """
        ctypes object corresponding to the Var object in the C code
        """
        _fields_ = [
            ("recs", CT.POINTER(CT.POINTER(rec_class))),
            ("nz", CT.c_int),
            ("nt", CT.c_int),
            ("supervar_index", CT.c_int),
            ("_internp", CT.c_void_p)
        ]
    return Var


Var32 = _gen_var_class(Rec32)
Var64 = _gen_var_class(Rec64)


def _gen_file_class(var_class):
    class File(CT.Structure):
        """
        ctypes object corresponding to the File object in the C code
        """
        _fields_ = [
            ("fd", CT.c_int),
            ("file_type", File_type),
            ("nvars", CT.c_int),
            ("vars", CT.POINTER(CT.POINTER(var_class))),
            ("_internp", CT.c_void_p)
        ]
    return File


File32 = _gen_file_class(Var32)
File64 = _gen_file_class(Var64)


class Enum:
    def __init__(self, *names):
        self.names = names

    def as_name(self, val):
        if isinstance(val, str):
            return val
        else:
            return self.names[val]

    def as_index(self, val):
        if isinstance(val, int):
            return val
        return self.names.index(val)


enum_file_format = Enum("PP", "FF")
enum_byte_ordering = Enum("little_endian", "big_endian")
enum_data_type = Enum("integer", "real")


class CInterface:
    """
    Interface to the C shared library functions
    """
    def __init__(self, lib_name="umfile.so"):
        lib_dir = os.path.join(
            os.path.dirname(__file__) or ".", "c-lib")
        lib_path = os.path.join(lib_dir, lib_name)
        self.lib = CT.CDLL(lib_path)

    def _is_null_pointer(self, ptr):
        try:
            ptr.contents
            return False
        except ValueError:
            return True

    def detect_file_type(self, fd):
        """
        auto-detect file type; returns a File_type ctypes object
        that can be passed to file_parse(), or
        raises an exception if file type cannot be detected
        """
        file_type = File_type()
        rv = self.lib.detect_file_type(fd, CT.pointer(file_type))
        if rv != 0:
            raise umfile.UMFileException("File type could not be detected")
        return file_type

    def file_type_obj_to_dict(self, file_type):
        """
        converts a FileType object returned by detect_file_type()
        into a dictionary that include meaningful string
        values in place of the integers that derive from the C enum
        statments, specifically:
           'format': 'PP' or 'FF'
           'byte_ordering': 'little_endian' or 'big_endian'
        and also
           'word_size': 4 or 8
        """
        format = enum_file_format.as_name(file_type.format)
        byte_ordering = enum_byte_ordering.as_name(file_type.byte_ordering)
        word_size = file_type.word_size
        return {
            'format': format,
            'byte_ordering': byte_ordering,
            'word_size': word_size
            }

    def create_file_type(self, format, byte_ordering, word_size):
        """
        takes string input values:
           'format': 'PP' or 'FF'
           'byte_ordering': 'little_endian' or 'big_endian'
           'word_size': 4 or 8
        and returns a FileType object (ctypes structure containing integer
        values) that can be passed to file_parse()
        """
        return File_type(
            format=enum_file_format.as_index(format),
            byte_ordering=enum_byte_ordering.as_index(byte_ordering),
            word_size=word_size
        )

    def set_word_size(self, val):
        """
        Sets the word size used to interpret returned pointers from subsequent
        calls, in particular the pointers to PP headers embedded in the tree of
        objects returned by file_parse() and the data array that is populated
        by read_record_data().

        the 'val' argument contains either just the word_size value to use or a
        file_type object from which it is to be extracted
        """
        if isinstance(val, File_type):
            word_size = val.word_size
        else:
            word_size = val

        if word_size == 4:
            self.file_class = File32
            self.file_data_int_type = numpy.int32
            self.file_data_real_type = numpy.float32
            self._int_ptr = CT.POINTER(CT.c_int32)
            self._real_ptr = CT.POINTER(CT.c_float)
        elif word_size == 8:
            self.file_class = File64
            self.file_data_int_type = numpy.int64
            self.file_data_real_type = numpy.float64
            self._int_ptr = CT.POINTER(CT.c_int64)
            self._real_ptr = CT.POINTER(CT.c_double)
        else:
            raise ValueError("word size must be 4 or 8 (not %s)" % word_size)

    def _get_ctypes_int_array(self, size=None):
        return _get_ctypes_array(self.file_data_int_type, size)

    def _get_ctypes_real_array(self, size=None):
        return _get_ctypes_array(self.file_data_real_type, size)

    def _get_empty_real_array(self, size):
        """
        get empty numpy real array according to word size previously
        set with set_word_size()
        """
        return numpy.empty(size, dtype=self.file_data_real_type)

    def _get_empty_int_array(self, size):
        """
        as _get_empty_real_array but for int
        """
        return numpy.empty(size, dtype=self.file_data_int_type)

    def parse_file(self, fh, file_type):
        """
        Given an open file handle, work out information from the file, and
        return this in a dictionary, of which currently the only key actually
        implemented is 'vars', containing a list of variables, as that is all
        that the caller requires.

        arguments: fh - low-level file handle (integer)
                   file_type - File_type object as returned by
                               detect_file_type() or create_file_type()
        """
        func = self.lib.file_parse
        file_p_type = CT.POINTER(self.file_class)
        func.restype = file_p_type
        file_p = func(fh, file_type)
        if self._is_null_pointer(file_p):
            raise umfile.UMFileException("file parsing failed")
        file = file_p.contents
        c_vars = file.vars[:file.nvars]
        rv = {'vars': list(map(self.c_var_to_py_var, c_vars))}
        # now that we have copied all the data into python objects for the
        # caller, free any memory allocated in the C code before returning
        free_func = self.lib.file_free
        free_func._fields_ = file_p_type
        free_func(file_p)
        return rv

    def c_var_to_py_var(self, c_var_p):
        """
        create a umfile.Var object from a ctypes object corresponding to 'Var*'
        in the C code
        """
        c_var = c_var_p.contents
        nz = c_var.nz
        nt = c_var.nt
        svi = c_var.supervar_index
        if svi < 0:
            svi = None
        c_recs = c_var.recs
        recs = [self.c_rec_to_py_rec(c_recs[recid]) for recid in
                range(nz * nt)]
        return umfile.Var(recs, nz, nt, svi)

    def c_rec_to_py_rec(self, c_rec_p):
        """
        create a umfile.Rec object from a ctypes object corresponding to 'Rec*'
        in the C code
        """
        c_rec = c_rec_p.contents

        # ============================================================
        # Previous code - causing memory leaks per
        # https://github.com/numpy/numpy/issues/6511
        #
        #     # numpy.copy used here so we can go back and free the memory
        #     # allocated by C without affecting the python object
        #     int_hdr = numpy.copy(numpy.ctypeslib.as_array(c_rec.int_hdr))
        #     real_hdr = numpy.copy(numpy.ctypeslib.as_array(c_rec.real_hdr))
        # ============================================================

        #
        # -------------------------------
        # Workaround: instead cast to a pointer, obtain the values knowing
        # the length of the header arrays, and copy into an appropriate
        # numpy array
        #
        ptr = CT.cast(c_rec.int_hdr, self._int_ptr)
        int_hdr = numpy.array(
            ptr[:_len_int_hdr], dtype=self.file_data_int_type)

        ptr = CT.cast(c_rec.real_hdr, self._real_ptr)
        real_hdr = numpy.array(
            ptr[:_len_real_hdr], dtype=self.file_data_real_type)
        # ===============================================================

        header_offset = c_rec.header_offset
        data_offset = c_rec.data_offset
        disk_length = c_rec.disk_length
        return umfile.Rec(
            int_hdr, real_hdr, header_offset, data_offset, disk_length)

    def get_type_and_num_words(self, int_hdr):
        """
        from integer header, work out data type and number of words to read
        (read_record_data requires this)

        returns 2-tuple of:
           data type: 'integer' or 'real'
           number of words
        """
        word_size = int_hdr.itemsize
        self.lib.get_type_and_num_words.argtypes = [
            CT.c_int,
            self._get_ctypes_int_array(),
            CT.POINTER(CT.c_int),
            CT.POINTER(CT.c_size_t)
        ]
        data_type = CT.c_int()
        num_words = CT.c_size_t()
        rv = self.lib.get_type_and_num_words(
            word_size,
            int_hdr,
            CT.pointer(data_type),
            CT.pointer(num_words)
        )
        if rv != 0:
            raise umfile.UMFileException(
                "error determining data type and size from integer header")
        return enum_data_type.as_name(data_type.value), num_words.value

    def get_extra_data_offset_and_length(
            self, int_hdr, data_offset, disk_length):
        """
        from integer header, gets offset and length of extra data
        returns 2-tuple (offset, length), both in units of BYTES
        """
        word_size = int_hdr.itemsize
        func = self.lib.get_extra_data_offset_and_length
        func.argtypes = [
            CT.c_int,
            self._get_ctypes_int_array(),
            CT.c_size_t,
            CT.c_size_t,
            CT.POINTER(CT.c_size_t),
            CT.POINTER(CT.c_size_t)
        ]
        extra_data_offset = CT.c_size_t()
        extra_data_length = CT.c_size_t()
        rv = func(
            word_size, int_hdr, data_offset, disk_length,
            CT.pointer(extra_data_offset), CT.pointer(extra_data_length)
        )
        if rv != 0:
            raise umfile.UMFileException(
                "error determining extra data length from integer header")
        return extra_data_offset.value, extra_data_length.value

    def read_header(
        self,
        fd,
        header_offset,
        byte_ordering,
        word_size
    ):
        """
        reads header from open file, returning as 2-tuple (int_hdr, real_hdr)
        of numpy arrays
        """
        self.lib.read_header.argtypes = [
            CT.c_int,
            CT.c_size_t,
            CT.c_int,
            CT.c_int,
            self._get_ctypes_int_array(),
            self._get_ctypes_real_array()
        ]

        int_hdr = self._get_empty_int_array(_len_int_hdr)
        real_hdr = self._get_empty_real_array(_len_real_hdr)
        rv = self.lib.read_header(
            fd,
            header_offset,
            enum_byte_ordering.as_index(byte_ordering),
            word_size,
            int_hdr,
            real_hdr
        )
        if rv != 0:
            raise umfile.UMFileException("error reading header data")

        return int_hdr, real_hdr

    def read_extra_data(
        self,
        fd,
        extra_data_offset,
        extra_data_length,
        byte_ordering,
        word_size
    ):
        """
        reads record data from open file

        inputs:
           fd - integer low-level file descriptor
           extra_data_offset - offset in bytes
           extra_disk_length - disk length of extra data in bytes
           byte_ordering - 'little_endian' or 'big_endian'
           word_size - 4 or 8

        returns: extra data as string)
        """

        extra_data = b"\0" * extra_data_length

        self.lib.read_extra_data.argtypes = [
            CT.c_int,
            CT.c_size_t,
            CT.c_size_t,
            CT.c_int,
            CT.c_int,
            CT.c_char_p
        ]

        rv = self.lib.read_extra_data(
            fd,
            extra_data_offset,
            extra_data_length,
            enum_byte_ordering.as_index(byte_ordering),
            word_size,
            extra_data
        )
        if rv != 0:
            raise umfile.UMFileException("error reading extra data")

        return extra_data

    def read_record_data(
        self,
        fd,
        data_offset,
        disk_length,
        byte_ordering,
        word_size,
        int_hdr,
        real_hdr,
        data_type,
        nwords
    ):
        """
        reads record data from open file

        inputs:
           fd - integer low-level file descriptor
           data_offset - offset in words
           disk_length - disk length of data record in words
           byte_ordering - 'little_endian' or 'big_endian'
           word_size - 4 or 8
           int_hdr - integer PP headers (numpy array)
           real_hdr - real PP headers (numpy array)
           data_type - 'integer' or 'real'
           nwords - number of words to read
              type and nwords should have been returned by
              get_type_and_num_words()
        """
        if data_type == 'integer':
            data = self._get_empty_int_array(nwords)
            ctypes_data = self._get_ctypes_int_array()
        elif data_type == 'real':
            data = self._get_empty_real_array(nwords)
            ctypes_data = self._get_ctypes_real_array()
        else:
            raise ValueError("data_type must be 'integer' or 'real'")

        self.lib.read_record_data.argtypes = [
            CT.c_int,
            CT.c_size_t,
            CT.c_size_t,
            CT.c_int,
            CT.c_int,
            self._get_ctypes_int_array(),
            self._get_ctypes_real_array(),
            CT.c_size_t,
            ctypes_data
        ]

        rv = self.lib.read_record_data(
            fd,
            data_offset,
            disk_length,
            enum_byte_ordering.as_index(byte_ordering),
            word_size,
            int_hdr,
            real_hdr,
            nwords,
            data
        )

        if rv != 0:
            raise umfile.UMFileException("error reading record data")

        return data


if __name__ == "__main__":
    import sys

    c = CInterface()
    fd = os.open(sys.argv[1], os.O_RDONLY)
    file_type = c.detect_file_type(fd)
    print(c.file_type_obj_to_dict(file_type))
    c.set_word_size(file_type)
    info = c.parse_file(fd, file_type)

    for var in info['vars']:
        print("nz = %s, nt = %s" % (var.nz, var.nt))
        for rec in var.recs:
            print(rec.hdr_offset)
            print("data offset", rec.data_offset)
            print("disk length", rec.disk_length)
            print("int hdr", rec.int_hdr)
            print("real hdr", rec.real_hdr)
            data_type, nwords = c.get_type_and_num_words(rec.int_hdr)
            print("data_type = %s nwords = %s" % (data_type, nwords))

            data = c.read_record_data(
                fd,
                rec.data_offset,
                rec.disk_length,
                file_type.byte_ordering,
                file_type.word_size,
                rec.int_hdr,
                rec.real_hdr,
                data_type,
                nwords
            )
            print("data (%s values): %s ... %s" % (
                len(data), data[:10], data[-10:]))
            extra_data_offset, extra_data_length = (
                c.get_extra_data_offset_and_length(
                    rec.int_hdr,
                    rec.data_offset,
                    rec.disk_length
                )
            )
            print("extra data offset: %s" % extra_data_offset)
            print("extra data length: %s" % extra_data_length)
            extra_data = c.read_extra_data(
                fd,
                extra_data_offset,
                extra_data_length,
                file_type.byte_ordering,
                file_type.word_size
            )
            print("extra data (%s bytes) read" % (len(extra_data)))

    print(c.read_header(
        fd,
        info['vars'][0].recs[0].hdr_offset,
        file_type.byte_ordering,
        file_type.word_size
    ))
