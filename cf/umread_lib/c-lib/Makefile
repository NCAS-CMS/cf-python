HEADERS=umfile.h umfileint.h bits/*.h

LIBRARY=umfile.so
CFLAGS=-Wall -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CC=gcc
    LD=ld
    LDFLAGS=-shared --build-id
    CFLAGS += -DLINUX
    LD_ARCHIVE_FLAGS=--whole-archive
endif
ifeq ($(UNAME_S),Darwin)
    CC=clang
    LD=clang
    LDFLAGS=-dynamiclib
    CFLAGS += -DOSX
    LD_ARCHIVE_FLAGS=-force_load
endif

CPP=gcc -E -P
OBJS = umfile.o error.o filetype.o \
	malloc.o linklist.o new_structs.o swap.o

TYPE_DEP_LIBRARY = umfile_typedep.a
TYPE_DEP_DIR = type-dep
TYPE_DEP_LIBRARY_PATH = $(TYPE_DEP_DIR)/$(TYPE_DEP_LIBRARY)

export CC CFLAGS CPP TYPE_DEP_LIBRARY

.PHONY: clean all type-dep

all: $(LIBRARY)

clean:
	rm -f $(OBJS)
	$(MAKE) -C $(TYPE_DEP_DIR) clean

type-dep:
	$(MAKE) -C $(TYPE_DEP_DIR)

$(LIBRARY): $(OBJS) type-dep
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LD_ARCHIVE_FLAGS) $(TYPE_DEP_LIBRARY_PATH)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $<

#bits/protos_sgl.h: bits/type_dep_protos.h
#	$(CPP) -DBUILD_HDR -DSINGLE $< > $@
#
#bits/protos_dbl.h: bits/type_dep_protos.h
#	$(CPP) -DBUILD_HDR -DDOUBLE $< > $@
