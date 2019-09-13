HEADERS=umfile.h umfileint.h bits/*.h

LIBRARY=umfile.so

CC=gcc
CFLAGS=-Wall -fPIC -g
CPP=gcc -E -P
LD=ld
LDFLAGS=-shared --build-id
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
	$(LD) $(LDFLAGS) -o $@ $(OBJS) --whole-archive $(TYPE_DEP_LIBRARY_PATH)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $<

#bits/protos_sgl.h: bits/type_dep_protos.h
#	$(CPP) -DBUILD_HDR -DSINGLE $< > $@
#
#bits/protos_dbl.h: bits/type_dep_protos.h
#	$(CPP) -DBUILD_HDR -DDOUBLE $< > $@
