STEMS = umfile_test_typedep interpret_header read process_vars \
	debug_dump date_and_time compare levels axes unwgdos

CFLAGS += -I..

SGL_OBJS=$(foreach stem, $(STEMS), $(stem)_sgl.o)
DBL_OBJS=$(foreach stem, $(STEMS), $(stem)_dbl.o)

OBJS = $(SGL_OBJS) $(DBL_OBJS)

LIB = $(TYPE_DEP_LIBRARY)

.PHONY: all clean

all: $(LIB)

clean:
	rm -f $(OBJS) $(LIB)

%_dbl.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -DDOUBLE -o $@ $<

%_sgl.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -DSINGLE -o $@ $<

$(LIB): $(OBJS)
	rm -f $@
	ar r $@ $(OBJS)
