STEMS = umfile_test_typedep interpret_header read process_vars \
	debug_dump date_and_time compare levels axes unwgdos

CFLAGS += -I..

SGL_OBJS=$(foreach stem, $(STEMS), $(stem)_sgl.o)
DBL_OBJS=$(foreach stem, $(STEMS), $(stem)_dbl.o)

SGL_TMP_OBJS=$(foreach stem, $(STEMS), $(stem)_sgl_tmp.o)
DBL_TMP_OBJS=$(foreach stem, $(STEMS), $(stem)_dbl_tmp.o)

TMP_OBJS = $(SGL_TMP_OBJS) $(DBL_TMP_OBJS)
OBJS = $(SGL_OBJS) $(DBL_OBJS)

REDEFINES_DBL = redefs_sgl
REDEFINES_SGL = redefs_dbl

LIB = $(TYPE_DEP_LIBRARY)

.PHONY: all clean

all: $(LIB)

clean:
	rm -f $(OBJS) $(TMP_OBJS) $(LIB) $(REDEFINES_SGL) $(REDEFINES_DBL)

%_dbl_tmp.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -DDOUBLE -o $@ $<

%_sgl_tmp.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -DSINGLE -o $@ $<

%_sgl.o: %_sgl_tmp.o $(REDEFINES_SGL)
	objcopy --redefine-syms=$(REDEFINES_SGL) $< $@

%_dbl.o: %_dbl_tmp.o $(REDEFINES_DBL)
	objcopy --redefine-syms=$(REDEFINES_DBL) $< $@

$(REDEFINES_SGL): $(SGL_TMP_OBJS)
	nm -A $(SGL_TMP_OBJS) | awk '/ T /{print $$3, $$3"_sgl"}' | grep -v thunk > $@

$(REDEFINES_DBL): $(DBL_TMP_OBJS)
	nm -A $(DBL_TMP_OBJS) | awk '/ T /{print $$3, $$3"_dbl"}' | grep -v thunk > $@

$(LIB): $(OBJS)
	rm -f $@
	ar r $@ $(OBJS)
