/* interpret_header.c */
Data_type get_type(const INTEGER *int_hdr);
size_t get_num_data_words (const INTEGER *int_hdr);
size_t get_extra_data_length(const INTEGER *int_hdr);

int var_is_missing(const INTEGER *int_hdr);
int get_var_stash_model(const INTEGER *int_hdr);
int get_var_stash_section(const INTEGER *int_hdr);
int get_var_stash_item(const INTEGER *int_hdr);
int get_var_compression(const INTEGER *int_hdr);
int get_var_gridcode(const INTEGER *int_hdr);
int get_var_packing(const INTEGER *int_hdr);
REAL get_var_real_fill_value(const REAL *int_hdr);

/* read.c */

size_t read_words(int fd, 
		  void *ptr,
		  size_t num_words,
		  Byte_ordering byte_ordering);

int read_extra_data_at_offset(int fd,
			      size_t extra_data_offset,
			      size_t extra_data_length,
			      Byte_ordering byte_ordering, 
			      void *extra_data_rtn);

int read_hdr(int fd,
	     Byte_ordering byte_ordering, 
	     INTEGER *int_hdr_rtn,
	     REAL *real_hdr_rtn);

Rec *get_record(File *file, List *heaplist);
int read_all_headers(File *file, List *heaplist);
size_t skip_fortran_record(File *file);
int skip_word(File *file);
int read_all_headers_pp(File *file, List *heaplist);
int read_all_headers_ff(File *file, List *heaplist);
size_t get_ff_disk_length(INTEGER *ihdr);
int get_valid_records_ff(int fd,
			 Byte_ordering byte_ordering,
			 size_t hdr_start, size_t hdr_size, int nrec,
			 int valid[], int *n_valid_rec_return);
int unpack_run_length_encoded(REAL *datain, INTEGER nin, REAL *dataout, INTEGER nout, REAL mdi);

/* process_vars.c */
int process_vars(File *file, List *heaplist);
int test_skip_var(const Rec *rec);
int initialise_records(Rec **recs, int nrec, List *heaplist);
int get_vars(int nrec, Rec **recs, 
	     List *vars, 
	     List *heaplist);
int set_disambig_index(Z_axis *z_axis, T_axis *t_axis, 
		       Rec **recs, int nvrec, int svindex);
int add_axes_to_var(Var *var, 
		    Z_axis *z_axis, T_axis *t_axis, 
		    List *z_axes, List *t_axes, 
		    List *heaplist);
int grid_supported(INTEGER *int_hdr);
int var_has_regular_z_t(Z_axis *z_axis, T_axis *t_axis, Rec **recs, int nvrec);

/* level.c */
int lev_set(Level *lev, const Rec *rec);
Lev_type level_type(const Rec *rec);

/* date_and_time.c */
REAL mean_period(const Time *time);
int is_time_mean(INTEGER LBTIM);
REAL time_diff(INTEGER lbtim, const Date *date, const Date *orig_date);
REAL sec_to_day(int64_t seconds);
Calendar_type calendar_type(INTEGER type);
int64_t gregorian_to_secs(const Date *date);
int time_set(Time *time, const Rec *rec);

/* axes.c */
Z_axis *new_z_axis(List *heaplist);
int free_z_axis(Z_axis *z_axis, List *heaplist);
T_axis *new_t_axis(List *heaplist);
int free_t_axis(T_axis *t_axis, List *heaplist);
int t_axis_add(T_axis *t_axis, const Time *time, 
	       int *index_return, List *heaplist);
int z_axis_add(Z_axis *z_axis, const Level *lev, 
	       int *index_return, List *heaplist);

/* compare.c */
int compare_records_between_vars(const Rec *a, const Rec *b);
int compare_mean_periods(const Rec *a, const Rec *b);
int compare_records_within_var(const Rec *a, const Rec *b);
int compare_records(const void *p1, const void *p2);
int records_from_different_vars(const Rec *a, const Rec *b);
int compare_lists(const List *l1, const List *l2, int (*compfunc)(const void*, const void*));
int compare_levels(const void *p1, const void *p2);
int compare_times(const void *p1, const void *p2);
int compare_dates(const Date *a, const Date *b);

/* unwgdos.c */
int unwgdos(void *datain, int nbytes, REAL *dataout, int nout, REAL mdi);


/* Debug_dump.c */
void debug_dump_all_headers(File *file);

#ifdef MAIN
int main();
#endif
