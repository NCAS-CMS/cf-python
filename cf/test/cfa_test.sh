# --------------------------------------------------------------------
# Test cfa
# --------------------------------------------------------------------
sample_files=tmp_cfa_dir_in
test_file=tmp_test_cfa.nc
test_dir=tmp_cfa_dir

rm -fr $sample_files $test_dir $test_file
mkdir -p $sample_files

for f in `ls -1 test_file.nc file*.nc *.pp`
do
    ln -sf $PWD/$f $sample_files/$f
done

mkdir -p $test_dir

cfa -vm $sample_files/test_file.nc >/dev/null
cfa --overwrite -o $test_file $sample_files/test_file.nc
cfa --overwrite -d $test_dir  $sample_files/test_file.nc

cfa -vm $sample_files/* >/dev/null
cfa -vc $sample_files/* >/dev/null
cfa -1 -vs $sample_files/file* >/dev/null  

#echo 0
cfa --overwrite -d $test_dir $sample_files/*

#echo 1
rm -f $test_file
cfa -o $test_file $sample_files

#echo 2
rm -f $test_file
cfa -n -o $test_file $sample_files/*

#echo 3
rm -fr $test_dir $test_file $sample_files

